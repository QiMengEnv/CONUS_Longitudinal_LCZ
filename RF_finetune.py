"""
Author: Meng Qi
Last modified: 07/23/2022

This script is to fine tune a RF model for LCZ;
Will use the fine-tuned hyper parameters to develop GEE models;

"""
import pandas as pd
import numpy as np
from numpy import mean, std
import os
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
import argparse
import joblib


def evaluate(y_test, y_test_pred, lcz_names, oaw):
    """
    Evaluate model performance; Evaluation metrics are adapted from Matthias et al., 2020
    :param y_test: ground truth
    :param y_test_pred: predicted value
    :param lcz_names: lcz classes
    :param oaw: LCZ weighted accuracy
    :return: model performance including acc, f1, cm, oaf
    """
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    # print confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(16))
    cm_df = pd.DataFrame(cm, columns=lcz_names, index=lcz_names)
    # rearrange the order of rows and columns
    cm_df = pd.concat([cm_df.drop(columns=['lcz_G_water']),
                       cm_df[['lcz_G_water']]],
                      axis=1)
    cm_df = pd.concat([cm_df.drop(index=['lcz_G_water']),
                       cm_df.loc[['lcz_G_water']]],
                      axis=0)

    # calculate metrics following Matthias et al., 2020
    farr = cm_df.to_numpy()
    # start calculation
    diag = farr.diagonal()
    # urban classes (Note: have excluded LCZ 7)
    diagOaurb = farr[:9, :9].diagonal()

    sumColumns = farr.sum(0)
    sumRows = farr.sum(1)

    sumDiag = diag.sum()
    sumDiagOaurb = diagOaurb.sum()

    sumTotal = farr.sum()
    sumTotalOaurb = farr[:9, :9].sum()  # because no lcz 7

    # weighted cm, https://www.mdpi.com/2072-4292/12/11/1769
    # The weighted accuracy is thus defined as
    # the sum of the element-wise products of the similarity weight matrix (wij) and the confusion matrix (cij)
    # divided by the sample size N
    cmw = oaw * farr
    sumOAWTotal = np.nansum(cmw)

    ua = diag / sumColumns  # UA or Precision
    pa = diag / sumRows  # PA or Recall

    oaf = np.zeros(20)
    oaf[0] = sumDiag / sumTotal  # OA
    oaf[1] = sumDiagOaurb / sumTotalOaurb  # OA_urb
    # OA_buitup, which is the OA of the classification in two classes only, i.e., urban and natural.
    # LCZ E is omitted, since it can be in both
    oaf[2] = (farr[:9, :9].sum() + farr[[9, 10, 11, 12, 14, 15], [9, 10, 11, 12, 14, 15]].sum()) / \
             (farr[:9, [9, 10, 11, 12, 14, 15]].sum() + farr[[9, 10, 11, 12, 14, 15], :9].sum() +
              farr[:9, :9].sum() + farr[[9, 10, 11, 12, 14, 15], [9, 10, 11, 12, 14, 15]].sum())

    oaf[3] = sumOAWTotal / sumTotal  # OA_weighted
    oaf[4:] = 2 * ((pa * ua) / (pa + ua))

    logger.info('\nOA:{}\nOA_urb：{}\nOA_bu: {}\nOA_weighted: {}'.format(oaf[0], oaf[1], oaf[2], oaf[3]))

    cm_df['Precision'] = ua
    cm_df['recall'] = pa
    cm_df['F1'] = oaf[4:]

    return acc, f1, cm_df, oaf


def get_parser():
    parser = argparse.ArgumentParser(description='LCZ_RandomForest')
    parser.add_argument('--TRAIN_path', type=str, nargs='?', default='/home/LCZ/')
    parser.add_argument('--area', type=int, nargs='?', default=180)
    parser.add_argument('--file_prefix', type=str, nargs='?', default='MTurk_Historical_')
    parser.add_argument('--output_folder', type=str, nargs='?', default='Result/TP/')
    parser.add_argument('--scenario', type=int, nargs='?', default=1)
    parser.add_argument('--group', type=str, nargs='?', default='grid_ID')
    parser.add_argument('--window_list', nargs='+', type=str, default=['mean', 'max', 'min', 'median', 'p25', 'p75'])
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--oaw_file', type=str, nargs='?', default='LCZ_metric_weighted_accuracy.csv')

    args = parser.parse_args()
    config = {}
    config['TRAIN_path'] = args.TRAIN_path
    config['area'] = args.area
    config['file_prefix'] = args.file_prefix
    config['output_folder'] = args.output_folder
    config['scenario'] = args.scenario
    config['group'] = args.group
    config['window_list'] = args.window_list
    config['save_model'] = args.save_model
    config['oaw_file'] = args.oaw_file

    return config


def hist(train_y, lcz_names, row_indx, class_range=16):
    """
    return the distribution of lcz classes
    """
    hist_ = []
    for i in range(class_range):
        hist_.append(np.array([(train_y == i).sum()]))
    hist_output = np.array(hist_)
    hist_output = pd.DataFrame(hist_output.reshape(1, -1), columns=lcz_names, index=[row_indx])
    return hist_output


def get_logger(logger_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_name, mode='a')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()  # set up for console
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    global args, logger
    args = get_parser()

    lcz_names = ['lcz_1_compact highrise',
                 'lcz_2_compact midrise',
                 'lcz_3_compact lowrise',
                 'lcz_4_open highrise',
                 'lcz_5_open midrise',
                 'lcz_6_open lowrise',
                 'lcz_G_water',
                 'lcz_8_large lowrise',
                 'lcz_9_sparsly built',
                 'lcz_10_heavy industry',
                 'lcz_A_dense trees',
                 'lcz_B_scattered trees',
                 'lcz_C_bush srub',
                 'lcz_D_low plants',
                 'lcz_E_bare rock or paved',
                 'lcz_F_bare soil or sand']

    oaw = pd.read_csv(args['oaw_file']).to_numpy()

    # load training and test data set
    inputfile_prefix = args['TRAIN_path'] + args['file_prefix'] + str(args['area'])
    raw = pd.read_csv(inputfile_prefix + '_ALL.csv')

    # set scenario
    meta_var = ['lcz_class'] + [args['group'], 'FID', 'longitude', 'latitude']
    composite_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tirs1', 'bci', 'ndbai', 'ndvi_min', 'ndvi_max',
                       'ndwi_max']
    composite_var = [band + '_' + texture for band in composite_bands for texture in args['window_list']]
    LCMAP_var = ['LCPRI_' + str(i + 1) for i in range(8)] + ['LCSEC_' + str(i + 1) for i in range(8)]
    LCMS_var = ['Land_Cover_' + str(i + 1) for i in range(15)] + ['Land_Use_' + str(i + 1) for i in range(7)]
    TP_var = ['TP', 'TP_dens']
    EMPL_var = ['C000'] + ['CNS' + str(i + 1).zfill(2) for i in range(20)]
    EMPL_var = EMPL_var + [var + '_dens' for var in EMPL_var]

    # set output folder and files
    output_path = args['TRAIN_path'] + args['output_folder']
    os.makedirs(output_path, exist_ok=True)

    # set scenario
    scenario_sets = ['_TP_NoYear', '_TP_EMPL_NoYear', '_TP', '_TP_EMPL']
    scenario = args['scenario']
    if scenario == 1:
        feature_name = composite_var + LCMAP_var + LCMS_var + TP_var
    elif scenario == 2:
        feature_name = composite_var + LCMAP_var + LCMS_var + TP_var + EMPL_var
    elif scenario == 3:
        LCMAP_var = LCMAP_var + ['Year']
        feature_name = composite_var + LCMAP_var + LCMS_var + TP_var
    else:
        LCMAP_var = LCMAP_var + ['Year']
        feature_name = composite_var + LCMAP_var + LCMS_var + TP_var + EMPL_var
    col_names = meta_var + feature_name

    sets = scenario_sets[args['scenario'] - 1]
    logfile = output_path + 'log' + sets + '.txt'
    logger = get_logger(logfile)
    # print spatial resolution
    logger.info('processing area: {} by {}m'.format(str(args['area']), str(args['area'])))
    logger.info(
        'Building models for scenario {}: {} with {} variables in total.'.format(scenario, sets[1:], len(feature_name)))
    logger.info('vars: {}'.format(col_names))

    outputfile_prefix = output_path + args['file_prefix'] + str(args['area']) + sets
    cm_save_path = outputfile_prefix + '_cm.csv'
    hist_savefile = outputfile_prefix + '_hist.csv'
    prob_savefile = outputfile_prefix + '_prob.csv'
    summary_eval_save_path = outputfile_prefix + '_summary_eval.csv'
    year_eval_save_path = outputfile_prefix + '_yearly_eval.csv'
    featureImp_save_path = outputfile_prefix + '_FeatureImportance_Group'
    #########################################################################################################
    # drop nan rows and save the updated train/test/raw files for GEE uploading afterwards
    raw = raw[col_names]
    raw = raw.dropna(axis=0)
    raw = raw.reset_index(drop=True)

    # split data into train/test files
    groups = np.array(raw[args['group']])
    X, y = np.array(raw[feature_name]), np.array(raw['lcz_class'])
    gkf = StratifiedGroupKFold(n_splits=5)

    # enumerate splits
    outer_results_acc = list()
    outer_results_f1 = list()

    outer_results_OA = list()
    outer_results_OA_urb = list()
    outer_results_OA_bu = list()
    outer_results_OA_weighted = list()

    best_f1 = np.NINF
    best_acc = np.NINF
    best_cm = []
    best_param = []
    nround = 0

    # Create random hyperparameter grid
    space = dict()
    # Number of trees in random forest
    space['n_estimators'] = [int(x) for x in np.arange(start=10, stop=60, step=10)]
    # Number of features to consider at every split
    space['max_features'] = ['sqrt', 'log2']
    # Maximum number of levels in tree
    space['max_depth'] = [int(x) for x in np.arange(start=10, stop=110, step=10)]
    space['max_depth'].append(None)
    # Minimum number of samples required to split a node
    space['min_samples_split'] = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70]
    # Minimum number of samples required at each leaf node
    space['min_samples_leaf'] = [1, 2, 4, 8, 10, 15, 20, 30, 40, 50]
    # Method of selecting samples for training each tree
    space['bootstrap'] = [True, False]

    hist_data = []
    hist_data.append(hist(y, lcz_names, row_indx='Raw', class_range=16))

    outer_prob = []
    for train_index, test_index in gkf.split(X, y, groups):
        nround += 1
        logger.info("Building models for group {}".format(nround))
        # split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # get the histogram for train and test set
        hist_data.append(hist(y_train, lcz_names, row_indx='train_' + str(nround), class_range=16))
        hist_data.append(hist(y_test, lcz_names, row_indx='test_' + str(nround), class_range=16))

        # configure the cross-validation procedure
        groups_train = np.array(raw.loc[train_index, args['group']])
        cv_inner = StratifiedGroupKFold(n_splits=3)

        #######################################################################################
        # set base model for comparison
        model_RandomSearch = RandomForestClassifier(n_estimators=10, random_state=42)

        # start building models and randomly search hyperparameters
        # define search
        model_RandomSearch_list = RandomizedSearchCV(estimator=model_RandomSearch,
                                                     param_distributions=space,
                                                     n_iter=20,
                                                     scoring='f1_weighted',
                                                     cv=cv_inner,
                                                     verbose=2,
                                                     random_state=100,
                                                     n_jobs=-1)
        # Fit the random search model
        model_RandomSearch_list.fit(X_train, y_train, groups=groups_train)

        # print best hyperparameter and best model
        best_RandomSearch_model = model_RandomSearch_list.best_estimator_
        best_RandomSearch_model_param = model_RandomSearch_list.best_params_
        logger.info('Best model parameters by Random Search Training for group {}: {}'.format(nround,
                                                                                              best_RandomSearch_model_param))
        # evaluate the best model by random search
        y_pred = best_RandomSearch_model.predict(X_test)
        acc_RandomSearch, f1_RandomSearch, cm_RandomSearch, oaf_RandomSearch = evaluate(y_test, y_pred, lcz_names, oaw)

        # store the result
        outer_results_acc.append(acc_RandomSearch)
        outer_results_f1.append(f1_RandomSearch)

        outer_results_OA.append(oaf_RandomSearch[0])
        outer_results_OA_urb.append(oaf_RandomSearch[1])
        outer_results_OA_bu.append(oaf_RandomSearch[2])
        outer_results_OA_weighted.append(oaf_RandomSearch[3])

        # export probability on test set
        y_prob = best_RandomSearch_model.predict_proba(X_test)
        if len(np.unique(y_train)) != 16:
            # find the missing class
            miss_class = np.setdiff1d(np.arange(16), np.unique(y_train))
            logger.info('miss class in training dataset: {}'.format(miss_class))
            y_prob = np.insert(y_prob, miss_class, 0, axis=1)

        y_prob_max = np.argmax(y_prob, axis=1)  # for quick check in
        y_prob_output = pd.DataFrame(np.concatenate((y_test.reshape(-1, 1),
                                                     y_pred.reshape(-1, 1),
                                                     raw.loc[test_index, ['Year', 'longitude',
                                                                          'latitude']].to_numpy(),
                                                     y_prob_max.reshape(-1, 1),
                                                     y_prob), axis=1),
                                     columns=['GT', 'Pred', 'Year', 'longitude', 'latitude',
                                              'max_prob'] + lcz_names)
        # add group info
        y_prob_output['Group'] = nround
        outer_prob.append(y_prob_output)

        # show feature importance
        feature_importances = pd.DataFrame({
            "features": feature_name,
            "score": best_RandomSearch_model.feature_importances_})

        # check feature importance
        feature_importances.sort_values("score", ascending=False, inplace=True)
        # save feature importance
        feature_importances.to_csv(featureImp_save_path + str(nround) + '.csv', index=False)

        # Print out the feature and importances
        logger.info('Print the top 10 important features')
        logger.info(feature_importances.head(10))

        # choose the model with best f1_weighted score as the best model
        # check if this gets the best model result
        if f1_RandomSearch > best_f1:
            # update best_f1
            best_f1 = f1_RandomSearch
            best_acc = acc_RandomSearch
            best_param = best_RandomSearch_model_param

    # export the distribution of each dataset
    hist_output = pd.concat(hist_data, axis=0)
    hist_output.to_csv(hist_savefile)

    # save prob
    prob_output = pd.concat(outer_prob, axis=0)
    prob_output.to_csv(prob_savefile, index=False)

    # calculate evaluation metrics using the concatenated test results
    logger.info('Evaluate concatenated test predictions!')
    acc_concat, f1_concat, cm_concat, oaf_concat = evaluate(prob_output['GT'], prob_output['Pred'], lcz_names, oaw)

    # save cm
    cm_concat.to_csv(cm_save_path)

    # summarize the estimated performance of the model
    # save summarized results
    summary_dict = {'Accuracy': outer_results_acc,
                    'F1_weighted': outer_results_f1,
                    'OA': outer_results_OA,
                    'OA_urb': outer_results_OA_urb,
                    'OA_bu': outer_results_OA_bu,
                    'OA_weighted': outer_results_OA_weighted
                    }
    summary_eval = pd.DataFrame(summary_dict)
    summary_eval.loc[len(outer_results_acc)] = [acc_concat, f1_concat,
                                                oaf_concat[0], oaf_concat[1],
                                                oaf_concat[2], oaf_concat[3]]
    summary_eval.loc[len(outer_results_acc) + 1] = [None for i in range(len(summary_dict))]
    summary_eval.loc[len(outer_results_acc) + 2] = [None for i in range(len(summary_dict))]
    summary_eval.insert(0, 'Group', ['Group1', 'Group2', 'Group3', 'Group4', 'Group5',
                                     'Group_concat', 'Group_mean', 'Group_std'])

    logger.info('Evaluate group stats!')
    for key, value in summary_dict.items():
        metric_mean, metric_std = mean(value), std(value)
        summary_eval.loc[len(outer_results_acc)+1, key] = metric_mean
        summary_eval.loc[len(outer_results_acc)+2, key] = metric_std
        logger.info('Group Summary - {0} (mean, std): {1:.2f}, {2:.2f}'.format(key, metric_mean, metric_std))

    # SAVE THE SUMMARY RESULT
    summary_eval.to_csv(summary_eval_save_path, index=False)

    # report the best model
    logger.info('Choose best group model!')
    logger.info('Best group model accuracy: {:5.2f}%; f1_weighted: {:5.2f}%'.format(best_acc * 100, best_f1 * 100))
    logger.info('Best group model param: {}'.format(best_param))

    # check model performance for each year
    year_list = list(set(prob_output['Year'].tolist()))
    year_list.sort()
    year_eval = pd.DataFrame(columns=['Year', 'Acc', 'F1_weighted',
                                      'OA', 'OA_urb', 'OA_bu', 'OA_weighted'])
    for year in year_list:
        logger.info('Evaluating concatenated test set for year {}...'.format(int(year)))
        prob_temp = prob_output[prob_output['Year'] == year]
        acc_year, f1_year, cm_year, oaf_year = evaluate(prob_temp['GT'], prob_temp['Pred'], lcz_names, oaw)
        logger.info(
            'Model accuracy, f1_weighted for year {}: {:5.2f}%, {:5.2f}%'.format(int(year), acc_year * 100,
                                                                                 f1_year * 100))
        cm_year.to_csv(cm_save_path[:-4] + '_' + str(int(year)) + '.csv')
        year_eval = year_eval.append({'Year': year,
                                      'Acc': acc_year,
                                      'F1_weighted': f1_year,
                                      'OA': oaf_year[0],
                                      'OA_urb': oaf_year[1],
                                      'OA_bu': oaf_year[2],
                                      'OA_weighted': oaf_year[3]
                                      }, ignore_index=True)

    # save yearly result
    year_eval.to_csv(year_eval_save_path, index=False)

    #######################################################################################
    # apply the same method to the whole training set
    # set base model for comparison
    model_whole = RandomForestClassifier(n_estimators=10, random_state=42)

    # start building models and randomly search hyperparameters
    # define search
    model_whole_list = RandomizedSearchCV(estimator=model_whole,
                                          param_distributions=space,
                                          n_iter=20,
                                          scoring='f1_weighted',
                                          cv=gkf,
                                          verbose=2,
                                          random_state=100,
                                          n_jobs=-1)
    # Fit the random search model
    model_whole_list.fit(X, y, groups=groups)

    # print best hyperparameter and best model
    best_whole_model = model_whole_list.best_estimator_
    best_whole_model_param = model_whole_list.best_params_
    best_whole_model_score = model_whole_list.best_score_

    logger.info('Whole model param: {}'.format(best_whole_model_param))
    logger.info('Whole model best score: {}'.format(best_whole_model_score))

    # save the model to disk
    if args['save_model'] == True:
        filename = output_path + 'final_model' + sets + '.pkl'
        joblib.dump(best_whole_model, filename)


if __name__ == '__main__':
    main()
