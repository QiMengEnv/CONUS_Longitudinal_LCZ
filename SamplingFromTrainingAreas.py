"""
Author: Meng Qi
Last modified: 06/07/2022

This script is to sample historical LCZ labels from Training Areas (TAs);
Feature collection for sampled points will be completed on GEE.

Workflow:
1. First sample at least 2 points from each TA polygon
2. Then randomly sample from polygons to approach the defined number of labels
"""
import time
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
import os


def hist(raw, row_indx, class_range=16):
    """
    Get the distribution of sampled labels
    """
    hist_ = []
    for i in range(class_range):
        hist_.append(np.array([(raw == i).sum()]))
    hist_output = np.array(hist_)
    hist_output = pd.DataFrame(hist_output.reshape(1, -1), index=[row_indx])
    return hist_output


def random_points_in_polygon_first(polygons, label_cols, N_ratio=2):
    """
    First round of sampling; sample a few points from each polygon in each available year
    """
    points = []
    # for each 100m*100m area, sample N point, set N=2
    for i in tqdm(range(polygons.shape[0])):
        polygon_temp = polygons.geometry[i]
        area = polygons['area'][i]
        min_x, min_y, max_x, max_y = polygon_temp.bounds
        n = 0

        # when set N_ratio = 0, it means to sample only one points for each polygon regardless of the polygon size
        if N_ratio == 0:
            sample_num = 1
        else:
            sample_num = np.ceil(area / 10000 * N_ratio)

        while n < sample_num:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if Point(x, y).within(polygon_temp):
                n += 1
                labels = list(polygons[label_cols].iloc[i, :])
                years = [int(year) for year in label_cols]
                gdf_points = gpd.GeoDataFrame({'lcz_class': labels,
                                               'year': years,
                                               'geometry': [Point(x, y)] * len(labels)},
                                              crs="EPSG:4326")
                gdf_points['long'] = gdf_points.geometry.x
                gdf_points['lat'] = gdf_points.geometry.y
                # drop points which don't have valid lcz labels
                points.append(gdf_points[gdf_points['lcz_class'] >= 0])

    random_points = gpd.GeoDataFrame(pd.concat(points, axis=0, ignore_index=True))

    return random_points


def random_points_in_polygon_second(polygons, sampled_points, hist, label_cols, target_N=1500):
    """
    Second round of sampling;
    """
    points = []
    for year in label_cols:
        for lcz_cat in range(hist.shape[1]):
            # retrieve points sampled from the first round
            sampled = sampled_points[(sampled_points['lcz_class'] == lcz_cat) & (sampled_points['year'] == int(year))]
            sample_num = sampled.shape[0]
            print('Processing LCZ {} in {}: {} in first sample round.'.format(lcz_cat, year, sample_num))

            if sample_num > target_N:
                # drop sampled points to target_N
                points.append(sampled.sample(n=target_N))

            else:
                # use all sampled points and continue to sample
                points.append(sampled)
                # start the second round of sampling to meet target_N
                subset = polygons[polygons[year] == lcz_cat]
                subset.reset_index(drop=True, inplace=True)

                # sample one point from each available polygon recursively until find enough points
                while sample_num < target_N:
                    # sample from one polygon and using area as weights
                    polygon_sampled = subset.sample(n=1, weights='area')
                    polygon_sampled.reset_index(drop=False, inplace=True)
                    polygon_temp = polygon_sampled.geometry[0]
                    min_x, min_y, max_x, max_y = polygon_sampled.total_bounds

                    subsample_n = 1
                    while subsample_n:
                        x = np.random.uniform(min_x, max_x)
                        y = np.random.uniform(min_y, max_y)
                        if Point(x, y).within(polygon_temp):
                            sample_num += 1
                            subsample_n -= 1
                            gdf_points = gpd.GeoDataFrame({'lcz_class': lcz_cat,
                                                           'year': int(year),
                                                           'geometry': [Point(x, y)]},
                                                          crs="EPSG:4326")
                            gdf_points['long'] = gdf_points.geometry.x
                            gdf_points['lat'] = gdf_points.geometry.y
                            # drop points which don't have valid lcz labels
                            points.append(gdf_points[gdf_points['lcz_class'] >= 0])

    random_points = gpd.GeoDataFrame(pd.concat(points, axis=0, ignore_index=True))

    return random_points


start = time.time()

confidence_level = 'highplus'  # including: 'highplus', 'mediumplus' and 'lowplus'
area = 200  # sample area, could be 500*500, 400*400, ..., 100*100
N_ratio = 0  # for each 100m*100m area, sample N_ratio point; if N_ratio=0, it means to sample only one point for each polygon
if N_ratio == 0:
    print('Sampling 1 point for each polygon.')
else:
    print('Sampling {} points for every 100*100 area.'.format(N_ratio))
target_N = 400  # the number of labels to sample;

output_path = './TrainingAreas/AllMTurk_Batch123_'+confidence_level+'_targetN' + str(target_N) + '/' + \
              'Batches_' + str(area) + 'm2_' + str(target_N) + '/'
os.makedirs(output_path, exist_ok=True)

hist_file_1 = output_path + 'sampling_hist_1.csv'
hist_file_2 = output_path + 'sampling_hist_2.csv'
label_csv_file = output_path + 'balanced_points_sampled.csv'
label_shp_file = output_path + 'balanced_points_sampled.shp'

# Randomly sample points from 3 rounds of TA polygons
lcz_label_folder = './TrainingAreas/'

# only choose useful data for each round of polygons before merging different rounds
label_cols = [str(year) for year in range(2020, 1985, -1)]
meta_cols = ['Agreement', 'Confidence', 'geometry']

# sample from round 1 data
lcz_polygons_path1 = lcz_label_folder + 'Round1/' + \
                     'AllMTurk_' + str(area) + 'm2_70_0709_historical_' + confidence_level + '.shp'
print('Sampling from {}'.format(os.path.basename(lcz_polygons_path1)))
lcz_polygons1 = gpd.read_file(lcz_polygons_path1)
# clean
lcz_polygons1 = lcz_polygons1[meta_cols + label_cols]

# sample from round 2 data
lcz_polygons_path2 = lcz_label_folder + 'Round2/' + \
                     'MTurk_' + str(area) + 'm2_70_0616_historical_' + confidence_level + '.shp'
print('Sampling from {}'.format(os.path.basename(lcz_polygons_path2)))
lcz_polygons2 = gpd.read_file(lcz_polygons_path2)
if area == 500:
    # The column names is correct, e.g., '2019', '2017', etc
    lcz_polygons2 = lcz_polygons2[meta_cols + label_cols]
else:
    # The column names in round 2 is 'F2019', 'F2017', etc
    lcz_polygons2 = lcz_polygons2[meta_cols + ['F' + y for y in label_cols]]
    # reset column names
    lcz_polygons2.columns = meta_cols + label_cols

# sample from round 3 data
if confidence_level == 'high':
    lcz_polygons_path3 = lcz_label_folder + 'Round3/' + \
                         'LCZ02_EarlySamples_' + str(area) + 'm2.shp'
    print('Sampling from {}'.format(os.path.basename(lcz_polygons_path3)))
    lcz_polygons3 = gpd.read_file(lcz_polygons_path3)
    lcz_polygons3 = lcz_polygons3[meta_cols + ['F' + y for y in label_cols]]
    # reset column names
    lcz_polygons3.columns = meta_cols + label_cols

    # merge round 1-2-3
    lcz_polygons = pd.concat([lcz_polygons1, lcz_polygons2, lcz_polygons3])

else:
    # merge round 1-2
    lcz_polygons = pd.concat([lcz_polygons1, lcz_polygons2])  # the third round are all high confidence


# reset index
lcz_polygons.reset_index(drop=True, inplace=True)
# add a new column for area
lcz_polygons['area'] = area * area


# start sampling
print('Start first round of random sampling...')
print('This round is to make sure each polygon have at least {} samples.'.format(N_ratio if N_ratio != 0 else 1))

label_points = random_points_in_polygon_first(lcz_polygons, label_cols, N_ratio=N_ratio)

# generate lcz distribution to decide the second round of sampling
# get distribution for each year
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
print('Generating LCZ distribution for first sampling...')
label_points_hist = []
for year in label_cols:
    label_points_hist.append(hist(label_points[label_points['year'] == int(year)]['lcz_class'],
                                  year, class_range=16))

# export the distribution of each dataset
hist_output = pd.concat(label_points_hist, axis=0)
print('Distribution of first sampling\n', hist_output.T)
# add column names for hist output for better visualization
hist_output.columns = lcz_names
hist_output.to_csv(hist_file_1)

# begin the second round of sampling
label_points_final = random_points_in_polygon_second(lcz_polygons, label_points, hist_output, label_cols,
                                                     target_N=target_N)
features = label_points_final.columns.tolist()
print('Finish sampling. A total of {} points are sampled.'.format(label_points_final.shape[0]))

# add county information
# load boundary file
gis_path = './CensusData/tl_2020_us_county.shp'
boundary = gpd.read_file(gis_path)
base = boundary.to_crs('epsg:4326')
# choose the desired metadata
bound_feature = ['GEOID']
features = features + bound_feature

# spatial join
sjoined = gpd.sjoin(label_points_final, base, how="inner", op='intersects')
print('After spatial join, the final data shape is {}.'.format(sjoined.shape))

# add a new column to save ID number
sjoined.reset_index(drop=False, inplace=True)
sjoined.rename(columns={'index': 'FID'}, inplace=True)

# save to shapefile
features.append('FID')
sjoined = sjoined[features]
sjoined.to_file(label_shp_file)

# save to csv file
features.remove('geometry')
sjoined = sjoined[features]
sjoined.rename(columns={'long': 'longitude', 'lat': 'latitude'}, inplace=True)
sjoined.to_csv(label_csv_file, index=False)

print('Generating LCZ distribution for final samples...')
label_points_hist_final = []
for year in label_cols:
    label_points_hist_final.append(hist(sjoined[sjoined['year'] == int(year)]['lcz_class'],
                                        year, class_range=16))

# export the distribution of each dataset
hist_output_final = pd.concat(label_points_hist_final, axis=0)
print('Distribution of first sampling\n', hist_output_final.T)
# add column names for hist output for better visualization
hist_output_final.columns = lcz_names
hist_output_final.to_csv(hist_file_2)

end = time.time()
print('Completed! Use {} min.'.format((end - start) / 60))

