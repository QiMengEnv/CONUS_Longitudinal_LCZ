"""
Author: Meng Qi
Last modified: 07/23/2022

This script is to generate prediction surface using GEE RF models
"""
from tqdm import tqdm
import ee
ee.Initialize()


# define a function to add layers
def StackLayers(image, previous):
    return ee.Image(previous).addBands(image)


# kernel size 7 by 7 pixels (210m by 210m window size)
kernel_size = ee.Number(7)
kernel_radius = kernel_size.divide(ee.Number(2))  # here radius should be half of the kernel size
kernel = ee.Kernel.square(radius=kernel_radius,
                          units='pixels',
                          normalize=False)
# set the boundary for export
conus = ee.FeatureCollection("users/meng_ee/boundary/us_2020")

# set training folder
train_folder = "users/meng_ee/LCZ_Training/"

# load point locations for model development
all_points = ee.FeatureCollection(train_folder + 'label/MTurk_Historical_GEE_210_ALL')

# ================== Choose Model Type, etc. ==================
model_type = 'TP'
# model_type = 'EMPL'
group = 'grid_1000'

# TP model hyper parameters
n_estimators = 50
min_samples_leaf = 4
bagFraction = 1
maxNodes = None
# ========================set variables=====================================
composite_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tirs1', 'bci', 'ndbai', 'ndvi_min', 'ndvi_max',
                   'ndwi_max']
composite_bands = [band + '_' + texture for band in composite_bands for texture in
                   ['mean', 'max', 'min', 'median', 'p25', 'p75']]

LCMAP_var = ['LCPRI_' + str(i + 1) for i in range(8)] + ['LCSEC_' + str(i + 1) for i in range(8)]
LCPRI_name = ee.List(['LCPRI_' + str(i + 1) for i in range(8)])
LCSEC_name = ee.List(['LCSEC_' + str(i + 1) for i in range(8)])
LCMAP_bands = LCPRI_name.cat(LCSEC_name)

LCMS_var = ['Land_Cover_' + str(i + 1) for i in range(15)] + ['Land_Use_' + str(i + 1) for i in range(7)]
Land_Cover_name = ee.List(['Land_Cover_' + str(i + 1) for i in range(15)])
Land_Use_name = ee.List(['Land_Use_' + str(i + 1) for i in range(7)])
LCMS_bands = Land_Cover_name.cat(Land_Use_name)

TP_var = ['TP', 'TP_dens']
EMPL_var = ['C000'] + ['CNS' + str(i + 1).zfill(2) for i in range(20)]
EMPL_var = EMPL_var + [var + '_dens' for var in EMPL_var]
tp_list = ee.List(TP_var)
empl_list = ee.List(EMPL_var)

if model_type == 'TP':
    bands = ee.List(composite_bands + LCMAP_var + LCMS_var + TP_var + ['Year'])
else:
    bands = ee.List(composite_bands + LCMAP_var + LCMS_var + TP_var + EMPL_var + ['Year'])

# filter out {} rows
all_points = all_points.filter(
    ee.Filter.notNull(bands)
)


# ========================================MODEL DEVELOPMENT=======================================================
label = 'lcz_class'

# Make a Random Forest classifier and train it.
classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=n_estimators,  # n_estimators
    variablesPerSplit=None,  # max_features;  If unspecified, uses the square root of the number of variables.
    minLeafPopulation=min_samples_leaf,  # min_samples_leaf=20
    bagFraction=bagFraction,  # bootstrap=True
    maxNodes=maxNodes  # Not the max_depth! max_leaf_nodes?
).train(
    features=all_points,
    classProperty=label,
    inputProperties=bands
)

#####################################################/
# prepare input features for model prediction

pred_tasks = []
for year in tqdm(range(2020, 1985, -1)):

    save_file = train_folder + 'prediction/' + model_type + '_' + str(year)

    #####################################################################################
    # Compute 6 statistics as texture of the composite images with the given kernel size.

    # define reducers for focal statistics
    composite_reducers = ee.Reducer.mean() \
        .combine(ee.Reducer.max(), "", True) \
        .combine(ee.Reducer.min(), "", True) \
        .combine(ee.Reducer.median(), "", True) \
        .combine(ee.Reducer.percentile([25]), "", True) \
        .combine(ee.Reducer.percentile([75]), "", True)

    # compute focal statistics for composite images
    composite_layer = ee.Image("users/meng_ee/CONUS_Composite_30/CONUS_" + str(year))

    stacked_composite = composite_layer.reduceNeighborhood(
        reducer=composite_reducers,
        kernel=kernel,
    ).round().toInt32()  # round and change the data format to singed int32

    #####################################################################################
    # processing LCMAP raster

    # LCMAP raster
    LCPRI = ee.Image("users/meng_ee/LCMAP/LCMAP_CU_" + str(year) + "_V12_LCPRI")
    LCSEC = ee.Image("users/meng_ee/LCMAP/LCMAP_CU_" + str(year) + "_V12_LCSEC")

    # extract each LCMAP class and generate new bands
    LCPRI = LCPRI.eq([1, 2, 3, 4, 5, 6, 7, 8]).rename(LCPRI_name)
    LCSEC = LCSEC.eq([1, 2, 3, 4, 5, 6, 7, 8]).rename(LCSEC_name)

    # stack LCPRI and LCSEC
    stacked_LCMAP = LCPRI.addBands(LCSEC)

    # count the number of each LCMAP type given the kernel size
    # multiply 100 to save to int format later
    stacked_LCMAP = stacked_LCMAP.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=kernel
    ).rename(LCMAP_bands) \
        .divide(kernel_size.multiply(kernel_size)) \
        .multiply(ee.Number(100)) \
        .round().toInt32()

    #####################################################################################
    # processing LCMS data

    LCMS_dataset = ee.ImageCollection('USFS/GTAC/LCMS/v2021-7')

    LCMS_Land_Cover = LCMS_dataset.filter(ee.Filter.And(
        ee.Filter.date(str(year), str(year + 1)),
        ee.Filter.eq('study_area', 'CONUS')
    )) \
        .first() \
        .select(['Land_Cover'])

    LCMS_Land_Use = LCMS_dataset.filter(ee.Filter.And(
        ee.Filter.date(str(year), str(year + 1)),
        ee.Filter.eq('study_area', 'CONUS')
    )) \
        .first() \
        .select(['Land_Use'])

    # extract each LCMAP class and generate new bands
    Land_Cover = LCMS_Land_Cover.eq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).rename(Land_Cover_name)
    Land_Use = LCMS_Land_Use.eq([1, 2, 3, 4, 5, 6, 7]).rename(Land_Use_name)

    # stack Land_Cover and Land_Use
    stacked_LCMS = Land_Cover.addBands(Land_Use)

    # count the number of each LCMS type given the kernel size
    # multiply 100 to save to int format later
    stacked_LCMS = stacked_LCMS.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=kernel
    ).rename(LCMS_bands) \
        .divide(kernel_size.multiply(kernel_size)) \
        .multiply(ee.Number(100)) \
        .round().toInt32()

    #####################################################################################
    # processing census data

    # load TP layers
    if year < 1990:
        TP = ee.FeatureCollection("users/meng_ee/census_scaled_top1/CONUS_TP_scaled_1990")
    elif year > 2018:
        TP = ee.FeatureCollection("users/meng_ee/census_scaled_top1/CONUS_TP_scaled_2018")
    else:
        TP = ee.FeatureCollection("users/meng_ee/census_scaled_top1/CONUS_TP_scaled_" + str(year))

    # convert census polygons to raster
    # define a function to convert the polygon to a raster according the selected attribute
    def tpToRaster(col):
        new_layer = TP.reduceToImage(
            properties=ee.List([col]),
            reducer=ee.Reducer.first()
        ).rename(ee.List([col]))
        return ee.List(ee.Image(new_layer))

    # convert all columns needed to separate raster
    tp = tp_list.map(tpToRaster)

    # prepare an initial layer for the iterate function later. This is a temporary layer and will be deleted.
    Init = ee.Image.constant(0).rename('temp')

    # ======= stack processed census raster and remove 'temp' band
    stacked_tp = ee.Image(tp.iterate(StackLayers, Init)).select(tp_list)

    # replace null as zero
    stacked_tp = stacked_tp.unmask(0)

    # collect EMPL features if applicable
    if year < 2002:
        stacked_census = stacked_tp
    else:
        if year > 2019:
            # if year == 2020, use 2019 empl layer
            EMPL = ee.FeatureCollection("users/meng_ee/census_scaled_top1/CONUS_EMPL_scaled_2019")
        else:
            # load EMPL layers
            EMPL = ee.FeatureCollection("users/meng_ee/census_scaled_top1/CONUS_EMPL_scaled_" + str(year))

        def emplToRaster(col):
            new_layer = EMPL.reduceToImage(
                properties=ee.List([col]),
                reducer=ee.Reducer.first()
            ).rename(ee.List([col]))
            return ee.List(ee.Image(new_layer))

        # convert all columns needed to separate raster
        empl = empl_list.map(emplToRaster)
        stacked_empl = ee.Image(empl.iterate(StackLayers, Init)).select(empl_list)

        # replace null as zero
        stacked_empl = stacked_empl.unmask(0)

        # mask pixels with -999
        # note: must first unmask, then remove -999 cells
        mask = stacked_empl.select('C000').neq(-999)
        stacked_empl = stacked_empl.updateMask(mask)

        # ====== stack all census layers
        stacked_census = stacked_tp.addBands(stacked_empl)

    #####################################################################################
    # stack processed composite images, LCMAP, LCMS and census images
    stacked_all = stacked_composite.addBands(stacked_LCMAP)
    stacked_all = stacked_all.addBands(stacked_LCMS)
    stacked_all = stacked_all.addBands(stacked_census)

    # add layer for year information
    year_layer = ee.Image.constant(year).rename('Year')
    stacked_all = stacked_all.addBands(year_layer)

    #####################################################################################

    # model prediction
    # export data

    beforeRemap = [0, 1, 2, 3, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13, 14, 15]
    afterRemap =  [1, 2, 3, 4, 5, 6, 17, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    pred_layer = stacked_all.clip(conus) \
        .reproject(crs='EPSG:5070',
                   scale=30) \
        .classify(classifier) \
        .remap(beforeRemap, afterRemap)

    pred_export = ee.batch.Export.image.toAsset(
        image=pred_layer,
        description=model_type + '_' + str(year),
        assetId=save_file,
        scale=100,
        region=conus.geometry(),
        maxPixels=1e13
    )

    pred_tasks.append(pred_export)
    pred_export.start()
