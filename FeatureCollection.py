"""
Author: Meng Qi
Last modified: 07/19/2022

This script is to collect predictors for sampled points from GEE and export the results to Google Drive Folder
Need to merge the feature collection later

"""
from tqdm import tqdm
import ee

ee.Initialize()


# define a function to add layers
def StackLayers(image, previous):
    return ee.Image(previous).addBands(image)


# define a function to extract focal statistics
def focal_stat(image, kernel_radius, train_points):
    image_neig = image.neighborhoodToArray(
        ee.Kernel.square(radius=kernel_radius,
                         units='pixels',
                         normalize=False))

    image_neig_collection = image_neig.reduceRegions(collection=train_points,
                                                     reducer=ee.Reducer.first())
    return image_neig_collection


# set training folder
train_folder = "users/meng_ee/LCZ_Training/"

# load point locations for model development
train_points_all = ee.FeatureCollection(train_folder + "balanced_points_sampled")

# kernel size 7 by 7 pixels (210m by 210m window size)
kernel_size = ee.Number(7)
kernel_radius = kernel_size.divide(ee.Number(2))  # here radius should be half of the kernel size

# loop over years
composite_tasks = []
LCMAP_tasks = []
LCMS_tasks = []
census_tasks = []

year_avail = list(range(2020, 1989, -1))

for year in tqdm(range(2020, 1985, -1)):

    # subsampling from train points
    train_points = train_points_all.filter(ee.Filter.eq('year', year))

    # =================================prepare input layers=========================================
    #####################################################################################
    # extract neighborhood pixel values for composite images
    file_prefix = 'train_composite_' + str(year)

    # collect from Landsat composite raster layers
    composite_layer = ee.Image("users/meng_ee/CONUS_Composite_30/CONUS_" + str(year))
    composite_export = ee.batch.Export.table.toDrive(
        collection=focal_stat(composite_layer, kernel_radius, train_points),
        description=file_prefix,
        folder='Research',
        fileNamePrefix=file_prefix,
        fileFormat='CSV'
    )

    composite_tasks.append(composite_export)
    composite_export.start()

    #####################################################################################
    # extract neighborhood pixel values for LCMAP rasters

    # LCMAP raster
    LCPRI = ee.Image("users/meng_ee/LCMAP/LCMAP_CU_" + str(year) + "_V12_LCPRI").rename('LCPRI')
    LCSEC = ee.Image("users/meng_ee/LCMAP/LCMAP_CU_" + str(year) + "_V12_LCSEC").rename('LCSEC')

    # stack LCPRI and LCSEC
    stacked_LCMAP = LCPRI.addBands(LCSEC)

    # export train set
    LCMAP_export = ee.batch.Export.table.toDrive(
        collection=focal_stat(stacked_LCMAP, kernel_radius, train_points),
        description='train_LCMAP_' + str(year),
        folder='Research',
        fileNamePrefix='train_LCMAP_' + str(year),
        fileFormat='CSV'
    )
    LCMAP_tasks.append(LCMAP_export)
    LCMAP_export.start()

    #####################################################################################
    # extract neighborhood pixel values for LCMS rasters

    LCMS_dataset = ee.ImageCollection('USFS/GTAC/LCMS/v2021-7')

    LCMS = LCMS_dataset.filter(ee.Filter.And(
        ee.Filter.eq('year', year),
        ee.Filter.eq('study_area', 'CONUS')
    )) \
        .first() \
        .select(['Land_Cover', 'Land_Use'])

    # export train set
    LCMS_export = ee.batch.Export.table.toDrive(
        collection=focal_stat(LCMS, kernel_radius, train_points),
        description='train_LCMS_' + str(year),
        folder='Research',
        fileNamePrefix='train_LCMS_' + str(year),
        fileFormat='CSV'
    )
    LCMS_tasks.append(LCMS_export)
    LCMS_export.start()

    #####################################################################################
    # extract pixel values for census layers

    # set up census var names
    tp_vars = ['TP', 'TP_dens']

    empl_vars = ['C000'] + ['CNS' + str(i + 1).zfill(2) for i in range(20)]
    empl_vars = empl_vars + [i + '_dens' for i in empl_vars]

    tp_list = ee.List(tp_vars)
    empl_list = ee.List(empl_vars)

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

    # ======= stack processed census rasters
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
        mask = stacked_empl.select('C000').neq(-999)
        stacked_empl = stacked_empl.updateMask(mask)

        # ====== stack all census layers
        stacked_census = stacked_tp.addBands(stacked_empl)
        print("Bands of Census:", stacked_census.bandNames())

    census_collection = stacked_census.reduceRegions(
        collection=train_points,
        reducer=ee.Reducer.first(),
        scale=100,
        tileScale=16,
    )

    # export census features
    census_export = ee.batch.Export.table.toDrive(
        collection=census_collection,
        description='train_census_' + str(year),
        folder='Research',
        fileNamePrefix='train_census_' + str(year),
        fileFormat='CSV'
    )
    census_tasks.append(census_export)
    census_export.start()