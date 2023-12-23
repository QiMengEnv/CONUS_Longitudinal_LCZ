// Author: Meng Qi
// This script is to apply LCZ post-classification processing on the raw LCZ maps

var train_folder = "raw_prediction/";
var save_folder = "final_prediction/";
var prefix = "TP_";
// Load conus boundary
var conus_bound = ee.FeatureCollection("boundary/us_2020");

// for visulization, remap back to lcz numbers in model development (1-15)
var beforeRemap = [1,2,3,4,5,6,17,8,9,10,11,12,13,14,15,16];
var afterRemap =  [0,1,2,3,4,5, 6,7,8, 9,10,11,12,13,14,15];

// Define a palette for the lcz classification.
var lczPalette = [
  '#910613',  // (class:0)  LCZ 1:compact highrise
  '#D9081C',  // (class:1)  LCZ 2:compact midrise
  '#FF0A22',  // (class:2)  LCZ 3:compact lowrise
  '#C54F1E',  // (class:3)  LCZ 4:open highrise
  '#FF6628',  // (class:4)  LCZ 5:open midrise
  '#FF985E',  // (class:5)  LCZ 6:open lowrise
  '#656BFA',  // (class:6)  LCZ G:water
  '#BBBBBB',  // (class:7)  LCZ 8:large lowrise
  '#FFCBAB',  // (class:8)  LCZ 9:sparsly built
  '#565656',  // (class:9)  LCZ 10:heavy industry
  '#006A18',  // (class:10) LCZ A:dense trees
  '#00A926',  // (class:11) LCZ B:scattered trees
  '#628432',  // (class:12) LCZ C:bush,scrub
  '#B5DA7F',  // (class:13) LCZ D:low plants
  '#000000',  // (class:14) LCZ E:bare rock or paved
  '#FCF7B1',  // (class:15) LCZ F:bare soil or sand
  '#FFFFFF',  // (set unmasked value:16)  NoLabel
];

var collection_list = [];
var max_year = 2020;
// 1) load original prediction map
for (var i = 1986; i < (2021) ; i++) {
  var layer = ee.Image(train_folder + prefix + i.toString());
  layer = layer.remap(beforeRemap, afterRemap);
  layer = layer.set('year', i);
  layer = layer.set('system:time_start', ee.Date.fromYMD(i, 1, 1));
  layer = layer.rename('lcz');
  collection_list.push(layer);
}

var classifiedComposites = ee.ImageCollection(collection_list);

// sort descending
classifiedComposites = classifiedComposites.sort('year', false);

print('Original_classifiedComposite', classifiedComposites);

/////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian filtering
var gaussianFilter = function(image) {

  // Rename image back to 'remapped'
  image = image.rename('remapped');

  var image_year = image.get('year');
  var image_date = image.get('system:time_start');

  // Get the LCZ class numbers present in the image
  var freqHist = image.reduceRegions({reducer: ee.Reducer.frequencyHistogram(),
                      collection: image.geometry().bounds(), scale: 100});

  // rewrite the distionary, get the available LCZclasses
  var LCZclasses = ee.List(freqHist.map(function(feat){
    feat = ee.Feature(feat);
    var vals = ee.Dictionary(feat.get('histogram')).keys();
    return ee.Feature(null, {vals: vals});
  }).aggregate_array('vals')).flatten().distinct();

    // Set the radius and sigmas for the gaussian kernel - WHEN LCZ numbers start from 0
  var dictRadius = ee.Dictionary({
    0: 200,  1: 300,  2: 300,  3: 300,  4: 300,  5: 300,  6: 50, 7: 500,  8:300, 9: 500,
    10:200, 11: 200, 12: 200, 13: 200, 14: 300, 15: 200
  });
  var dictSigma = ee.Dictionary({
    0: 100,  1: 150,  2: 150,  3: 150,  4: 150,  5: 150,  6: 25, 7: 250, 8:150, 9: 250,
    10:75,  11: 75,  12: 75,  13: 75,  14: 150, 15: 75
  });


  var applyKernel = function(i){

    var i_int = ee.Number.parse(i).toInt();
    var radius = dictRadius.get(i_int);
    var sigma  = dictSigma.get(i_int);
    var kernel = ee.Kernel.gaussian(radius, sigma, 'meters');

    var lcz = image.eq(i_int).convolve(kernel).addBands(ee.Image(i_int).toInt().rename('lcz'));


    return lcz;
  };

  // Make mosaic from collection
  var coll = ee.ImageCollection.fromImages(LCZclasses.map(applyKernel));

  // Select highest value per pixel:
  var mos = coll.qualityMosaic('remapped');

  // Select lcz bands again to obtain filtered LCZ map
  var lczF = mos.select('lcz');

  lczF = lczF.set('year', image_year);
  lczF = lczF.set('system:time_start', image_date);

  return lczF.rename('lcz');
};

var filteredComposite = classifiedComposites.map(gaussianFilter);
filteredComposite = filteredComposite.map(function(img){
  return img.unmask(16);
});

print('Gaussian_filteredComposite', filteredComposite);
Map.addLayer(filteredComposite, {palette: lczPalette, min: 0, max: 16}, 'Gaussian_filtered');

/////////////////////////////////////////////////////////////////////////////////////////////
// Temporal Moving-Window Smoothing
var time_smoothing = function(year_interval, Composite, round){
  // year_interval: Specify the time-window
  // filteredComposite: prediction maps
  // round: round of smoothing, just to print the infomation
    
  // We use a 'join' to find all images that are within the time-window
  // The join will add all matching images into a new property called 'images'
  var join = ee.Join.saveAll({
    matchesKey: 'images'
  });

  // This filter will match all images that are captured
  // within the specified time interval of the source image
  var diffFilter = ee.Filter.maxDifference({
    difference: year_interval,
    leftField: 'year',
    rightField: 'year'
  });

  var joinedCollection = join.apply({
    primary: Composite,
    secondary: Composite,
    condition: diffFilter
  });


  // Each image in the joined collection will contain
  // matching images in the 'images' property
  // Extract and return the mode of matched images

  var smoothedCollection = ee.ImageCollection(joinedCollection.map(function(image) {
    var image_year = image.get('year');
    var collection = ee.ImageCollection.fromImages(image.get('images'));
    var currentYearImage = collection.filter(ee.Filter.eq('year', image_year)).first();
    var countDistinct = collection.reduce(ee.Reducer.countDistinct());
    var modeImage = collection.mode();

    if (image_year == 1986 || image_year == 2020) {
      // if there are three distict count, substitue mode with the current value
      modeImage = modeImage.where(countDistinct.eq(3), currentYearImage);
    } else if (image_year == 1987 || image_year == 2019) {
      modeImage = modeImage.where(countDistinct.eq(4), currentYearImage);
    } else {
      modeImage = modeImage.where(countDistinct.eq(5), currentYearImage);
    }

    return ee.Image(image).addBands(modeImage.rename('smoothed'));
  }));


  // sort
  var smoothedCollection_updated = smoothedCollection.select('smoothed');
  smoothedCollection_updated =smoothedCollection_updated.sort('year', false);

  // rename
  smoothedCollection_updated = smoothedCollection_updated.map(function(img){
    return img.rename('lcz');
  });
  print('smoothedCollection_'+round, smoothedCollection_updated);

  return smoothedCollection_updated;
};

// round 1) ===================================
filteredComposite = time_smoothing(2, filteredComposite, '1');  // year_interval=1, round="1"

// round 2) ===================================
filteredComposite = time_smoothing(2, filteredComposite, '2');  // year_interval=1, round="2"

// round 3) ===================================
filteredComposite = time_smoothing(2, filteredComposite, '3');  // year_interval=1, round="2"

Map.addLayer(filteredComposite, {palette: lczPalette, min: 0, max: 16}, 'Temporal_smoothed');
print('Temporal_smoothedComposite', filteredComposite);

