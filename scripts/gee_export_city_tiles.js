// Google Earth Engine script to export real tile-level metadata and a
// Landsat-based composite for one city / AOI.
//
// Outputs (to Google Drive):
// 1) <CITY_NAME>_tiles_metadata.csv
//    Columns (at least):
//      - tile_id
//      - row, col
//      - min_lon, min_lat, max_lon, max_lat  (EPSG:4326)
//      - lon, lat   (tile centroid, EPSG:4326)
//      - date       (YYYY-MM-DD)
//      - image_id   (tile_id + "_" + date)
//      - lst        (tile-mean LST in Celsius, real, from Landsat ST_B10)
//      - NDVI_mean  (tile-mean NDVI)
//      - scene_id
//      - cloud_cover_land
//
// 2) <CITY_NAME>_landsat_composite.tif
//    Bands (in this exact order, matching scripts/prepare_real_inputs.py):
//      1: RED   (surface reflectance)
//      2: GREEN
//      3: BLUE
//      4: NDVI
//      5: LST_C (Celsius)
//
// After export + download, use scripts/prepare_real_inputs.py locally to
// convert these into:
//   - raw/satellite_metadata.csv
//   - raw/satellite_images/<image_id>.png

// ------------------------------------------------------------
// USER PARAMETERS (EDIT THESE FIRST)
// ------------------------------------------------------------

// Name of the city / experiment (used in export filenames).
var CITY_NAME = 'ExampleCity';

// Define Area Of Interest (AOI).
// Option 1: Paste bounding box [minLon, minLat, maxLon, maxLat].
// var aoi = ee.Geometry.Rectangle([77.0, 12.0, 78.0, 13.0]);  // Bengaluru-ish example

// Option 2 (recommended): Draw a polygon in the Code Editor GUI,
// rename it to 'aoi', and uncomment the line below.
// var aoi = aoi;

// Date range for Landsat scenes.
var START_DATE = '2022-01-01';
var END_DATE   = '2022-12-31';

// Grid configuration (rows x cols ~ number of tiles).
var GRID_ROWS = 5;   // 5 x 10 = 50 tiles
var GRID_COLS = 10;

// Maximum number of dates (samples) per tile.
var MAX_DATES_PER_TILE = 10;

// ------------------------------------------------------------
// DATASETS AND HELPERS
// ------------------------------------------------------------

// Landsat Collection 2 Level-2 (L8 + L9) – surface reflectance + surface temperature.
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');

// Simple cloud/shadow mask for L2.
function maskL2Clouds(image) {
  var qaPixel = image.select('QA_PIXEL');
  var cloudBit = 1 << 3;
  var cloudShadowBit = 1 << 4;
  var mask = qaPixel.bitwiseAnd(cloudBit).eq(0)
    .and(qaPixel.bitwiseAnd(cloudShadowBit).eq(0));
  return image.updateMask(mask);
}

// Prepare an image with:
// RED, GREEN, BLUE (surface reflectance),
// NDVI,
// LST_C (Celsius).
function prepareLandsat(image) {
  // Surface reflectance scaling for SR_B* bands (USGS C2 L2 guide).
  var srScale = 0.0000275;
  var srOffset = -0.2;

  var sr = image.select(['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5'])
    .multiply(srScale)
    .add(srOffset);

  var red  = sr.select('SR_B4').rename('RED');
  var green = sr.select('SR_B3').rename('GREEN');
  var blue  = sr.select('SR_B2').rename('BLUE');
  var nir   = sr.select('SR_B5');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

  // Surface temperature from ST_B10 band.
  // Scale/offset from USGS Collection 2 documentation:
  //   ST_K = ST_B10 * 0.00341802 + 149.0
  //   LST_C = ST_K - 273.15
  var lstK = image.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0);
  var lstC = lstK.subtract(273.15).rename('LST_C');

  return ee.Image.cat([red, green, blue, ndvi, lstC])
    .copyProperties(image, image.propertyNames());
}

// ------------------------------------------------------------
// 1. BUILD TILE GRID OVER AOI
// ------------------------------------------------------------

if (typeof aoi === 'undefined') {
  throw 'Please define "aoi" (either rectangle or drawn polygon) before running.';
}

var bounds = aoi.bounds();
var coords = ee.List(bounds.coordinates().get(0));
var lons = coords.map(function(pt) { return ee.Number(ee.List(pt).get(0)); });
var lats = coords.map(function(pt) { return ee.Number(ee.List(pt).get(1)); });

var minLon = ee.Number(lons.reduce(ee.Reducer.min()));
var maxLon = ee.Number(lons.reduce(ee.Reducer.max()));
var minLat = ee.Number(lats.reduce(ee.Reducer.min()));
var maxLat = ee.Number(lats.reduce(ee.Reducer.max()));

var lonStep = maxLon.subtract(minLon).divide(GRID_COLS);
var latStep = maxLat.subtract(minLat).divide(GRID_ROWS);

var rows = ee.List.sequence(0, GRID_ROWS - 1);
var cols = ee.List.sequence(0, GRID_COLS - 1);

var tiles = ee.FeatureCollection(
  rows.map(function(r) {
    r = ee.Number(r);
    return cols.map(function(c) {
      c = ee.Number(c);
      var tileMinLon = minLon.add(c.multiply(lonStep));
      var tileMaxLon = tileMinLon.add(lonStep);
      var tileMinLat = minLat.add(r.multiply(latStep));
      var tileMaxLat = tileMinLat.add(latStep);

      var geom = ee.Geometry.Rectangle(
        [tileMinLon, tileMinLat, tileMaxLon, tileMaxLat],
        null,
        false
      );

      var centroid = geom.centroid();
      var tileId = ee.String('tile_r')
        .cat(r.format('%d'))
        .cat('_c')
        .cat(c.format('%d'));

      return ee.Feature(geom, {
        tile_id: tileId,
        row: r,
        col: c,
        min_lon: tileMinLon,
        min_lat: tileMinLat,
        max_lon: tileMaxLon,
        max_lat: tileMaxLat,
        lon: centroid.coordinates().get(0),
        lat: centroid.coordinates().get(1)
      });
    });
  }).flatten()
).filterBounds(aoi);

print('Tile grid', tiles);
Map.centerObject(aoi, 11);
Map.addLayer(tiles, {color: 'yellow'}, 'Tiles');

// ------------------------------------------------------------
// 2. PREPARE LANDSAT COLLECTION & COMPOSITE
// ------------------------------------------------------------

var collection = l8.merge(l9)
  .filterBounds(aoi)
  .filterDate(START_DATE, END_DATE)
  .map(maskL2Clouds)
  .map(prepareLandsat);

print('Prepared Landsat collection size:', collection.size());

// Composite for imagery chips: median of prepared bands.
var composite = collection.median().select(['RED', 'GREEN', 'BLUE', 'NDVI', 'LST_C']);

// Quick visualization (RGB only).
Map.addLayer(
  composite.select(['RED', 'GREEN', 'BLUE']),
  {min: 0.03, max: 0.4},
  'Landsat composite RGB'
);

// ------------------------------------------------------------
// 3. BUILD TILE-LEVEL METADATA WITH PER-DATE LST
// ------------------------------------------------------------

// For each tile, collect up to MAX_DATES_PER_TILE dates and compute tile-mean LST.
var maxDates = MAX_DATES_PER_TILE;

var samples = tiles.map(function(tile) {
  tile = ee.Feature(tile);
  var geom = tile.geometry();
  var tileId = tile.getString('tile_id');

  var tileCol = collection.filterBounds(geom);

  // Get distinct dates present for this tile.
  var dateList = ee.List(
    tileCol.aggregate_array('system:time_start')
  ).map(function(t) {
    return ee.Date(t).format('YYYY-MM-dd');
  }).distinct().sort();

  // Limit to first maxDates (deterministic order).
  var limitedDates = dateList.slice(0, maxDates);

  var perDate = limitedDates.map(function(dateStr) {
    dateStr = ee.String(dateStr);
    var start = ee.Date(dateStr);
    var end = start.advance(1, 'day');

    // Choose the least-cloudy image for this tile and date.
    var img = tileCol
      .filterDate(start, end)
      .sort('CLOUD_COVER_LAND')
      .first();

    return ee.Algorithms.If(
      img,
      (function() {
        img = ee.Image(img);
        var lstImg = img.select('LST_C');
        var ndviImg = img.select('NDVI');

        var lstMean = lstImg.reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e7
        }).get('LST_C');

        var ndviMean = ndviImg.reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e7
        }).get('NDVI');

        var imageId = tileId.cat('_').cat(dateStr);

        return ee.Feature(geom, {
          tile_id: tileId,
          row: tile.get('row'),
          col: tile.get('col'),
          min_lon: tile.get('min_lon'),
          min_lat: tile.get('min_lat'),
          max_lon: tile.get('max_lon'),
          max_lat: tile.get('max_lat'),
          lon: tile.get('lon'),
          lat: tile.get('lat'),
          date: dateStr,
          image_id: imageId,
          lst: lstMean,
          NDVI_mean: ndviMean,
          scene_id: img.get('LANDSAT_PRODUCT_ID'),
          cloud_cover_land: img.get('CLOUD_COVER_LAND')
        });
      })(),
      null
    );
  });

  return ee.FeatureCollection(perDate).filter(ee.Filter.notNull(['lst']));
}).flatten();

print('Sample metadata (first 10)', samples.limit(10));

// ------------------------------------------------------------
// 4. EXPORTS
// ------------------------------------------------------------

// 4.1. Tile-level metadata table (for scripts/prepare_real_inputs.py)
Export.table.toDrive({
  collection: samples,
  description: CITY_NAME + '_tiles_metadata',
  fileFormat: 'CSV',
  fileNamePrefix: CITY_NAME + '_tiles_metadata'
});

// 4.2. Composite GeoTIFF with bands: [RED, GREEN, BLUE, NDVI, LST_C]
Export.image.toDrive({
  image: composite,
  description: CITY_NAME + '_landsat_composite',
  fileNamePrefix: CITY_NAME + '_landsat_composite',
  region: aoi,
  scale: 30,  // Landsat native resolution
  maxPixels: 1e13
});

// ------------------------------------------------------------
// HOW TO USE THIS SCRIPT
// ------------------------------------------------------------
// 1. Open code.earthengine.google.com in your browser.
// 2. Paste this entire script into a new script tab.
// 3. Define "aoi" (bounding box or drawn polygon) and set CITY_NAME, START_DATE, END_DATE.
// 4. Click "Run" to preview tiles and composite.
// 5. In the Tasks panel, start the two exports:
//      - <CITY_NAME>_tiles_metadata (CSV to Google Drive)
//      - <CITY_NAME>_landsat_composite (GeoTIFF to Google Drive)
// 6. After export completes, download both files and place them locally, e.g.:
//      raw/landsat/<CITY_NAME>_tiles_metadata.csv
//      raw/landsat/<CITY_NAME>_landsat_composite.tif
// 7. Then run the local script:
//      python scripts/prepare_real_inputs.py \
//        --ee_metadata_csv raw/landsat/<CITY_NAME>_tiles_metadata.csv \
//        --composite_tif   raw/landsat/<CITY_NAME>_landsat_composite.tif \
//        --output_metadata_csv raw/satellite_metadata.csv \
//        --output_images_dir   raw/satellite_images
// 8. Next, use scripts/fetch_open_meteo_weather.py to create raw/weather_downloaded.csv.

