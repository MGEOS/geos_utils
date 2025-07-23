"""
Raster data processing toolbox.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------

"""


import fiona
import rasterio
from rasterio.mask import mask


#%% I/O

def read_raster_array(path_in, mask=False, band=1):

    # read raster
    with rasterio.open(path_in) as src:

        array = src.read(band, masked=mask)  # read, If masked is True the return value will be a masked array.
        meta = {"crs": src.crs,
                "transform_gdal": src.get_transform(),
                "transform_affine": src.transform,
                "descr": src.descriptions[band-1],
                "height": src.height,
                "width": src.width,
                "bounds": src.bounds,
                "nodata": src.nodata}

    return array, meta

def write_raster_array(path_out, array, crs, transform, b_descr=None):

    # multiband array
    if len(array.shape) > 2:

        with rasterio.open(
            path_out,
            'w',
            driver = 'GTiff',
            height = array.shape[1],  # correct! np shape shows firs no. rows = height
            width = array.shape[2],
            count = array.shape[0],  # bands
            dtype = str(array.dtype),
            crs = crs,
            transform = transform,
        ) as dst:
            for id, layer in enumerate(array, start=1):
                dst.write_band(id, layer)
                if b_descr is not None:
                    dst.set_band_description(id, b_descr[id-1])

    else:  # single band array
        with rasterio.open(
            path_out,
            'w',
            driver = 'GTiff',
            height = array.shape[0],  # correct! np shape shows firs no. rows = height
            width = array.shape[1],
            count = 1,
            dtype = str(array.dtype),
            crs = crs,
            transform = transform,
        ) as dst:
            dst.write(array, 1)
            if b_descr is not None:
                dst.set_band_description(1, b_descr[0])


#%% operations

def crop_raster(raster_in, raster_out, crop_shp, crop=True):
    # https://rasterio.readthedocs.io/en/stable/topics/masking-by-shapefile.html
    # crop = False : Applying the features in the shapefile as a mask on the raster sets all pixels outside of the features to be zero. 
    # Features are assumed to be in the same coordinate reference system as the input raster.

    with fiona.open(crop_shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(raster_in) as src:
        out_image, out_transform = mask(src, shapes, crop=crop)
        out_meta = src.meta


    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(raster_out, "w", **out_meta) as dest:
        dest.write(out_image)


#%% transform

def transform_affine_from_origin(x_min, y_max, x_width, y_height):

    return rasterio.transform.from_origin(x_min, y_max, x_width, y_height)

