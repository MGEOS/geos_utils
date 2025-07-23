"""
Vector data processing toolbox.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------

"""



### import modules
import geopandas as gpd
import matplotlib.path as mpltPath


def polygon_bounds(polygon_path):
    """Get the bounding box coordinates of a polygon.
    
    This function reads a polygon shapefile and returns its bounding box coordinates
    as [x_min, y_min, x_max, y_max].
    
    Parameters
    ----------
    polygon_path : str
        Path to the polygon shapefile to be analyzed.
        
    Returns
    -------
    numpy.ndarray
        Array containing the bounding box coordinates [x_min, y_min, x_max, y_max]
        of the polygon.
    """

    polygon_gdf = gpd.read_file(polygon_path)  # read 
    polygon_bounds = polygon_gdf.total_bounds  # get bounds
    print(f"Polygon boundary size: {polygon_bounds[2:] - polygon_bounds[0:2]}")

    return polygon_bounds

def crop_xy(pcXY, polygon):
    """Check if XY coordinates are inside a polygon.
    
    This function tests whether each point in a set of 2D coordinates falls within
    a specified polygon boundary. It implements a fast point-in-polygon test using
    matplotlib.path. For further information check: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    
    
    Parameters
    ----------
    pcXY : numpy.ndarray
        Array of shape (n, 2) containing the X and Y coordinates to test.
    polygon : numpy.ndarray
        Array of shape (m, 2) defining the vertices of the polygon boundary.
        
    Returns
    -------
    numpy.ndarray
        Boolean array of shape (n,) where True indicates points that are inside the
        polygon and False indicates points that are outside.
    """
    
    path = mpltPath.Path(polygon)
    return path.contains_points(pcXY)