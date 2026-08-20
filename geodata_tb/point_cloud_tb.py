"""
Pointcloud data processing toolbox.
--------------------------
author: Matthias Gassilloud
date: 05.06.2025
--------------------------

"""


### import modules
import sys
from numba import jit, prange
import numpy as np
import geopandas as gpd
import laspy
import warnings
from pathlib import Path


current_dir = Path(__file__).parent.parent.parent  # append utils
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import geos_utils.geodata_tb.vector as geo_vec
import geos_utils.numba_tb.numba_tb as ntb



def read_las(file_path, dimensions=None, no_points=None):
    """
    Read .las point cloud and return as numpy array.
    For further information check out: https://laspy.readthedocs.io/en/latest/complete_tutorial.html

    Parameters
    ----------
    file_path : str
        Path to the .las file.
    dimensions : list, optional
        List of dimension names to read from the .las file (e.g., ['gps_time', 'x', 'y', 'z', 'return_number']).
        If None, read all dimensions.
    no_points : int, optional
        Number of points to read. If None, reads all points.
        
    Returns
    -------
    pc_array : numpy.ndarray
        Array of shape (n, m) where n is the number of points read and m is the number
        of dimensions.
    meta : dict
        Dictionary containing metadata:
        - Points Read: Number of points actually read
        - Total point_count: Total number of points in the file
        - gps_encoding_value: GPS encoding value from header
        - gps_time_type: GPS time type from header
        - dimensions: List of dimensions that were read
        - las_header: Original .las header object
        
    Raises
    ------
    Warning
        If dimensions parameter is None, the function will print a warning and read all dimensions.
        
    Notes
    -----
    Use lowercase 'x', 'y', 'z' for scaled coordinates (= dimension * scale + offset).
    """

    with laspy.open(file_path) as f:
        header = f.header

        if no_points==None:
            print("    read all points.")
            pc = f.read()  # read all
        else:
            print(f"    read {int(no_points)} points.")
            pc = f.read_points(int(no_points))  # read points for testing

        if dimensions == None:
            dimensions = list(pc.point_format.dimension_names)
            warnings.warn("\n    no dimensions provided" \
                "\n    use lowercase 'x', 'y', 'z' for scaled values (= dimension * scale + offset)" \
                "\n    reading all dimensions")


        # read dimensions
        print(f"    read dimensions: {dimensions}")
        pc_array = np.column_stack([pc[att] for att in dimensions])
        points_read = pc_array.shape[0]


        # read some metadata
        meta = {"Points Read": points_read if points_read < f.header.point_count else f.header.point_count,
                "Total point_count": f.header.point_count,
                "gps_encoding_value": f.header.global_encoding.value,
                "gps_time_type": f.header.global_encoding.gps_time_type,
                "dimensions": dimensions,
                "las_header": header}

        print(f"    metadata: {meta}")
        
        return pc_array, meta  # return array and metadata

def crop_point_cloud(pointcloud_xy, polygon_path, status=True):
    """Crop a point cloud using a 2D polygon.
    
    This function filters points from a point cloud based on whether they fall within
    a specified polygon boundary. Only the X and Y coordinates are considered for
    this spatial filtering operation.
    
    Parameters
    ----------
    pointcloud_xy : numpy.ndarray
        Array of shape (n, 2) containing the X and Y coordinates of the point cloud.
        If the original point cloud has more dimensions, only the first two columns 
        should be passed to this function.
    polygon_path : str
        Path to the polygon shapefile used for cropping. The shapefile should contain
        a single polygon feature or be exploded if it contains multiple features.
    status : bool, optional
        Whether to print status information during processing, by default True.
        
    Returns
    -------
    crop_idx : numpy.ndarray
        Boolean array of shape (n,) where True indicates points that are inside the
        polygon and False indicates points that are outside.
    stats : dict
        Dictionary containing statistics about the cropping operation:
        - Original number of points: Total number of points in the input
        - Remaining points: Number of points inside the polygon
        - Removed points: Number of points outside the polygon
        
    Notes
    -----
    The function uses the crop_xy utility function to perform the actual point-in-polygon
    test. It reads the polygon from the specified shapefile using geopandas and converts
    it to a numpy array of boundary coordinates.
    """

    if status:
        print(f"Filter points within {polygon_path}")


    ### read polygon
    boundary_gdf = gpd.read_file(polygon_path).explode()
    bx, by = boundary_gdf.iloc[0]["geometry"].exterior.xy
    boundary_xy = np.c_[bx, by]  # https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array


    ### crop
    crop_idx = geo_vec.crop_xy(pointcloud_xy, boundary_xy)
    stats = {"Original number of points": pointcloud_xy.shape[0],
             "Remaining points": np.count_nonzero(crop_idx)}
    stats["Removed points"] = stats["Original number of points"] - stats["Remaining points"]


    if status:
        print(" "*4 + f"{stats}")

    return crop_idx, stats

def point_clouds_xyz_range(las_paths):
    """Get the x, y, z extents across multiple point cloud files.
    
    This function reads the headers of multiple .las files and extracts the minimum
    and maximum values for the X, Y, and Z coordinates. This is useful for determining
    the spatial extent of a collection of point clouds without loading the full data.
    
    Parameters
    ----------
    las_paths : list
        List of strings containing paths to .las files to be analyzed.
        
    Returns
    -------
    x_min : list
        List of minimum X values, one per input .las file.
    x_max : list
        List of maximum X values, one per input .las file.
    y_min : list
        List of minimum Y values, one per input .las file.
    y_max : list
        List of maximum Y values, one per input .las file.
    z_min : list
        List of minimum Z values, one per input .las file.
    z_max : list
        List of maximum Z values, one per input .las file.
        
    Notes
    -----
    This function only reads the header information from each .las file, making it
    much faster than loading the entire point cloud data when only the spatial
    extents are needed.
    """


    # value storage
    x_min = []
    x_max = []

    y_min = []
    y_max = []

    z_min = []
    z_max = []


    # iterate through las files
    for las_in in las_paths:
        print("Get XYZ range for ", las_in)

        with laspy.open(las_in) as f:  # only read header

            x_min.append(f.header.x_min)
            x_max.append(f.header.x_max)

            y_min.append(f.header.y_min)
            y_max.append(f.header.y_max)

            z_min.append(f.header.z_min)
            z_max.append(f.header.z_max)


    return x_min, x_max, y_min, y_max, z_min, z_max

def voxelize_pointcloud(las_file_in, xyz_bounds, cell_size, nb_cell, scaling=True):
    """Voxelize a point cloud.

    This function reads a .las file containing a point cloud and converts them
    into a 3D voxel grid.

    Parameters
    ----------
    las_file_in : str or numpy.ndarray
        Path to the input LAS file containing the point cloud data or numpy array containing
        point coordinates  [x, y, z] with shape (n,3).
    xyz_bounds : numpy.ndarray
        Array of shape (2, 3) specifying the minimum and maximum bounds [[min_x, min_y, min_z],
        [max_x, max_y, max_z]] of the voxel grid.
    cell_size : float
        Size of each voxel cell in the same units as the point cloud coordinates (usually metric).
    nb_cell : numpy.ndarray
        Array of shape (3,) specifying the number of cells [nx, ny, nz] in each dimension
        of the voxel grid.

    Returns
    -------
    numpy.ndarray
        3D binary array of shape (nz, ny, nx) where each element is 1 if the 
        corresponding voxel contains at least one point, and 0 otherwise.

    Notes
    -----
    The function applies scaling based on the cell_size precision to reduce floating-point
    errors during the voxelization process. Points outside the specified xyz_bounds are
    excluded from the resulting voxel grid.
    """

    print("Voxelize point cloud.")

    if isinstance(las_file_in, str):
        point_cloud_array, meta = read_las(las_file_in, dimensions=["x", "y", "z"])  # read las

    elif isinstance(las_file_in, np.ndarray):
        assert las_file_in.shape[1] == 3  # assert arrays has 3 cols (x,y,z)
        point_cloud_array = las_file_in

    else:
        print(f"Can't read point cloud data: {las_file_in} for voxelization.")


    ### [ OPTIONAL ] calculate scaling factor to reduce floating point precision error (e.g. for cell_size = 0.1)
    if scaling:
        
        decimals_cell_size = ntb.nb_float_to_string(cell_size)[::-1].find('.')  # number of float decimals

        if decimals_cell_size == -1:
            decimals_cell_size = 0

        # scaling factor
        scaling_factor = 10**decimals_cell_size

        # scale bounds
        cell_size = np.round(cell_size * scaling_factor)
        xyz_bounds = np.round(xyz_bounds * scaling_factor)

        # scale coordinates
        point_cloud_array *= scaling_factor # replace


    ### point cell index
    cidx = np.floor((point_cloud_array - xyz_bounds[0]) / cell_size).astype(np.int_)  # normalize
    point_cloud_array = None  # close


    ### mask points outside boundary
    xidx_mask = (cidx[:,0] >= 0) & (cidx[:,0] < nb_cell[0])
    yidx_mask = (cidx[:,1] >= 0) & (cidx[:,1] < nb_cell[1])
    zidx_mask = (cidx[:,2] >= 0) & (cidx[:,2] < nb_cell[2])
    cidx_masked = cidx[xidx_mask & yidx_mask & zidx_mask]

    xidx_mask, yidx_mask, zidx_mask = None, None, None  # close
    cidx = None  # close


    ### storage
    storage_points = np.full((nb_cell[2], nb_cell[1], nb_cell[0]), 0, dtype=np.byte)  # Byte (-128 to 127)
    storage_points[cidx_masked[:,2], cidx_masked[:,1], cidx_masked[:,0]] = 1  # idx of points inside 

    cidx_masked = None  # close


    return storage_points

def normalize_vox_array(vox_array, dtm, xyz_bounds, cell_size):
    """Normalizes a voxel array based on ground height from a DTM.

    This function transforms a voxel array from absolute height coordinates to
    height-above-ground coordinates using a Digital Terrain Model (DTM). For each
    horizontal position (x,y), voxels are shifted vertically so that the ground
    level (from DTM) corresponds to the bottom of the normalized space.

    Parameters
    ----------
    vox_array : numpy.ndarray
        3D array of shape (nz, ny, nx) containing the voxel classification values
        in absolute height coordinates.
    dtm : numpy.ndarray
        2D array of shape (ny, nx) containing ground elevation values that match
        the horizontal dimensions of the voxel array.
    xyz_bounds : numpy.ndarray
        Array of shape (2, 3) specifying the minimum and maximum bounds [[min_x, min_y, min_z],
        [max_x, max_y, max_z]] of the voxel grid in absolute coordinates.
    cell_size : float
        Size of each voxel cell in the same units as the xyz_bounds and DTM (usually meters).

    Returns
    -------
    numpy.ndarray
        3D array of shape (nz, ny, nx) containing the voxel classification values
        in height-above-ground coordinates.
    numpy.ndarray
        Array of shape (2, 3) containing the new [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        boundaries, where min_z and max_z are normalized by height.

    Raises
    ------
    ValueError
        If voxel array dimensions don't match expected format, DTM dimensions don't match
        voxel array's horizontal dimensions, or DTM values are outside the vertical range
        of the voxel grid.

    Notes
    -----
    The normalization process effectively "flattens" the terrain while preserving the 
    relative heights of objects above ground. This makes it easier to analyze vegetation
    structure and other above-ground features across areas with varying terrain.
    """


    ### checks
    if vox_array.ndim != 3:
        raise ValueError("Vox array needs to have [[z],[y],[x]] dimensions.")
    if dtm.shape != (vox_array.shape[-2], vox_array.shape[-1]):
        raise ValueError("DTM [y,x] dimensions need to match vox array [y,x] dimensions.")
    if np.any(dtm > xyz_bounds[1][2]):
        raise ValueError("DTM values are larger than maximum Z voxel grid boundary.")
    if np.any(dtm < xyz_bounds[0][2]):
        raise ValueError("DTM values are smaller than minimum Z voxel grid boundary.")



    ### calculate index from z bounds top to dtm ground
    norm_idx = np.floor((xyz_bounds[1][2] - dtm) / cell_size)  # max z - dtm = height, idx from boundary top to dtm
    if np.any(norm_idx < 0):
        raise ValueError("Maximum z in xyz_bounds of voxel grid is smaller than highest DTM value.")


    ### initialize new array
    storage_normalized = np.full((vox_array.shape[-3], vox_array.shape[-2], vox_array.shape[-1]), np.nan, dtype=vox_array.dtype)  # create empty


    ### iterate through cells
    for y_idx in range(vox_array.shape[-2]):
        for x_idx in range(vox_array.shape[-1]):
            nidx_cell = norm_idx[y_idx, x_idx]
            if np.isnan(nidx_cell):  # masked
                continue
            else:
                z_idx = int(nidx_cell)
                z_insert = vox_array[:z_idx, y_idx, x_idx]
                storage_normalized[-z_insert.shape[0] : , y_idx, x_idx] = z_insert  # insert z at end of array, according to z_shape


    ### new z bounds
    xyz_bounds_norm = xyz_bounds.copy()
    xyz_bounds_norm[0,2] = 0
    xyz_bounds_norm[1,2] = vox_array.shape[-3] * cell_size


    return storage_normalized, xyz_bounds_norm


#%% point cloud cleaning
@jit(nopython=True, parallel=True)
def check_strictly_increasing(return_vector_length, first_idx, last_idx):

    distance_mask = np.ones(first_idx.shape[0], dtype = np.bool_)

    for idx in prange(first_idx.shape[0]):
        f_idx = first_idx[idx]
        l_idx = last_idx[idx] + 1  # for from indexing to add +1
        returns_vl = return_vector_length[f_idx : l_idx]
        if np.any(np.isnan(returns_vl)):
            distance_mask[idx] = False
        else:
            distance_mask[idx] = np.all(np.diff(returns_vl) > 0)  # if empty, returns nan > 0 = true (works for single returns)

    return distance_mask

@jit(nopython=True, parallel=True)
def mask_rays_from_returns(return_mask, first_idx, last_idx):
    '''mask whole ray when one return masked'''

    ray_return_mask = np.ones(return_mask.shape[0], dtype = np.bool_)
    ray_mask = np.ones(first_idx.shape[0], dtype = np.bool_)

    for idx in prange(first_idx.shape[0]):
        f_idx = first_idx[idx]
        l_idx = last_idx[idx] + 1  # for from indexing to add .1
        check_returns = np.all(return_mask[f_idx : l_idx])
        ray_return_mask[f_idx : l_idx] = check_returns  # if empty, returns nan > 0 = true (works for single returns)
        ray_mask[idx] = check_returns

    return ray_return_mask, ray_mask


