# -*- coding: utf-8 -*-
"""
Ray box intersection and voxel traversal algorithm.
--------------------------
author: Matthias Gassilloud
date: 03.06.2025
--------------------------

Ray box intersection by William et al. (2005) [1]
Voxel traversal algorithm by Amanatides & Woo (1987) [2]
Implementation by Gassilloud et al. (2025) [3]


References:

[1] Williams, A., Barrus, S., Morley, R. K., & Shirley, P. (2005). An efficient and robust ray-box intersection algorithm. In ACM SIGGRAPH 2005 Courses (pp. 9-es).
[2] Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. In Eurographics (Vol. 87, No. 3, pp. 3-10).
[3] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""


# modules
import sys
import numpy as np
from numba import jit, prange


from pathlib import Path
current_dir = Path(__file__).parent.parent.parent.parent  # append utils
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import geos_utils.numba_tb.numba_tb as ntb  # import numba toolbox



### boundary voxelization
@jit(nopython=True) #, parallel=True, fastmath=True, cache=True)
def point_within_boundary(point, boundary):
    return np.all(boundary[0] < point) and np.all(point < boundary[1])

@jit(nopython=True) #, parallel=True, fastmath=True, cache=True)
def vox_aoi(boundary, cell_size):
    '''
    boundary to define points 
    np.all(boundary[0] < point) and np.all(point < boundary[1])
    '''

    # make fit to grid
    decimals_cell_size = ntb.nb_float_to_string(cell_size).find('.')
    if decimals_cell_size == -1:
        decimals_cell_size = 0
        
        
    XYZmin = (np.ceil(boundary[0] / cell_size)).astype(np.int_) * cell_size
    XYZmax = (np.floor(boundary[1] / cell_size)).astype(np.int_) * cell_size  # 1 is inside, 10 is outside
    nb_cell = np.round((XYZmax - XYZmin) / cell_size).astype(np.int_)  # image space
    boundary = np.round(np.vstack((XYZmin, XYZmax)), decimals_cell_size)  # round to correct floating point precision of cell_size
    
    # print("Voxel Boundary:\n"\
    #       f"Min: {XYZmin}\n"\
    #       f"Max: {XYZmax}")
    
    return boundary, nb_cell


### ray box intersection
@jit(nopython=True, fastmath=True) #, parallel=True, cache=True)
def box_intersect3D(bounds, origin, direction):
    
    t0, t1 = 0, 1

    inv_direction = np.divide(1, direction)
    sign = (inv_direction < 0).astype(np.int8)


    tmin = (bounds[sign[0],0] - origin[0]) * inv_direction[0]
    tmax = (bounds[1-sign[0],0] - origin[0]) * inv_direction[0]
    tymin = (bounds[sign[1],1] - origin[1]) * inv_direction[1]
    tymax = (bounds[1-sign[1],1] - origin[1]) * inv_direction[1]
    
    
    if ( (tmin > tymax) or (tymin > tmax) ):
        return None
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax
    
    tzmin = (bounds[sign[2],2] - origin[2]) * inv_direction[2]
    tzmax = (bounds[1-sign[2],2] - origin[2]) * inv_direction[2]
    
    if ((tmin > tzmax) or (tzmin > tmax) ):
        return None
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    
    if ( (tmin < t1) and (tmax > t0) ):
        return np.array([tmin, tmax])
    else:
        return None

@jit(nopython=True) #, parallel=True, fastmath=True, cache=True)
def ray_box_intersect(origin, end, boundary):
    
    if not point_within_boundary(origin, boundary) or not point_within_boundary(end, boundary):
        
        direction = end-origin
        t = box_intersect3D(bounds=boundary, origin=origin, direction=direction)
    
        if t is not None:  # some intersection is found
            if t[0] >= 0:  # start outside boundary
                intersection_point = origin + direction * t[0]  # new start
                #print("intersection point", intersection_point)
                return intersection_point, end
            else:  # start inside boundary, end outside boundary 
                intersection_point = end + direction * t[1]  # new end
                #print("intersection point", intersection_point)
                return origin, intersection_point
        else:
            #print("No intersection")
            return None, None
    
    else: # both points are within, no intersection needed
        return origin, end


### vox traversal
@jit(nopython=True) #, cache=True)
def frac0(point, cell_size):
    return (point / cell_size) - np.floor(point / cell_size)

@jit(nopython=True) #, cache=True)
def frac1(point, cell_size):
    return 1 - (point / cell_size) + np.floor(point / cell_size)

@jit(nopython=True)  #, parallel=True, fastmath=True, cache=True)
def initialize_step(ray_vector):
    # StepX will be 0, 1, -1 depending on the ray's x direction.
    Step = ray_vector.copy()  # copy
    Step[Step > 0] = 1
    Step[Step < 0] = -1
    #Step[Step == 0] = 0  # no need to change
    return Step.astype(np.int_)

@jit(nopython=True) #, cache=True)
def vox_trav_initialization(origin, direction, cell_size):

    d = np.sign(direction)
    tDelta = abs(np.divide(cell_size, direction))  # use np.divide for divide by zero handling, other option: @njit(error_model='numpy')
    if (d > 0):
        tMax = tDelta * frac1(origin, cell_size)
    else:
        tMax = np.nan_to_num(tDelta * frac0(origin, cell_size), nan=np.inf)  # inf / 0 = nan
    #print("tDelta, tMax", tDelta, tMax)
    return tDelta, tMax

@jit(nopython=False, parallel=True, fastmath=True, cache=True)  #, fastmath=True, cache=True)  # parallel={"setitem":False}
def ray_vox_trav_nb(rays, boundary, cell_size): #, geospace=False):


    ### [optional] - factor to reduce floating point precision errors


    
    ### regular grid boundary
    boundary, nb_cell = vox_aoi(boundary, cell_size)  # match aoi with vox grid
    
    
    ### storage
    storage = np.full((nb_cell[2], nb_cell[1], nb_cell[0]), 0, dtype=np.byte)  # Byte (-128 to 127)
    
    
    ### counter
    no_intersection = 0
    
    
    
    ### for each ray
    for idx in prange(rays.shape[0]):
        
        
        
        ### ray
        ray = rays[idx]
        # origin = ray[0]
        # end = ray[1]
    
    
        ### intersection
        origin, end = ray_box_intersect(ray[0], ray[1], boundary)
        
        if origin is None:
            no_intersection +=1
            continue  # no intersection
        
        
        ### vector
        origin, end = origin.copy(), end.copy()
        direction = end - origin
        
        
        ### vox ID initialization
        point_voxID_img_space = np.floor((origin - boundary[0]) / cell_size).astype(np.int_)
    
        
        ### initialize tDelta, tMax, step
        tDeltaX, tMaxX = vox_trav_initialization(origin[0], direction[0], cell_size)
        tDeltaY, tMaxY = vox_trav_initialization(origin[1], direction[1], cell_size)
        tDeltaZ, tMaxZ = vox_trav_initialization(origin[2], direction[2], cell_size)
        step = initialize_step(direction)
    
    
        ### start voxel
        X_img = int(point_voxID_img_space[0])  # img space
        Y_img = int(point_voxID_img_space[1])  # img space
        Z_img = int(point_voxID_img_space[2])  # img space
    
    
        ### append start voxel
        if (0 <= X_img < nb_cell[0]) and (0 <= Y_img < nb_cell[1]) and (0 <= Z_img < nb_cell[2]):
            # vox_trav.append(vox)  # append (if in range)
            '''in some cases of tDetla=inf first vox is appended twice. this doesnt realy matter for mapping''' 
            
            # img space
            storage[Z_img, Y_img, X_img] = 1  # count initial voxel
    
        
        ### vox traversal
        while True:
        
            if (tMaxX < tMaxY):
                
                if (tMaxX < tMaxZ):  # x step
                    X_img += step[0]  # img space
                    if not (0 <= X_img < nb_cell[0]):
                        break
                    tMaxX += tDeltaX
                    
                else:  # z step
                    Z_img += step[2]  # img space
                    if not (0 <= Z_img < nb_cell[2]):
                        break
                    tMaxZ += tDeltaZ
    
            else:
                
                if (tMaxY < tMaxZ):  # y step
                    Y_img += step[1]  # img space
                    if not (0 <= Y_img < nb_cell[1]):
                        break
                    tMaxY += tDeltaY
                    
                else:  # z step
                    Z_img += step[2]  # img space
                    if not (0 <= Z_img < nb_cell[2]):
                        break
                    tMaxZ += tDeltaZ
        
            
            # check if inside boundary
            X_in = (0 <= X_img < nb_cell[0])
            Y_in = (0 <= Y_img < nb_cell[1])
            Z_in = (0 <= Z_img < nb_cell[2])
            
            if not X_in and not Y_in and not Z_in:  # stop by boundary
                break  # (if not in range)
            elif not X_in or not Y_in or not Z_in:
                continue  # only one axis not in range, could be just before finish
            else:
                storage[Z_img, Y_img, X_img] = 1  # store
                
    
            # check if max distance reached
            if (np.around(tMaxX, decimals = 15) >= 1) and (np.around(tMaxY, decimals=15) >= 1) and (np.around(tMaxZ, decimals = 15) >= 1):  # stop by distance, floating point error
                break
    
    
    stats = {"Number of Intersection": no_intersection,
              "Number of Rays" : rays.shape[0]}
    
    
    # if geospace:
    #     return storage[::-1,::-1,:], boundary, stats
    # else:
    return storage, boundary, stats


