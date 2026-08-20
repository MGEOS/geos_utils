import math
import sys
from pathlib import Path
import numpy as np
from numba import jit, prange


# add geos_utils to path for importing required functions
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


from CANOPy.geos_utils.algorithms_tb.voxel_traversal.ray_vox_trav_3d_nb import ray_box_intersect, vox_aoi
from CANOPy.occlusion_mapping.pulse_origin_reconstruction import extend_trajectory_to_height
from CANOPy.geos_utils.algorithms_tb.voxel_traversal.ray_vox_trav_3d_nb import initialize_step, ray_box_intersect, vox_trav_initialization
from CANOPy.geos_utils.numba_tb.numba_tb import ellipse_area, vector_length_nb


# transmittance estimation
def pulse_fraction(number_returns, return_number):
    return 1 - ((1 / number_returns) * (return_number - 1))

@jit(nopython=True, fastmath=True, cache=True)  # fastmath safe
def beam_radius(distance, divergence, initial_diameter):
    '''
    divergence = divergence (half-angle, mrad)
    initial_diameter (meters)
    return: beam_diameter (meters)
    '''
    beam_radius = np.divide(initial_diameter, 2) + distance * math.tan(np.divide(divergence, 1000))
    return beam_radius

@jit(nopython=True, parallel=True, cache=True, debug=False)
def transmittance_vincent2017(
    initial_beam_diameter,
    beam_div_semi_major,
    beam_div_semi_minor,
    vox_space_shape,
    vox_filled_idx_unique,
    ray_ids_vox_sorted,
    first_vox_idx,
    last_vox_idx,
    vox_min,
    vox_max,
    vox_center,
    start_coordinate,
    end_coordinate,
    first_idx,
    last_idx,
    return_vector_length,
    PF,
    ray_count_dtype = np.uint64
    ):
    """transmittance estimation as described in Vincent2017"""


    # vox space for transmittance values
    P_transmittance = np.full((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), np.nan, dtype=np.float64)  # initialize vox_ID with non-existing pointer


    # vox space for explored voxel volume
    volume_explored = np.full((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), np.nan, dtype=np.float64)  # initialize vox_ID with non-existing pointer


    # vox space for ray_count
    ray_count = np.zeros((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), dtype=ray_count_dtype)  # initialize vox_ID with non-existing pointer


    for i in prange(vox_filled_idx_unique.shape[0]):


        # get ray ids
        ray_ids = np.unique(ray_ids_vox_sorted[first_vox_idx[i]:last_vox_idx[i]])  # ensure only unique ray ids per voxel


        # get vox idx
        vox_idx = vox_filled_idx_unique[i]  # z,x,y


        # value storage
        dividend = 0
        divisor = 0
        l = 0
        vox_vol_explored = 0



        # n - number of rays in voxel
        n = 0


        # voxel boundary
        vox_boundary = np.vstack((vox_min[i], vox_max[i]))  # voxel boundary [min, max], use [i] instead of [vox_id] since vox_id values higher than actual number of vox_ids



        for r_i in prange(ray_ids.shape[0]):  # loop over rays in vox


            # get ray id
            ray_id = ray_ids[r_i]  # ensure only unique ray ids


            # ray voxel intersection
            vox_enter, vox_exit = ray_box_intersect(start_coordinate[ray_id], end_coordinate[ray_id], vox_boundary)


            if vox_enter is None:
                #sys.exit("No interaction with voxel found, something went wrong...")
                # should theoretically never happen since beams were mapped in voxel.
                # might happen due to inprecision of voxel traversal algorithms that voxels
                # are mapped where beam did not pass.
                continue  # skip ray


            # get ray return vector length
            ray_first_return_idx = first_idx[ray_id].item()  # index of first return
            ray_last_return_idx = last_idx[ray_id].item() + 1  # index of last return - if it is from -> to: need to add +1 for correct indexing
            ray_returns_vl = return_vector_length[ray_first_return_idx:ray_last_return_idx]  # vector length from pulse origin to returns


            # l_i - length of pulse optical path (in voxel)
            l_i = vector_length_nb(vox_enter, vox_exit)


            # PF - Pusle fractions of ray
            PF_ray = np.append(PF[ray_first_return_idx:ray_last_return_idx], 0)  # append 0 to include PFOut of last return


            # PFEnt - entering fraction of pulse
            vox_enter_vector_length = vector_length_nb(start_coordinate[ray_id], vox_enter)
            vox_enter_vector_length = np.round(vox_enter_vector_length, 5)  # ray_returns_vl lost some precision due to scaling/rescaling, therefore round everything to 5 digits
            PFEnt_i = PF_ray[np.searchsorted(ray_returns_vl, vox_enter_vector_length, side="left")]


            # PFOut - exiting fraction of pulse
            vox_exit_vector_length = vector_length_nb(start_coordinate[ray_id], vox_exit)
            vox_exit_vector_length = np.round(vox_exit_vector_length, 5)  # ray_returns_vl lost some precision due to scaling/rescaling, therefore round everything to 5 digits
            PFOut_i = PF_ray[np.searchsorted(ray_returns_vl, vox_exit_vector_length, side="right")]


            # S_i - cross section of pulse at voxel center
            vox_center_vector_length = vector_length_nb(start_coordinate[ray_id], vox_center[i])
            radius1 = beam_radius(distance=vox_center_vector_length, divergence=beam_div_semi_major, initial_diameter=initial_beam_diameter)  # meter
            radius2 = beam_radius(distance=vox_center_vector_length, divergence=beam_div_semi_minor, initial_diameter=initial_beam_diameter)  # meter
            S_i = ellipse_area(radius1, radius2)  # sqm


            # aggregate beam values
            dividend += PFOut_i * S_i * l_i
            divisor += PFEnt_i * S_i * l_i
            l += l_i
            vox_vol_explored += S_i * l_i

            n += 1  # count beams, some may not actually pass this voxel due to floating point precision errors of ray tracing algorithm


        if n == 0:
            continue  # do not store result, keep P_vox initialized with np.nan


        # P_tranmissivity
        basis = np.divide(dividend, divisor)
        exponent = np.divide(1, (np.divide(1, n) * l))
        P_vox = np.power(basis, exponent)


        # store result
        P_transmittance[vox_idx[2], vox_idx[1], vox_idx[0]] = P_vox


        # if return_volume_explored:
        volume_explored[vox_idx[2], vox_idx[1], vox_idx[0]] = vox_vol_explored

        # if return_ray_count:
        ray_count[vox_idx[2], vox_idx[1], vox_idx[0]] = n


    return P_transmittance[::-1,::-1,:], volume_explored[::-1,::-1,:], ray_count[::-1,::-1,:]

@jit(nopython=True, fastmath=True, parallel=True, cache=True, debug=False)  # fastmath safe
def err_transmittance_estimate(transmittance, beam_section_out_sum, beam_section_in, full_vox_path_length):
    '''python implementation of amapvox transmittance estimation error algorithm'''
    # ! entering_beam_section needs to be array of individual beams (as well as full_vox_path_length) 
    cum_bs_in = np.sum(beam_section_in * (transmittance ** full_vox_path_length))  # solve equation
    return abs(cum_bs_in - beam_section_out_sum) / beam_section_out_sum  # i dont know why this is divided again by beam_section_out_sum implemented like this in amapvox, should anyway propagate towards the desired value

@jit(nopython=True, fastmath=True, parallel=True, cache=True, debug=False)  # fastmath safe
def transmittance_grid_search(PFEnt, PFOut, S, full_vox_path_length):
    '''python implementation of amapvox grid search algorithmm'''

    beam_section_in = PFEnt * S
    beam_section_in_sum = np.sum(beam_section_in)
    beam_section_out_sum = np.sum(PFOut * S)

    # unsampled voxel - should not be the case
    if beam_section_in_sum == 0:
        return np.nan

    # only passing rays -> no light interception, transmittance = 1
    if beam_section_in_sum == beam_section_out_sum:
        return 1.0

    # all rays intercepted -> no light transmitted, transmittance = 0
    if abs(beam_section_out_sum) < 1e-5: # beam_section_out out precision
        return 0.0


    # initialize value
    transmittance = np.nan  # transmittance
    transmittance_min = 0.0  # lower grid search bound
    transmittance_max = 1.0  # upper grid search bound
    incr = 0.1  # grid step size
    err = np.inf  # initialize error
    transm_err = 10e-7  # transmittance error precision


    # grid search, converted while to for loop
    for n in range(20):  # iter=20, (former 200, placed n+=1 outside inner-for-loop, reduced therefore n/10)

        grid = np.arange(transmittance_min, transmittance_max + incr, incr)
        err_tmp = np.full(grid.shape[0], np.inf)

        for i in prange(grid.shape[0]): # iterate over grid
            tr_tmp = grid[i]
            err_tmp[i] = err_transmittance_estimate(tr_tmp, beam_section_out_sum, beam_section_in, full_vox_path_length)

        min_idx = np.argmin(err_tmp)
        err_tmp_min = err_tmp[min_idx]  # get smallest error
        tr_tmp_estimated = grid[min_idx]  # get best transmittance estimation

        if err_tmp_min < err:  # if new error is smaller, assign
            err = err_tmp_min  # new err val
            transmittance = tr_tmp_estimated  # new transmittance value

        if err <= transm_err:
            break

        transmittance_min = max([transmittance - incr, 0.0])  # place new interval around current transmittance value
        transmittance_max = min([transmittance + incr, 1.0])
        incr /= 10.0  # decrease grid step size


    # # grid search original implementation
    # while err > transm_err and n < 200:  # iter=200, initialize err with max float value

    #     for tr_tmp in np.arange(transmittance_min, transmittance_max + incr, incr):  # no need for for-loop in python, calculate for all grids
    #         err_tmp = err_transmittance_estimate(tr_tmp, beam_section_out_sum, beam_section_in, full_vox_path_length)  # cannot run loop in parallel
    #         n += 1  # could think about placing in while loop, is more intuitive
    #         if err_tmp < err:  # if err_tmp smaller error
    #             err = err_tmp  # new err val
    #             transmittance = tr_tmp  # new tr val
    #     transmittance_min = max([transmittance - incr, 0.0])  # place new interval around current transmittance value
    #     transmittance_max = min([transmittance + incr, 1.0])
    #     incr /= 10.0  # decrease grid step size

    return transmittance

@jit(nopython=True, fastmath=True, cache=True, debug=False)  # fastmath possible since no NaN/Inf input dependence
def transmittance_bisection(PFEnt, PFOut, S, l, tolerance=1e-8, max_iterations=200):
    """
    Solve f(P) = sum(BFEnt*S*P**L) - sum(BFOut*S) = 0 using bisection
    should be faster than grid search used in Amapvox
    """

    # pre-calculate left side of term
    PFEnt_norm = PFEnt * S  #  incoming energy normalized with footprint
    PFEnt_norm_sum = np.sum(PFEnt_norm)  # sum

    # pre-calculate right side of term
    PFOut_norm_sum = np.sum(PFOut * S)  # outgoing energy normalized with footprint


    # unsampled voxel - should not be the case
    if PFEnt_norm_sum == 0:
        return np.nan

    # only passing rays -> no light interception, transmittance = 1
    if PFEnt_norm_sum == PFOut_norm_sum:
        return 1.0

    # all rays intercepted -> no light transmitted, transmittance = 0
    if abs(PFOut_norm_sum) < 1e-5: # precision as used in amapvox
        return 0.0


    # initialize bracket
    lower_bound = 0
    upper_bound = 1


    # estimate transmittance with bisection loop, iteratively converge towards true value
    for _ in range(max_iterations):
        transmittance_estimated = (lower_bound + upper_bound) / 2  # initialize transmittance estimation
        f_transmittance = np.sum(PFEnt_norm * (transmittance_estimated ** l)) - PFOut_norm_sum  # solve equation with estimated transmittance value

        # if below tolerance, transmittance found
        if abs(f_transmittance) < tolerance:
            return transmittance_estimated  # return transmittance

        # set new boundaries
        if f_transmittance < 0:  # if smaller
            lower_bound = transmittance_estimated  # value is larger previous estimation
        else:
            upper_bound = transmittance_estimated  # value is smaller previous estimation

    return (lower_bound + upper_bound) / 2  # if max iterations reached, return estimated value

@jit(nopython=True, parallel=True, cache=True, debug=False)
def transmittance_vincent2018(
    initial_beam_diameter,
    beam_div_semi_major,
    beam_div_semi_minor,
    vox_space_shape,
    vox_filled_idx_unique,
    ray_ids_vox_sorted,
    first_vox_idx,
    last_vox_idx,
    vox_min,
    vox_max,
    vox_center,
    start_coordinate,
    end_coordinate,
    first_idx,
    last_idx,
    return_vector_length,
    PF,
    ray_count_dtype = np.uint64
    ):
    """transmittance estimation as described in Vincent2018"""


    # extend end coordinate along trajectory to ground
    end_coordinate_extended = extend_trajectory_to_height(start_coordinate, end_coordinate, 0)

    # vox space for transmittance values
    P_transmittance = np.full((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), np.nan, dtype=np.float64)  # initialize vox_ID with non-existing pointer

    # vox space for explored voxel volume
    volume_explored = np.full((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), np.nan, dtype=np.float64)  # initialize vox_ID with non-existing pointer

    # vox space for ray_count
    ray_count = np.zeros((vox_space_shape[2], vox_space_shape[1], vox_space_shape[0]), dtype=ray_count_dtype)  # initialize vox_ID with non-existing pointer


    for i in prange(vox_filled_idx_unique.shape[0]):

        # get ray ids
        ray_ids = ray_ids_vox_sorted[first_vox_idx[i]:last_vox_idx[i]] # np.unique(ray_ids_vox_sorted[first_vox_idx[i]:last_vox_idx[i]])  # ensure only unique ray ids per voxel

        # get vox idx
        vox_idx = vox_filled_idx_unique[i]  # x,y,z

        # n - number of rays in voxel
        n = 0

        # voxel boundary
        vox_boundary = np.vstack((vox_min[i], vox_max[i]))  # voxel boundary [min, max], use [i] instead of [vox_id] since vox_id values higher than actual number of vox_ids


        # initialize vox vol
        vox_vol_explored = 0
        vox_vol_explored_extended = 0

        # initialize storayge for ray values
        PFEnt = np.full(ray_ids.shape[0], np.nan, dtype=np.float64)
        PFOut = np.full(ray_ids.shape[0], np.nan, dtype=np.float64)
        S = np.full(ray_ids.shape[0], np.nan, dtype=np.float64)
        l = np.full(ray_ids.shape[0], np.nan, dtype=np.float64)
        l_extended = np.full(ray_ids.shape[0], np.nan, dtype=np.float64)


        for r_i in range(ray_ids.shape[0]):  # inner loop runs sequentially anyway


            # get ray id
            ray_id = ray_ids[r_i]  # ensure only unique ray ids


            # ray voxel intersection
            vox_enter, vox_exit = ray_box_intersect(start_coordinate[ray_id], end_coordinate[ray_id], vox_boundary)

            # ray voxel intersection extended
            vox_enter_extended, vox_exit_extended = ray_box_intersect(start_coordinate[ray_id], end_coordinate_extended[ray_id], vox_boundary)


            if vox_enter is None:
                # should theoretically never happen since beams were mapped in voxel.
                # might happen due to inprecision of voxel traversal algorithms that voxels
                # are mapped where beam did not pass.
                continue  # skip ray


            # get ray return vector length
            ray_first_return_idx = first_idx[ray_id].item()  # index of first return
            ray_last_return_idx = last_idx[ray_id].item() + 1  # index of last return - if it is from -> to: need to add +1 for correct indexing
            ray_returns_vl = return_vector_length[ray_first_return_idx:ray_last_return_idx]  # vector length from pulse origin to returns


            # l_i - length of pulse optical path (in voxel)
            l_i = vector_length_nb(vox_enter, vox_exit)
            l[r_i] = l_i


            # l_i_extended - length of full pulse optical path (in voxel)
            l_i_extended = vector_length_nb(vox_enter, vox_exit_extended)
            l_extended[r_i] = l_i_extended


            # PF - Pulse fractions of ray
            PF_ray = np.append(PF[ray_first_return_idx:ray_last_return_idx], 0)  # append 0 to include PFOut of last return


            # PFEnt - entering fraction of pulse
            vox_enter_vector_length = vector_length_nb(start_coordinate[ray_id], vox_enter)
            vox_enter_vector_length = np.round(vox_enter_vector_length, 5)  # ray_returns_vl lost some precision due to scaling/rescaling, therefore round everything to 5 digits
            PFEnt_i = PF_ray[np.searchsorted(ray_returns_vl, vox_enter_vector_length, side="left")]
            PFEnt[r_i] = PFEnt_i


            # PFOut - exiting fraction of pulse
            vox_exit_vector_length = vector_length_nb(start_coordinate[ray_id], vox_exit)
            vox_exit_vector_length = np.round(vox_exit_vector_length, 5)  # ray_returns_vl lost some precision due to scaling/rescaling, therefore round everything to 5 digits
            PFOut_i = PF_ray[np.searchsorted(ray_returns_vl, vox_exit_vector_length, side="right")]
            PFOut[r_i] = PFOut_i


            # S_i - cross section of pulse at voxel center
            vox_center_vector_length = vector_length_nb(start_coordinate[ray_id], vox_center[i])
            radius1 = beam_radius(distance=vox_center_vector_length, divergence=beam_div_semi_major, initial_diameter=initial_beam_diameter)  # meter
            radius2 = beam_radius(distance=vox_center_vector_length, divergence=beam_div_semi_minor, initial_diameter=initial_beam_diameter)  # meter
            S_i = ellipse_area(radius1, radius2)  # sqm
            S[r_i] = S_i


            # volume explored
            vox_vol_explored += S_i * l_i
            vox_vol_explored_extended += S_i * l_i_extended

            n += 1  # count beams, some may not actually pass this voxel due to floating point precision errors of ray tracing algorithm


        if n == 0:
            continue  # keep P_vox initialized with np.nan


        # remove nan
        mask_invalid = np.isnan(PFEnt)

        if np.any(mask_invalid):
            PFEnt = PFEnt[~mask_invalid]  # remove nans
            PFOut = PFOut[~mask_invalid]
            S = S[~mask_invalid]
            l = l_extended[~mask_invalid]
            l_extended = l_extended[~mask_invalid]

            if PFEnt.size == 0:
                continue  # no rays left in this voxel


        # calculate / estimate transmissivity
        P_vox = transmittance_bisection(PFEnt, PFOut, S, l_extended)  # l or l_extended


        # store result
        P_transmittance[vox_idx[2], vox_idx[1], vox_idx[0]] = P_vox

        # if return_volume_explored:
        volume_explored[vox_idx[2], vox_idx[1], vox_idx[0]] = vox_vol_explored_extended  ### attention: vox vol extended returned

        # if return_ray_count:
        ray_count[vox_idx[2], vox_idx[1], vox_idx[0]] = n


    return P_transmittance[::-1,::-1,:], volume_explored[::-1,::-1,:], ray_count[::-1,::-1,:]


# voxel mapping helper functions
@jit(nopython=True, parallel=True, cache=True)
def ray_box_intersection_mask(rays_start_coordinates, rays_end_coordinates, boundary):
    """mask rays intersecting intersecting boundary box"""

    ### check
    assert rays_start_coordinates.shape == rays_end_coordinates.shape
    ray_count = rays_start_coordinates.shape[0]  # number of rays


    ### storage
    print("initialize storage")
    intersection_mask = np.full(ray_count, 1, dtype=np.bool_)  # bool


    ### counter of no intersection
    no_intersection_count = 0


    ### for each ray
    print("iterate through rays")
    for idx in prange(ray_count):

        # ray
        start_coordinate = rays_start_coordinates[idx]
        end_coordinate = rays_end_coordinates[idx]

        if np.any(np.isnan(start_coordinate)) or np.any(np.isnan(end_coordinate)):
            intersection_mask[idx] = False  # no intersection
            no_intersection_count +=1

        else:
            origin, end = ray_box_intersect(start_coordinate, end_coordinate, boundary)  # intersection

            if origin is None:
                intersection_mask[idx] = False  # no intersection
                no_intersection_count +=1


    stats = {"Number of Intersection": ray_count - no_intersection_count,
            "Number of Rays" : ray_count}


    return intersection_mask, stats

@jit(nopython=True, parallel=True, cache=True)
def vox_trav_rayfilledvoxpasscount(rays_start_coordinates, rays_end_coordinates, boundary, cell_size, vox_filled):
    '''count the number of filled voxels each ray passes'''
    # 1) count how many filled voxels a beam traverses
    # create storage with size of beams, each one needs to hold only one int value (np. int 16 should be sufficient)
    # if beam passes filled voxel: count +=1


    # check
    assert rays_start_coordinates.shape == rays_end_coordinates.shape
    ray_count = rays_start_coordinates.shape[0]  # number of rays


    ## regular grid boundary
    print("boundary calculation")
    boundary, nb_cell = vox_aoi(boundary, cell_size)  # match aoi with vox grid


    # storage
    print("initialize storage")
    filled_vox_pass_count = np.zeros(ray_count, dtype=np.uint16)  # unsigned int (0 to 65_535), should be enough


    # counter
    no_intersection_count = 0  # number of no intersection
    filled_vox_interaction_count = 0  # number of filled vox intersection


    # tMax accuracy to remove np.around and reduce non-deterministic floating-point evaluation
    EPS = 1e-12

    ### voxel traversal - for each ray
    print("iterate through rays")
    for idx in prange(ray_count):

        filled_vox_interaction = 0

        # ray
        start_coordinate = rays_start_coordinates[idx]
        end_coordinate = rays_end_coordinates[idx]


        # intersection
        origin, end = ray_box_intersect(start_coordinate, end_coordinate, boundary)

        if origin is None:
            no_intersection_count +=1
            continue  # no intersection


        # vector
        origin, end = origin.copy(), end.copy()
        direction = end - origin


        # vox ID initialization
        point_voxID_img_space = np.floor((origin - boundary[0]) / cell_size).astype(np.int_)


        # initialize tDelta, tMax, step
        tDeltaX, tMaxX = vox_trav_initialization(origin[0], direction[0], cell_size)
        tDeltaY, tMaxY = vox_trav_initialization(origin[1], direction[1], cell_size)
        tDeltaZ, tMaxZ = vox_trav_initialization(origin[2], direction[2], cell_size)
        step = initialize_step(direction)


        # start voxel
        X_img = int(point_voxID_img_space[0])  # img space
        Y_img = int(point_voxID_img_space[1])  # img space
        Z_img = int(point_voxID_img_space[2])  # img space


        ### append start voxel
        if (0 <= X_img < nb_cell[0]) and (0 <= Y_img < nb_cell[1]) and (0 <= Z_img < nb_cell[2]):
            '''in some cases of tDetla=inf first vox is appended twice. this doesnt really matter for mapping'''

            # count filled vox
            if vox_filled[Z_img, Y_img, X_img] == 1:  # fast lookup
                filled_vox_pass_count[idx] += 1  # count initial voxel
                filled_vox_interaction = 1  # count if beam interacted with filled vox


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
                # count filled vox
                if vox_filled[Z_img, Y_img, X_img] == 1:  # fast lookup
                    filled_vox_pass_count[idx] += 1  # count voxel
                    filled_vox_interaction = 1  # count if beam interacted with filled vox

            # check if max distance reached
            if (tMaxX >= 1.0 - EPS) and (tMaxY >= 1.0 - EPS) and (tMaxZ >= 1.0 - EPS):  # stop by distance, floating point precision error
                break

        if filled_vox_interaction == 1:  # count if beam interacted with filled vox
            filled_vox_interaction_count += 1

    stats = {"Number of Intersection": ray_count - no_intersection_count,
              "Number of Rays": ray_count,
              "Number of Rays that interacted with filled voxels": filled_vox_interaction_count}


    return filled_vox_pass_count, boundary, stats

@jit(nopython=True, parallel=True, cache=True)
def vox_trav_rayfilledvoxidmapping(rays_start_coordinates, rays_end_coordinates, boundary, cell_size, vox_filled, filled_vox_pass_max, sentinel_value, data_type):
    '''map filled vox IDs a ray passes'''
    # 2) map filled vox ids a ray passes through
    # storage = number of rays * max(number of fileld voxels traversed), np int ... dependent on number of filled voxels
    # map for each ray the filled vox IDs it passed



    # check
    assert rays_start_coordinates.shape == rays_end_coordinates.shape
    ray_count = rays_start_coordinates.shape[0]  # number of rays


    # regular grid boundary
    print("boundary calculation")
    boundary, nb_cell = vox_aoi(boundary, cell_size)  # match aoi with vox grid


    # storage
    print("initialize storage")
    vox_filled_idx = np.where(vox_filled == 1)  # idx of filled voxels
    vox_filled_count = vox_filled_idx[0].shape[0]  # number of filled voxels

    # vox space with vox IDs stored
    vox_idx = np.full((nb_cell[2], nb_cell[1], nb_cell[0]), sentinel_value, dtype=data_type)  # initialize vox_ID with non-existing pointer

    # assign vox IDs to vox array - multi indexing at once does not work in numba, therefore assignment with parallel loop
    for idx in prange(vox_filled_count):
        vox_idx[vox_filled_idx[0][idx],vox_filled_idx[1][idx],vox_filled_idx[2][idx]] = idx #c[idx]  # assign vox IDs to vox space (is this necessary to have voxel space??)


    # initialize storage of rays -> vox IDs - this is rather a bottleneck for memory allocation
    # !!!! this can be optimized by creating array with size only number of beams interacting with filled voxels.... maybe extra field for ray ID to make conversion easier
    print("initialize storage")
    filled_vox_mapping = np.full((ray_count, filled_vox_pass_max), sentinel_value, dtype=data_type)  # same as vox_id, filled_vox_pass_max, initialize with non existing pointer

    # counter
    no_intersection_count = 0

    # tMax accuracy to remove np.around and reduce non-deterministic floating-point evaluation
    EPS = 1e-12

    ### voxel traversal - for each ray
    print("iterate through rays")
    for idx in prange(ray_count):

        # idx
        vox_filled_counter = 0


        # ray
        start_coordinate = rays_start_coordinates[idx]
        end_coordinate = rays_end_coordinates[idx]


        # intersection
        origin, end = ray_box_intersect(start_coordinate, end_coordinate, boundary)

        if origin is None:
            no_intersection_count +=1
            continue  # no intersection


        # vector
        origin, end = origin.copy(), end.copy()
        direction = end - origin


        # vox ID initialization
        point_voxID_img_space = np.floor((origin - boundary[0]) / cell_size).astype(np.int_)


        # initialize tDelta, tMax, step
        tDeltaX, tMaxX = vox_trav_initialization(origin[0], direction[0], cell_size)
        tDeltaY, tMaxY = vox_trav_initialization(origin[1], direction[1], cell_size)
        tDeltaZ, tMaxZ = vox_trav_initialization(origin[2], direction[2], cell_size)
        step = initialize_step(direction)


        # start voxel
        X_img = int(point_voxID_img_space[0])  # img space
        Y_img = int(point_voxID_img_space[1])  # img space
        Z_img = int(point_voxID_img_space[2])  # img space


        ### append start voxel
        if (0 <= X_img < nb_cell[0]) and (0 <= Y_img < nb_cell[1]) and (0 <= Z_img < nb_cell[2]):
            '''in some cases of tDetla=inf first vox is appended twice. this doesnt realy matter for mapping'''

            # add filled vox id
            if vox_filled[Z_img, Y_img, X_img] == 1:  # fast lookup - is vox filled?
                if vox_filled_counter < filled_vox_pass_max:  # prevent out of bounds writing
                    filled_vox_mapping[idx, vox_filled_counter] = vox_idx[Z_img, Y_img, X_img]  # assign vox id to ray
                    vox_filled_counter += 1  # increment pointer


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

                # add filled vox id
                if vox_filled[Z_img, Y_img, X_img] == 1:  # fast lookup - is vox filled?         
                    if vox_filled_counter < filled_vox_pass_max:  # prevent out of bounds writing        
                        filled_vox_mapping[idx, vox_filled_counter] = vox_idx[Z_img, Y_img, X_img]  # assign vox id to ray
                        vox_filled_counter += 1  # increment pointer


            # check if max distance reached
            if (tMaxX >= 1.0 - EPS) and (tMaxY >= 1.0 - EPS) and (tMaxZ >= 1.0 - EPS):  # stop by distance, floating point precision error
                break


    stats = {"Number of Intersection": ray_count - no_intersection_count,
              "Number of Rays": ray_count}

    return filled_vox_mapping, boundary, stats


