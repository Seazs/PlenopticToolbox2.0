"""
The main architecture of the disparity estimation algorithm is here:
the structure containing the microimage is used to calculate per-image disparity map
the final images are returned
----
@veresion v1.1 - Januar 2017
@author Luca Palmieri
"""
import disparity.disparity_calculation as rtxdisp
import plenopticIO.lens_grid as rtxhexgrid
import rendering.render as rtxrender
import plenopticIO.imgIO as rtxIO
import disparity.sgm as rtxsgm
import numpy as np
import argparse
import math
import os
import json
import pdb
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
#import matplotlib.image


def estimate_disp(args):

    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T

    rings = [int(i) for i in args.use_rings.split(',')]
    
    nb_offsets = []
    for i in rings: 
        nb_offsets.extend(rtxhexgrid.HEX_OFFSETS[i])
    
    scene_type = args.scene_type
    print("******************\nLoading the scene..\n")
    #pdb.set_trace()
    if args.differentNames == False:
        lenses = rtxIO.load_scene(args.filename, args.analyze_err)
    else:
        lenses = rtxIO.load_scene_diffNames(args.filename, args.configfilename)
    
    diam = lenses[0, 0].diameter
    max_disp = float(args.max_disp) 
    min_disp = float(args.min_disp) 
    num_disp = float(args.num_disp)
    #pdb.set_trace()
    #max_lens_dist = np.linalg.norm(np.dot(B, rtxhexgrid.HEX_OFFSETS[args.max_ring][0]))
    
    disparities = np.arange(min_disp, max_disp, (max_disp - min_disp) / num_disp) #16.0 / max_lens_dist)
    #disparities = np.arange(min_disp, 0.7 * diam, 4.0 / max_lens_dist)
    print("Disparities: {0}".format(disparities))
    
    strategy_args = dict()
    selection_strategy = None
    use_torch = getattr(args, "use_torch", False)
    torch_device = getattr(args, "torch_device", "auto")
    torch_interp = getattr(args, "torch_interp", "bilinear")
    torch_batch = getattr(args, "torch_batch", 1)
    torch_cache = getattr(args, "torch_cache", False)
    sgm_only_dp = getattr(args, "sgm_only_dp", False)
    compute_conf = getattr(args, "compute_conf", True)
    sgm_workers = getattr(args, "sgm_workers", 0)
    cost_workers = getattr(args, "cost_workers", 0)
    timing = getattr(args, "timing", False)

    t_cost_start = time.perf_counter() if timing else None

    if args.method == 'plain':
        fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons  = calc_costs_plain(
            lenses,
            disparities,
            nb_offsets,
            args.max_cost,
            args.technique,
            hws=args.match_hws,
            use_torch=use_torch,
            torch_device=torch_device,
            torch_interp=torch_interp,
            torch_batch=torch_batch,
            torch_cache=torch_cache,
            num_workers=cost_workers,
        )
    elif args.method == 'real_lut':
        strategy_args = dict()
        strategy_args['target_lenses'] = _precalc_angular()
        strategy_args['min_disp'] = min_disp
        strategy_args['max_disp'] = max_disp
        strategy_args['trade_off'] = args.lut_trade_off
        selection_strategy = real_lut
        
    if selection_strategy is not None:
        print("Selection strategy: {0}".format(selection_strategy))
        print("\nStep 1) Calculating the costs..\n")
        fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons, disp_avg = calc_costs_selective_with_lut(
            lenses,
            disparities,
            selection_strategy,
            args.technique,
            nb_args=strategy_args,
            refine=args.refine,
            max_cost=args.max_cost,
            hws=args.match_hws,
            use_torch=use_torch,
            torch_device=torch_device,
            torch_interp=torch_interp,
            torch_batch=torch_batch,
            torch_cache=torch_cache,
            num_workers=cost_workers,
        )

    if timing:
        t_cost_end = time.perf_counter()
        print("Timing: cost volume {:.2f}s".format(t_cost_end - t_cost_start))

    if args.coarse is True:
        coarse_disp = regularize_coarse(lenses, coarse_costs_merged, disparities, penalty1=args.coarse_penalty1, penalty2=args.coarse_penalty2)
        fine_costs = augment_costs_coarse(fine_costs, coarse_disp, lens_variance, disparities, coarse_weight=args.coarse_weight, struct_var=args.struct_var)
    
    #pdb.set_trace()
    print("\nStep 2) Regularizing and extracing disparity map..\n")
    t_reg_start = time.perf_counter() if timing else None
    fine_disps, fine_disps_interp, fine_val, wta_depths, wta_depths_interp, wta_val, confidence = regularized_fine(
        lenses,
        fine_costs,
        disparities,
        args.penalty1,
        args.penalty2,
        args.max_cost,
        conf_tec=args.confidence_technique,
        conf_sigma=args.conf_sigma,
        only_dp=sgm_only_dp,
        compute_conf=compute_conf,
        num_workers=sgm_workers,
    )
    if timing:
        t_reg_end = time.perf_counter()
        print("Timing: SGM + labels {:.2f}s".format(t_reg_end - t_reg_start))
       
    Dsgm = rtxrender.render_lens_imgs(lenses, fine_disps_interp)
    Dwta = rtxrender.render_lens_imgs(lenses, wta_depths_interp)
    
    lens_data = dict()
    col_data = dict()
    if args.analyze_err:
        gt_disp = dict()
        
    for lcoord in lenses:
        lens_data[lcoord] = lenses[lcoord].img
        col_data[lcoord] = lenses[lcoord].col_img
        if args.analyze_err:
            gt_disp[lcoord] = lenses[lcoord].disp_img
        
    I = rtxrender.render_lens_imgs(lenses, lens_data)
    Dconf = rtxrender.render_lens_imgs(lenses, confidence)
    Icol = rtxrender.render_lens_imgs(lenses, col_data)
    new_offset = [lenses[0,0].pcoord[0] - (Icol.shape[0]/2), lenses[0,0].pcoord[1] - (Icol.shape[1]/2)]

    Dcoarse = None
    if args.coarse is True:
        Dcoarse = rtxrender.render_lens_imgs(lenses, coarse_disp)
        
    if args.analyze_err:
        # if gt_disp is empty (if you are trying to evaluate error but didn't provide a valid disparity map)
        # it will throw an error the rendering method (line 100)
        Dgt = rtxrender.render_lens_imgs(lenses, gt_disp)
        error_measurements = analyze_disp(lenses, fine_disps_interp, True)
    else:
        Dgt = None
        sgm_err = None
        wta_err = None
        sgm_err_mask = None
        sgm_err_mse = None
        err_img_r = None
        img_s = None
        error_measurements = None

    return Icol, Dsgm, Dwta, Dgt, Dconf, Dcoarse, disparities, num_comparisons, disp_avg, new_offset, error_measurements, lenses[0,0]

def _has_neighbours(lens, lenses, neighbours):
        
    for l in neighbours:
        if tuple(l + lens.lcoord) not in lenses:
            return False
    
    return True

def get_depth_discontinuities(lenses):
    """
    It computes via Canny (opencv implementation) the discontinuities on the ground truth
    It returns two dictionaries with image masks (one discontinuities, one smooth areas)
    After Canny dilation is used to get more consistent border (ideally 3 pixels large edges)
    ---
    January 2018
    """
    disc = dict()
    smooth = dict()
    canny_thr_low = 100 # threshold for canny method
    canny_thr_high = 200 # threshold for canny method
    kernel_size = 3 # one pixel dilation in every direction (3x3 matrix)
    iterations_dilate = 1 # one time dilation --> one pixel is enough
    uint8_norm_value = 255 # just switching [0,1] to [0,255] and back
    for key in lenses:
        current_disp = lenses[key].disp_img
        norm_disp = current_disp / np.max(current_disp)
        int_disp = np.uint8(norm_disp * uint8_norm_value)
        canny = cv2.Canny(int_disp, canny_thr_low, canny_thr_high)
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        dilation = cv2.dilate(canny,kernel,iterations = iterations_dilate)
        disc[key] = dilation / uint8_norm_value
        smooth[key] = (uint8_norm_value - dilation) / uint8_norm_value
    
    return disc, smooth
    
def analyze_disp(lenses, est_depths, depth_discontinuities=False, max_ring=5):

    """
    Used only on synthetic images
    Loop through the estimated depth and calculate the following error measurements:
    - average error
    - mean squared error (MSE)
    - standard deviation
    - BadPix 1.0 and 2.0 (% of pixels with error higher than 1% or 2%
    - (Error on depth discontinuities if the third parameter is True)
    """
    #pdb.set_trace()
    err_avg = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    err_mask = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    err_mse = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    bump = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    bump_thresh = 0.25 # bumpiness threshold - can be changed if disparity range varies 
    badPix1 = np.zeros((len(est_depths)))
    badPix2 = np.zeros((len(est_depths)))
    badPixGraph = np.zeros((len(est_depths), 21))
    badpixindex = 0
    
    # evaluate error on depth discontinuities
    badPix1Disc = np.zeros((len(est_depths)))
    badPix1Smooth = np.zeros((len(est_depths)))
    badPix2Disc = np.zeros((len(est_depths)))
    badPix2Smooth = np.zeros((len(est_depths)))
    badPixGraphDisc = np.zeros((len(est_depths), 21))
    badPixGraphSmooth = np.zeros((len(est_depths), 21))
    disc, smooth = get_depth_discontinuities(lenses)
    avgErrDisc = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    avgErrSmooth = {0: np.array([]), 1: np.array([]), 2: np.array([])}
    
    nb_tmp = []
    for ring in range(max_ring):
        nb_tmp.extend(rtxhexgrid.HEX_OFFSETS[ring])
        
    #pdb.set_trace()    
    for lcoord in est_depths:
        if _has_neighbours(lenses[lcoord], lenses, nb_tmp) is False:
            continue
    
        est = est_depths[lcoord]
        gt = lenses[lcoord].disp_img
        ind = est > 0
        #pdb.set_trace()
        ft = lenses[lcoord].focal_type
        lens = lenses[lcoord]
        mask = ((lens.grid.xx**2 + lens.grid.yy**2) <= lens.inner_radius**2)       
        err_avg[ft] = np.append(err_avg[ft], np.ravel(np.abs(est[mask] - gt[mask])))
        #pdb.set_trace()
        abs_diff = np.ravel(np.abs(est[mask] - gt[mask]))
        err_mask[ft] = np.append(err_mask[ft], abs_diff)
        bumpy = np.clip(abs_diff, 0, bump_thresh)
        bump[ft] = bumpy
        err_mse[ft] = np.append(err_mse[ft], np.ravel( (est[mask] - gt[mask])**2 ))
        err_img_cur = np.abs(est - gt)
        # mask
        to_zero = ((lens.grid.xx**2 + lens.grid.yy**2) > lens.inner_radius**2)
        err_img_cur[to_zero] = 0
        badPix1[badpixindex] += len(np.where(err_img_cur > 1)[0])
        badPix2[badpixindex] += len(np.where(err_img_cur > 2)[0])
        #pdb.set_trace()
        for i in range(0,21):
            badPixGraph[badpixindex, i] += len(np.where(err_img_cur > 0.1*i)[0])
        
        if depth_discontinuities:
            #pdb.set_trace()
            disc_err = err_img_cur[disc[lcoord] > 0.5]
            smth_err = err_img_cur[disc[lcoord] < 0.5]
            #print(len(disc_err), len(smth_err), len(np.ravel( (est[mask] - gt[mask])**2 )))
            avgErrDisc[ft] = np.append(avgErrDisc[ft], disc_err)
            avgErrSmooth[ft] = np.append(avgErrSmooth[ft], smth_err)
            badPix1Disc[badpixindex] += len(np.where(disc_err > 1)[0])
            badPix1Smooth[badpixindex] += len(np.where(smth_err > 1)[0])
            badPix2Disc[badpixindex] += len(np.where(disc_err > 2)[0])
            badPix2Smooth[badpixindex] += len(np.where(smth_err > 2)[0])
            for i in range(0,21):
                badPixGraphDisc[badpixindex, i] += len(np.where(disc_err > 0.1*i)[0])
                badPixGraphSmooth[badpixindex, i] += len(np.where(smth_err > 0.1*i)[0])
        badpixindex += 1    
        
    #pdb.set_trace()    
    final_err = dict()
    final_err_mask = dict()
    final_err_mse = dict()
    bumpiness = dict()
    depth_disc = dict() 
    depth_smooth = dict()
    
    for key in err_avg:
    
        #pdb.set_trace()
        final_err[key] = dict()
        final_err[key]['err'] = np.mean(err_avg[key])
        #print(final_err[key]['err'])
        final_err[key]['std'] = np.std(err_avg[key])
        final_err[key]['max'] = np.max(err_avg[key])
        final_err[key]['num'] = np.mean(err_avg[key] >= (final_err[key]['err'] + 2*final_err[key]['std']))   
         


        final_err_mask[key] = dict()
        final_err_mask[key]['err'] = np.mean(err_mask[key])
        #print(final_err[key]['err'])
        final_err_mask[key]['std'] = np.std(err_mask[key])
        final_err_mask[key]['max'] = np.max(err_mask[key])
        final_err_mask[key]['num'] = np.mean(err_mask[key] >= (final_err_mask[key]['err'] + 2*final_err_mask[key]['std']))



        final_err_mse[key] = dict()
        final_err_mse[key]['err'] = np.mean(err_mse[key])
        #print(final_err[key]['err'])
        final_err_mse[key]['std'] = np.std(err_mse[key])
        final_err_mse[key]['max'] = np.max(err_mse[key])
        final_err_mse[key]['num'] = np.mean(err_mse[key] >= (final_err_mse[key]['err'] + 2*final_err_mse[key]['std']))
    
    

        bumpiness[key] = dict()
        bumpiness[key]['err'] = np.mean(bump[key])
        #print(final_err[key]['err'])
        bumpiness[key]['std'] = np.std(bump[key])
        bumpiness[key]['max'] = np.max(bump[key])
        bumpiness[key]['num'] = np.mean(bump[key] >= (bumpiness[key]['err'] + 2*bumpiness[key]['std']))  
    
       
        if depth_discontinuities:
 
            depth_disc[key] = dict()
            depth_disc[key]['err'] = np.mean(avgErrDisc[key])
            depth_disc[key]['std'] = np.std(avgErrDisc[key])
            depth_disc[key]['max'] = np.max(avgErrDisc[key])
            depth_smooth[key] = dict()
            depth_smooth[key]['err'] = np.mean(avgErrSmooth[key])
            depth_smooth[key]['std'] = np.std(avgErrSmooth[key])
            depth_smooth[key]['max'] = np.max(avgErrSmooth[key])
        
    return final_err, final_err_mask, final_err_mse, [badPix1, badPix2, badPixGraph], bumpiness, [depth_disc, depth_smooth, badPix1Disc, badPix2Disc, badPix1Smooth, badPix2Smooth, badPixGraphDisc, badPixGraphSmooth]
    
def _rel_to_abs(lcoord, lenses, offsets):

    """
    Generate the axial coordinates for the lens lcoord from the given nb offsets
    """
    
    elements = [lenses.get((lcoord[0] + d[0], lcoord[1] + d[1])) for d in offsets]
    return [x for x in elements if x is not None]
    
def _precalc_angular():
    
    # hex basis
    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T

    # next 6 neighbours
    ring1 = rtxhexgrid.HEX_OFFSETS[1]
    
    eps = 0.0001
    
    l = dict()
    
    for src in ring1:
        # directional vector in the rectangular coordinate system
        v = np.dot(B, src)
        v = v / np.linalg.norm(v)
        l[tuple(src)] = []

        # cosine of the angle between w an v
        for i, ring in enumerate(reversed(rtxhexgrid.HEX_OFFSETS)):
            if i == 1:
                continue
            tmp = []
            for dst in ring:
                w = np.dot(B, dst)
                w = w / np.linalg.norm(w)

                # k = cosine of the angle between w and v
                k = np.dot(v, w)

                # use only lenses within the correct sector
                if k < 0 or k < np.cos(np.pi/6.0):
                    continue

                tmp.append(dst)
                
            l[tuple(src)].append(tmp)

    return l

def real_lut(lens, lenses, coarse_costs, disparities, max_cost=10.0, nb_args=None):
    
    # get some parameters
    target_lenses = list(nb_args['target_lenses'].values())
    min_disp = nb_args['min_disp']
    max_disp = nb_args['max_disp']
    trade_off = nb_args['trade_off']
    
    
    # LUT to pick best combination or best performances
    B = np.array([[np.sqrt(3)/2, 0.5], [0, 1]]).T
    
    #assert len(coarse_costs) <= len(target_lenses)
    
    offsets = []

    tref = rtxhexgrid.hex_focal_type(lens.lcoord)
    
    #read the lut    
    lut_filename = '../disparity/lut_table.json'
    with open(lut_filename, 'r') as f:
        lut_str = json.load(f)
    
    lut_length = len(lut_str['most_acc_0'])
    lut_step = (max_disp - min_disp) / lut_length
    mavg = 0
    mvavg = 0
    counter = 0
    
    for i, ctmp in enumerate(coarse_costs):
        
        m, mval = rtxdisp.cost_minimum_interp(ctmp, disparities)
        
        mvavg += mval
        mavg += m
        counter += 1
    
    # m avg should be the value for the disparity    
    mavg /= (counter)
    mvavg /= counter
    
    # look for the correct index
    disp_int = lut_str['disp_int_interp']
    disp_int['0'][0] = 20.0 #disparities[len(disparities)-1]
    disp_int[str(len(disp_int)-1)][1] = 0.0
    disp_vals = lut_str['disp_vals_interp']
    found = False
    finished = False
    jj = 0
    while (not found and not finished):
        if jj >= len(disp_int):
            finished = True
            jj = 0
        elif mavg < disp_int[str(jj)][0] and mavg > disp_int[str(jj)][1]:
            found = True
        else:
            jj += 1
    
    ### need to show somehow if I didn't find it
    #if (finished and not found):
        #print("Not found! \nmavg={0}\ndisp_int={1}\ndisp_vals={2}\n".format(mavg, disp_int, disp_vals))
    index_lut = jj #lut_length - math.floor(m / (lut_step)) 
    #print("m:{0}, mval:{1}, index:{2}".format(mavg, mvavg, jj))
    
    if trade_off == 1:
        if tref == 0:
            strat = lut_str['most_acc_0'][index_lut]    
        elif tref == 1:
            strat = lut_str['most_acc_1'][index_lut]
        elif tref == 2:
            strat = lut_str['most_acc_2'][index_lut]
    elif trade_off == 0:
        if tref == 0:
            strat = lut_str['best_perf_0'][index_lut]    
        elif tref == 1:
            strat = lut_str['best_perf_1'][index_lut]
        elif tref == 2:
            strat = lut_str['best_perf_2'][index_lut]
            
    targets = from_strat_to_offsets(strat)
        
    return targets , mavg

def from_strat_to_offsets(strat):

    if strat == 'f1':
        selection_strategy = fixed_selection_strategy_1()
    elif strat == 'f2':
        selection_strategy = fixed_selection_strategy_2()
    elif strat == 'f3':
        selection_strategy = fixed_selection_strategy_3()
    elif strat == 'f4':
        selection_strategy = fixed_selection_strategy_4()
    elif strat == 'f5':
        selection_strategy = fixed_selection_strategy_5()
    elif strat == 'f6':
        selection_strategy = fixed_selection_strategy_6()
    elif strat == 'f7':
        selection_strategy = fixed_selection_strategy_7()
    elif strat == 'f8':
        selection_strategy = fixed_selection_strategy_8()  
    elif strat == 'f9':
        selection_strategy = fixed_selection_strategy_9()  
    elif strat == 'f10':
        selection_strategy = fixed_selection_strategy_10()
    elif strat == 'f11':
        selection_strategy = fixed_selection_strategy_11()
    elif strat == 'f12':
        selection_strategy = fixed_selection_strategy_12()
    elif strat == 'f13':
        selection_strategy = fixed_selection_strategy_13()
    elif strat == 'f14':
        selection_strategy = fixed_selection_strategy_14()
        
    return selection_strategy

"""
The strategies from 1 to 15 are fixed strategies, used only for experimental purposes:
Here they are copied as fixed_strategies (1-8) in order to be able to use them without passing any parameters
For other purposes this part of the code is absolutely useless
"""
def fixed_selection_strategy_1():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_2():
    raytrix
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_3():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_4():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
            
    return [offset for offset in nb_offsets]

def fixed_selection_strategy_5():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[6])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[6][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[7])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[7][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_6():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[6])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[6][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[7])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[7][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                        
    return [offset for offset in nb_offsets]
    
def fixed_selection_strategy_7():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    return [offset for offset in nb_offsets]        
            
def fixed_selection_strategy_8():
    
    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]       
 
def fixed_selection_strategy_9():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                
    return [offset for offset in nb_offsets]  
    
def fixed_selection_strategy_10():

    """
raytrix
    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
                
    return [offset for offset in nb_offsets]  
    
def fixed_selection_strategy_11():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)

    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_12():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 

    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_13():

    """

    """
    
    nb_offsets = dict()
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[0])):
        o0 = tuple(rtxhexgrid.HEX_OFFSETS[0][i])
        if not o0 in nb_offsets:
            nb_offsets[tuple(o0)] = np.array(o0) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[1])):
        o1 = tuple(rtxhexgrid.HEX_OFFSETS[1][i])
        if not o1 in nb_offsets:
            nb_offsets[tuple(o1)] = np.array(o1)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def fixed_selection_strategy_14():

    """

    """
    
    nb_offsets = dict()
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[2])):
        o2 = tuple(rtxhexgrid.HEX_OFFSETS[2][i])
        if not o2 in nb_offsets:
            nb_offsets[tuple(o2)] = np.array(o2)
    
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[3])):
        o3 = tuple(rtxhexgrid.HEX_OFFSETS[3][i])
        if not o3 in nb_offsets:
            nb_offsets[tuple(o3)] = np.array(o3) 
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[4])):
        o4 = tuple(rtxhexgrid.HEX_OFFSETS[4][i])
        if not o4 in nb_offsets:
            nb_offsets[tuple(o4)] = np.array(o4)
            
    for i in range(0, len(rtxhexgrid.HEX_OFFSETS[5])):
        o5 = tuple(rtxhexgrid.HEX_OFFSETS[5][i])
        if not o5 in nb_offsets:
            nb_offsets[tuple(o5)] = np.array(o5)
                
    return [offset for offset in nb_offsets]  

def calc_costs_plain(
    lenses,
    disparities,
    nb_offsets,
    max_cost,
    technique,
    hws=1,
    progress_hook=print,
    use_torch=False,
    torch_device="auto",
    torch_interp="bilinear",
    torch_batch=1,
    torch_cache=False,
    num_workers=0,
):
    
    coarse_costs = dict()
    coarse_costs_merged = dict()
    fine_costs = dict()
    lens_variance = dict()

    num_lenses = len(lenses)
    num_comparisons = 0
    
    if num_workers is not None and num_workers > 1:
        progress_hook("Parallel cost volume is not enabled for method 'plain'; using single process.")

    for i, lcoord in enumerate(lenses):
        nb_lenses = _rel_to_abs(lcoord, lenses, nb_offsets)
        lens = lenses[lcoord]
        
        if i%5000==0:
            progress_hook("Processing lens {0}/{1} Coord: {2}".format(i, num_lenses, lcoord))

        fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(
            lens,
            nb_lenses,
            disparities,
            max_cost,
            technique,
            hws=hws,
            use_torch=use_torch,
            torch_device=torch_device,
            torch_interp=torch_interp,
            torch_batch=torch_batch,
            torch_cache=torch_cache,
        )
        
        coarse_costs_merged[lcoord] = coarse_merged
        coarse_costs[lcoord] = coarse
        fine_costs[lcoord], _ = np.array(rtxdisp.merge_costs_additive(fine, max_cost))
        lens_variance[lcoord] = lens_var
        num_comparisons += len(fine)
    
        rtxdisp.assign_last_valid(fine_costs[lcoord])
    return fine_costs, coarse_costs, coarse_costs_merged, lens_variance, num_comparisons

def regularize_coarse(lenses, coarse_costs, disparities, penalty1=0.08, penalty2=0.15, max_cost=10.0):

    intensity_grid = dict()
    tmp_costs = dict()
    coarse_disp = dict()

    for lcoord in lenses:
        intensity_grid[lcoord] = np.mean(lenses[lcoord].img[lenses[lcoord].mask > 0])

    sgm_cost = rtxsgm.hex_sgm(coarse_costs, intensity_grid, penalty1, penalty2, max_cost=max_cost)
    
    for lcoord in sgm_cost:
        coarse_disp[lcoord], _ = rtxdisp.cost_minimum_interp(sgm_cost[lcoord], disparities)
    return coarse_disp


_SGM_LENSES = None
_SGM_FINE_COSTS = None
_SGM_DISP = None
_SGM_PENALTY1 = None
_SGM_PENALTY2 = None
_SGM_MAX_COST = None
_SGM_CONF_TEC = None
_SGM_CONF_SIGMA = None
_SGM_ONLY_DP = None
_SGM_COMPUTE_CONF = None
_SGM_MIN_THRESH = None
_SGM_EPS = None

_COST_LENSES = None
_COST_DISPARITIES = None
_COST_MAX_COST = None
_COST_TECHNIQUE = None
_COST_HWS = None
_COST_USE_TORCH = None
_COST_TORCH_DEVICE = None
_COST_TORCH_INTERP = None
_COST_TORCH_BATCH = None
_COST_TORCH_CACHE = None
_COST_NB_OFFSETS = None
_COST_REFINE = None
_COST_NB_STRATEGY = None
_COST_NB_ARGS = None


def _sgm_worker_init(
    lenses,
    fine_costs,
    disp,
    penalty1,
    penalty2,
    max_cost,
    conf_tec,
    conf_sigma,
    only_dp,
    compute_conf,
    min_thresh,
    eps,
):

    global _SGM_LENSES
    global _SGM_FINE_COSTS
    global _SGM_DISP
    global _SGM_PENALTY1
    global _SGM_PENALTY2
    global _SGM_MAX_COST
    global _SGM_CONF_TEC
    global _SGM_CONF_SIGMA
    global _SGM_ONLY_DP
    global _SGM_COMPUTE_CONF
    global _SGM_MIN_THRESH
    global _SGM_EPS

    _SGM_LENSES = lenses
    _SGM_FINE_COSTS = fine_costs
    _SGM_DISP = disp
    _SGM_PENALTY1 = penalty1
    _SGM_PENALTY2 = penalty2
    _SGM_MAX_COST = max_cost
    _SGM_CONF_TEC = conf_tec
    _SGM_CONF_SIGMA = conf_sigma
    _SGM_ONLY_DP = only_dp
    _SGM_COMPUTE_CONF = compute_conf
    _SGM_MIN_THRESH = min_thresh
    _SGM_EPS = eps


def _cost_plain_worker_init(
    lenses,
    disparities,
    nb_offsets,
    max_cost,
    technique,
    hws,
    use_torch,
    torch_device,
    torch_interp,
    torch_batch,
    torch_cache,
):

    global _COST_LENSES
    global _COST_DISPARITIES
    global _COST_MAX_COST
    global _COST_TECHNIQUE
    global _COST_HWS
    global _COST_USE_TORCH
    global _COST_TORCH_DEVICE
    global _COST_TORCH_INTERP
    global _COST_TORCH_BATCH
    global _COST_TORCH_CACHE
    global _COST_NB_OFFSETS

    _COST_LENSES = lenses
    _COST_DISPARITIES = disparities
    _COST_MAX_COST = max_cost
    _COST_TECHNIQUE = technique
    _COST_HWS = hws
    _COST_USE_TORCH = use_torch
    _COST_TORCH_DEVICE = torch_device
    _COST_TORCH_INTERP = torch_interp
    _COST_TORCH_BATCH = torch_batch
    _COST_TORCH_CACHE = torch_cache
    _COST_NB_OFFSETS = nb_offsets


def _cost_selective_worker_init(
    lenses,
    disparities,
    nb_strategy,
    nb_args,
    max_cost,
    technique,
    hws,
    refine,
    use_torch,
    torch_device,
    torch_interp,
    torch_batch,
    torch_cache,
):

    global _COST_LENSES
    global _COST_DISPARITIES
    global _COST_MAX_COST
    global _COST_TECHNIQUE
    global _COST_HWS
    global _COST_USE_TORCH
    global _COST_TORCH_DEVICE
    global _COST_TORCH_INTERP
    global _COST_TORCH_BATCH
    global _COST_TORCH_CACHE
    global _COST_REFINE
    global _COST_NB_STRATEGY
    global _COST_NB_ARGS

    _COST_LENSES = lenses
    _COST_DISPARITIES = disparities
    _COST_MAX_COST = max_cost
    _COST_TECHNIQUE = technique
    _COST_HWS = hws
    _COST_USE_TORCH = use_torch
    _COST_TORCH_DEVICE = torch_device
    _COST_TORCH_INTERP = torch_interp
    _COST_TORCH_BATCH = torch_batch
    _COST_TORCH_CACHE = torch_cache
    _COST_REFINE = refine
    _COST_NB_STRATEGY = nb_strategy
    _COST_NB_ARGS = nb_args


def _cost_plain_worker(lcoord):

    lens = _COST_LENSES[lcoord]
    nb_lenses = _rel_to_abs(lcoord, _COST_LENSES, _COST_NB_OFFSETS)
    fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(
        lens,
        nb_lenses,
        _COST_DISPARITIES,
        _COST_MAX_COST,
        _COST_TECHNIQUE,
        hws=_COST_HWS,
        use_torch=_COST_USE_TORCH,
        torch_device=_COST_TORCH_DEVICE,
        torch_interp=_COST_TORCH_INTERP,
        torch_batch=_COST_TORCH_BATCH,
        torch_cache=_COST_TORCH_CACHE,
    )

    fine_merged = np.array(rtxdisp.merge_costs_additive(fine, _COST_MAX_COST))
    rtxdisp.assign_last_valid(fine_merged)

    return lcoord, fine_merged, coarse, coarse_merged, lens_var, len(fine)


def _cost_selective_worker(lcoord):

    lens = _COST_LENSES[lcoord]

    pos1 = [[-1, -1], [-1, 2], [1, 1], [1, -2], [0, -1], [0, 1]]
    pos2 = [[-1, -1], [-2, 1], [1, 1], [2, -1], [0, -1], [0, 1]]
    pos3 = [[-2, 1], [-1, 2], [2, -1], [1, -2], [0, -1], [0, 1]]

    if rtxhexgrid.hex_focal_type(lcoord) == 0:
        pos = rtxhexgrid.HEX_OFFSETS[1]
    elif rtxhexgrid.hex_focal_type(lcoord) == 1:
        pos = pos1
    elif rtxhexgrid.hex_focal_type(lcoord) == 2:
        pos = pos1
    else:
        pdb.set_trace()

    nb_lenses = _rel_to_abs(lcoord, _COST_LENSES, pos)

    fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(
        lens,
        nb_lenses,
        _COST_DISPARITIES,
        _COST_MAX_COST,
        _COST_TECHNIQUE,
        hws=_COST_HWS,
        use_torch=_COST_USE_TORCH,
        torch_device=_COST_TORCH_DEVICE,
        torch_interp=_COST_TORCH_INTERP,
        torch_batch=_COST_TORCH_BATCH,
        torch_cache=_COST_TORCH_CACHE,
    )

    nb_offsets, curr_disp_avg = _COST_NB_STRATEGY(
        lens,
        _COST_LENSES,
        coarse,
        _COST_DISPARITIES,
        max_cost=_COST_MAX_COST,
        nb_args=_COST_NB_ARGS,
    )

    nb_lenses = _rel_to_abs(lcoord, _COST_LENSES, nb_offsets)
    if len(nb_lenses) > 0:
        fine_2, coarse_2, _, _ = calc_costs_per_lens(
            lens,
            nb_lenses,
            _COST_DISPARITIES,
            _COST_MAX_COST,
            _COST_TECHNIQUE,
            hws=_COST_HWS,
            use_torch=_COST_USE_TORCH,
            torch_device=_COST_TORCH_DEVICE,
            torch_interp=_COST_TORCH_INTERP,
            torch_batch=_COST_TORCH_BATCH,
            torch_cache=_COST_TORCH_CACHE,
        )

        if _COST_REFINE is True:
            fine = np.append(fine, fine_2, axis=0)
            coarse = np.append(coarse, coarse_2, axis=0)
        else:
            fine = fine_2
            coarse = coarse_2

    num_targets = len(fine)
    coarse_merged = rtxdisp.merge_costs_additive(coarse, _COST_MAX_COST)
    fine_merged = np.array(rtxdisp.merge_costs_additive(fine, _COST_MAX_COST))

    return lcoord, fine_merged, coarse, coarse_merged, lens_var, num_targets


def _regularized_single(
    lcoord,
    lens,
    fine_cost,
    disp,
    penalty1,
    penalty2,
    max_cost,
    conf_tec,
    conf_sigma,
    only_dp,
    compute_conf,
    min_thresh,
    eps,
):

    F = np.flipud(np.rot90(fine_cost.T))

    sgm_cost = rtxsgm.sgm(lens.img, F, lens.mask, penalty1, penalty2, only_dp, max_cost)

    fine_depth = np.argmin(sgm_cost, axis=2)
    fine_depth_interp, fine_depth_val = rtxdisp.cost_minima_interp(sgm_cost, disp)

    wta_depth = np.argmin(F, axis=2)
    wta_depth_interp, wta_depth_val = rtxdisp.cost_minima_interp(F, disp)

    fine_depth_val = wta_depth_val

    if not compute_conf:
        confidence = np.zeros_like(lens.img)
        return (
            lcoord,
            fine_depth,
            fine_depth_interp,
            fine_depth_val,
            wta_depth,
            wta_depth_interp,
            wta_depth_val,
            confidence,
        )

    minimum_costs = np.min(sgm_cost, axis=2)

    if conf_tec == 'oev':
        num_denom = 0
        dmax = np.max(sgm_cost)
        dmin = np.min(sgm_cost)
        denom_denom = max(sgm_cost - minimum_costs[:, :, None], 1)
        for n in range(0, sgm_cost.shape[2]):
            index_map = np.ones((sgm_cost.shape[0], sgm_cost.shape[1])) * n
            tmp_num = np.pow(max(min(index_map - fine_depth, (dmax - dmin) / 3), 0), 2)
            num_denom += tmp_num / denom_denom[:, :, n]
        confidence = 1 / num_denom
    elif conf_tec == 'rtvbf':
        confidence = np.sum(np.exp(-((sgm_cost - fine_depth_val[:, :, None]) ** 2) / conf_sigma), axis=2) - 1

        ind = confidence > eps
        confidence[confidence <= 0] = 0.0
        confidence[ind] = 1.0 / confidence[ind]
    else:
        # numerically stable MLM: factor out minimum cost to avoid overflow
        scaled = -(sgm_cost - minimum_costs[:, :, None]) / (2 * np.power(conf_sigma, 2))
        denom_cost = np.sum(np.exp(scaled), axis=2)
        denom_cost[denom_cost == 0] = np.inf
        confidence = 1.0 / denom_cost
        confidence[np.isnan(confidence)] = 0

    return (
        lcoord,
        fine_depth,
        fine_depth_interp,
        fine_depth_val,
        wta_depth,
        wta_depth_interp,
        wta_depth_val,
        confidence,
    )


def _sgm_worker(lcoord):

    lens = _SGM_LENSES[lcoord]
    fine_cost = _SGM_FINE_COSTS[lcoord]
    return _regularized_single(
        lcoord,
        lens,
        fine_cost,
        _SGM_DISP,
        _SGM_PENALTY1,
        _SGM_PENALTY2,
        _SGM_MAX_COST,
        _SGM_CONF_TEC,
        _SGM_CONF_SIGMA,
        _SGM_ONLY_DP,
        _SGM_COMPUTE_CONF,
        _SGM_MIN_THRESH,
        _SGM_EPS,
    )


def _get_mp_context():

    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def _calc_chunksize(num_items, num_workers):

    if num_items == 0:
        return 1
    return max(1, num_items // (num_workers * 4))

def regularized_fine(
    lenses,
    fine_costs,
    disp,
    penalty1,
    penalty2,
    max_cost,
    conf_tec='mlm',
    conf_sigma=0.3,
    min_thresh=2.0,
    eps=0.0000001,
    only_dp=False,
    compute_conf=True,
    num_workers=0,
):

    fine_depths = dict()
    fine_depths_interp = dict()
    fine_depths_val = dict()
    wta_depths = dict()
    wta_depths_interp = dict()
    wta_depths_val = dict()
    num_lenses = len(lenses)
    confidence = dict()
    
    if num_workers is None or num_workers <= 1:
        for i, l in enumerate(fine_costs):
            if i % 100 == 0:
                print("Regularization: Processing lens {:05d}/{:05d}".format(i, num_lenses), end="\r", flush=True)

            lens = lenses[l]
            result = _regularized_single(
                l,
                lens,
                fine_costs[l],
                disp,
                penalty1,
                penalty2,
                max_cost,
                conf_tec,
                conf_sigma,
                only_dp,
                compute_conf,
                min_thresh,
                eps,
            )
            (
                lcoord,
                fine_depth,
                fine_depth_interp,
                fine_depth_val,
                wta_depth,
                wta_depth_interp,
                wta_depth_val,
                conf,
            ) = result

            fine_depths[lcoord] = fine_depth
            fine_depths_interp[lcoord] = fine_depth_interp
            fine_depths_val[lcoord] = fine_depth_val
            wta_depths[lcoord] = wta_depth
            wta_depths_interp[lcoord] = wta_depth_interp
            wta_depths_val[lcoord] = wta_depth_val
            confidence[lcoord] = conf
    else:
        keys = list(fine_costs.keys())
        num_workers = min(int(num_workers), len(keys))
        if num_workers <= 1:
            return regularized_fine(
                lenses,
                fine_costs,
                disp,
                penalty1,
                penalty2,
                max_cost,
                conf_tec=conf_tec,
                conf_sigma=conf_sigma,
                min_thresh=min_thresh,
                eps=eps,
                only_dp=only_dp,
                compute_conf=compute_conf,
                num_workers=0,
            )

        ctx = _get_mp_context()
        chunksize = _calc_chunksize(len(keys), num_workers)
        with ctx.Pool(
            processes=num_workers,
            initializer=_sgm_worker_init,
            initargs=(
                lenses,
                fine_costs,
                disp,
                penalty1,
                penalty2,
                max_cost,
                conf_tec,
                conf_sigma,
                only_dp,
                compute_conf,
                min_thresh,
                eps,
            ),
        ) as pool:
            for idx, result in enumerate(pool.imap_unordered(_sgm_worker, keys, chunksize=chunksize), 1):
                (
                    lcoord,
                    fine_depth,
                    fine_depth_interp,
                    fine_depth_val,
                    wta_depth,
                    wta_depth_interp,
                    wta_depth_val,
                    conf,
                ) = result

                fine_depths[lcoord] = fine_depth
                fine_depths_interp[lcoord] = fine_depth_interp
                fine_depths_val[lcoord] = fine_depth_val
                wta_depths[lcoord] = wta_depth
                wta_depths_interp[lcoord] = wta_depth_interp
                wta_depths_val[lcoord] = wta_depth_val
                confidence[lcoord] = conf

                if idx % 100 == 0 or idx == num_lenses:
                    print(
                        "Regularization: Processing lens {:05d}/{:05d}".format(idx, num_lenses),
                        end="\r",
                        flush=True,
                    )

    print("\nDone!")
    return fine_depths, fine_depths_interp, fine_depths_val, wta_depths, wta_depths_interp, wta_depths_val, confidence
    
def calc_costs_selective_with_lut(
    lenses,
    disparities,
    nb_strategy,
    technique,
    nb_args,
    max_cost,
    refine=True,
    hws=1,
    progress_hook=print,
    use_torch=False,
    torch_device="auto",
    torch_interp="bilinear",
    torch_batch=1,
    torch_cache=False,
    num_workers=0,
):
    
    """
    it firstly calculates the fine and coarse depth map based on the first "circle" (HEX_OFFSETS[1]) with lenses of same focal lens
    then it adds the other lenses (based on strategy, but the first one is always the same) and either 
    - refine the values or
    - substitute the values
    
    Then it merges the costs and returns fine and coarse
    """
    coarse_costs = dict()
    coarse_costs_merged = dict()
    fine_costs = dict()
    lens_std = dict()
    num_lenses = len(lenses)
    num_targets = 0
    
    # 4+2
    # using four lenses from the first circle (the 4 corners of a virtual rectangle around the lens)
    # + 2 lenses that are the closest one to the center lens
    pos1 = [[-1,-1],[-1,2],[1,1],[1,-2],[0,-1],[0,1]]
    pos2 = [[-1,-1],[-2,1],[1,1],[2,-1],[0,-1],[0,1]]
    pos3 = [[-2,1],[-1,2],[2,-1],[1,-2],[0,-1],[0,1]]

    if num_workers is None or num_workers <= 1:
        for i, lcoord in enumerate(lenses):
            lens = lenses[lcoord]

            # some lenses have troubles, mainly the 0-type lenses, when they are far away
            # using this solution seems better
            if rtxhexgrid.hex_focal_type(lcoord) == 0:
                pos = rtxhexgrid.HEX_OFFSETS[1]
            elif rtxhexgrid.hex_focal_type(lcoord) == 1:
                pos = pos1
            elif rtxhexgrid.hex_focal_type(lcoord) == 2:
                pos = pos1
            else:
                pdb.set_trace()

            nb_lenses = _rel_to_abs(lcoord, lenses, pos)

            if i % 100 == 0:
                print("Building Cost Volume: processing microlens {:05d}/{:05d}".format(i, num_lenses), end="\r", flush=True)

            # calculate a first guess of the disparity based on the first circle
            fine, coarse, coarse_merged, lens_var = calc_costs_per_lens(
                lens,
                nb_lenses,
                disparities,
                max_cost,
                technique,
                hws=hws,
                use_torch=use_torch,
                torch_device=torch_device,
                torch_interp=torch_interp,
                torch_batch=torch_batch,
                torch_cache=torch_cache,
            )
            nb_offsets, curr_disp_avg = nb_strategy(lens, lenses, coarse, disparities, max_cost=max_cost, nb_args=nb_args)

            nb_lenses = _rel_to_abs(lcoord, lenses, nb_offsets)

            if len(nb_lenses) > 0:
                fine_2, coarse_2, _, _ = calc_costs_per_lens(
                    lens,
                    nb_lenses,
                    disparities,
                    max_cost,
                    technique,
                    hws=hws,
                    use_torch=use_torch,
                    torch_device=torch_device,
                    torch_interp=torch_interp,
                    torch_batch=torch_batch,
                    torch_cache=torch_cache,
                )

                if refine is True:
                    fine = np.append(fine, fine_2, axis=0)
                    coarse = np.append(coarse, coarse_2, axis=0)
                else:
                    fine = fine_2
                    coarse = coarse_2

            num_targets += len(fine)
            coarse_costs_merged[lcoord] = rtxdisp.merge_costs_additive(coarse, max_cost)
            coarse_costs[lcoord] = coarse
            fine_costs[lcoord] = np.array(rtxdisp.merge_costs_additive(fine, max_cost))

            lens_std[lcoord] = lens_var
    else:
        ctx = _get_mp_context()
        if ctx.get_start_method() != "fork":
            progress_hook("Parallel cost volume requires fork; falling back to single process.")
            return calc_costs_selective_with_lut(
                lenses,
                disparities,
                nb_strategy,
                technique,
                nb_args,
                max_cost,
                refine=refine,
                hws=hws,
                progress_hook=progress_hook,
                use_torch=use_torch,
                torch_device=torch_device,
                torch_interp=torch_interp,
                torch_batch=torch_batch,
                torch_cache=torch_cache,
                num_workers=0,
            )

        if use_torch:
            progress_hook("Torch backend disabled for parallel cost volume. Use --cost_workers 0 to keep GPU sweep.")
            use_torch = False

        keys = list(lenses.keys())
        num_workers = min(int(num_workers), len(keys))
        if num_workers <= 1:
            return calc_costs_selective_with_lut(
                lenses,
                disparities,
                nb_strategy,
                technique,
                nb_args,
                max_cost,
                refine=refine,
                hws=hws,
                progress_hook=progress_hook,
                use_torch=use_torch,
                torch_device=torch_device,
                torch_interp=torch_interp,
                torch_batch=torch_batch,
                torch_cache=torch_cache,
                num_workers=0,
            )

        chunksize = _calc_chunksize(len(keys), num_workers)
        with ctx.Pool(
            processes=num_workers,
            initializer=_cost_selective_worker_init,
            initargs=(
                lenses,
                disparities,
                nb_strategy,
                nb_args,
                max_cost,
                technique,
                hws,
                refine,
                use_torch,
                torch_device,
                torch_interp,
                torch_batch,
                torch_cache,
            ),
        ) as pool:
            for idx, result in enumerate(pool.imap_unordered(_cost_selective_worker, keys, chunksize=chunksize), 1):
                lcoord, fine_merged, coarse, coarse_merged, lens_var, targets = result
                fine_costs[lcoord] = fine_merged
                coarse_costs[lcoord] = coarse
                coarse_costs_merged[lcoord] = coarse_merged
                lens_std[lcoord] = lens_var
                num_targets += targets

                if idx % 100 == 0 or idx == num_lenses:
                    print(
                        "Building Cost Volume: processing microlens {:05d}/{:05d}".format(idx, num_lenses),
                        end="\r",
                        flush=True,
                    )

    print("\nDone!\nNum comparisons: {0}\n".format(num_targets))
    
    return fine_costs, coarse_costs, coarse_costs_merged, lens_std, num_targets, 0.0   
   
def calc_costs_per_lens(
    lens,
    nb_lenses,
    disparities,
    max_cost,
    technique,
    hws=1,
    use_torch=False,
    torch_device="auto",
    torch_interp="bilinear",
    torch_batch=1,
    torch_cache=False,
):

    if use_torch and technique in ("sad", "ssd"):
        if torch_batch is not None and int(torch_batch) > 1:
            cost, img, d = rtxdisp.lens_sweep_torch_batched(
                lens,
                nb_lenses,
                disparities,
                technique,
                hws=hws,
                max_cost=max_cost,
                device=torch_device,
                interp=torch_interp,
                batch_neighbors=int(torch_batch),
                cache_images=torch_cache,
            )
        else:
            cost, img, d = rtxdisp.lens_sweep_torch(
                lens,
                nb_lenses,
                disparities,
                technique,
                hws=hws,
                max_cost=max_cost,
                device=torch_device,
                interp=torch_interp,
                cache_images=torch_cache,
            )
    else:
        cost, img, d = rtxdisp.lens_sweep(lens, nb_lenses, disparities, technique, hws=hws, max_cost=max_cost)
    coarse_costs = rtxdisp.sweep_to_shift_costs(cost, max_cost)
    coarse_costs_merged = rtxdisp.merge_costs_additive(coarse_costs, max_cost)
    lens_std = np.std(lens.img[lens.mask > 0])
  
    return cost, coarse_costs, coarse_costs_merged, lens_std
    
class EvalParameters(object):

    def __init__(self):

        self.max_disp_fac = 0.3
        self.min_disp_fac = 0.02 
        self.max_ring = 7
        self.max_cost = 10.0
        self.penalty1 = 0.1
        self.penalty2 = 0.4
        self.method = 'plain'
        self.use_rings = '0,1'
        self.refine = True
        self.coc_thresh = 1.2#1.5
        self.conf_sigma = 0.2
        self.max_conf = 2.0
        self.filename = None
        self.match_hws = 1
        self.coarse = False
        self.coarse_weight = 0.01
        self.struct_var = 0.01
        self.coarse_penalty1 = 0.01
        self.coarse_penalty2 = 0.03
        self.technique = 'sad'
        self.lut_trade_off = 1
        self.num_disp = 12
        self.method = 'real_lut'
        self.differentNames = False
        self.configfilename = ''
        self.maskpath = ''
        self.disppath = ''
        self.colorimagepath = ''
        self.use_torch = False
        self.torch_device = 'auto'
        self.torch_interp = 'bilinear'
        self.torch_batch = 1
        self.torch_cache = False
        self.sgm_only_dp = False
        self.compute_conf = True
        self.timing = False
        self.sgm_workers = 0
        self.cost_workers = 0
