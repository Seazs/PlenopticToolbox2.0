"""
Sample code to read an image and estimate the disparity
Parameters can be input by hand or predefinite ones will be used
----
@veresion v1.1 - Januar 2017
@author Luca Palmieri
"""
import disparity.disparity_methods as rtxmain
import plenopticIO.imgIO as imgIO
import argparse
import os
import json
import pdb
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read an image and estimate disparity")
    # input filename (the .xml) - the .png must have same name!
    parser.add_argument(dest='input_filename', nargs='?', help="Name of the lens config file")
    # allow different names for image and config
    parser.add_argument('--img', dest='img_path', default=None, help="Path to the .png image")
    parser.add_argument('--xml', dest='xml_path', default=None, help="Path to the .xml config")
    # output path (should be a folder, without / in the end)
    parser.add_argument('-o', dest='output_path', default='./')
    # if you want to calculate also a coarse disparity map (1 value per lens)
    parser.add_argument('--coarse', default=False, action='store_true')
    # technique used to calculate the cost volume
    # - 'sad' = Sum of Absolute Difference
    # - 'ssd' = Sum of Squared Difference
    # - 'census' = Census Transform Difference
    # - 'ncc' = Normalized Cross Correlation
    parser.add_argument('-t', dest='technique', default='sad')
    # minimum disparity (should not be negative)
    parser.add_argument('-dmin', dest='min_disp', default='1')
    # maximum disparity (should not be higher than half of the diameter of a lens (35-39 pixels))
    parser.add_argument('-dmax', dest='max_disp', default='17')
    # number of disparities (nd)
    parser.add_argument('-nd', dest='num_disp', default='16')
    # matching window half-size (hws)
    parser.add_argument('--hws', dest='match_hws', type=int, default=1)
    # SGM penalties (smoothness)
    parser.add_argument('--p1', dest='penalty1', type=float, default=0.1)
    parser.add_argument('--p2', dest='penalty2', type=float, default=0.4)
    # cost volume settings
    parser.add_argument('--max_cost', dest='max_cost', type=float, default=10.0)
    parser.add_argument('--use_rings', dest='use_rings', default='0,1')
    parser.add_argument('--lut_trade_off', dest='lut_trade_off', type=float, default=1.0)
    parser.add_argument('--no_refine', dest='refine', default=True, action='store_false')
    # coarse-to-fine options
    parser.add_argument('--coarse_p1', dest='coarse_penalty1', type=float, default=0.01)
    parser.add_argument('--coarse_p2', dest='coarse_penalty2', type=float, default=0.03)
    parser.add_argument('--coarse_weight', dest='coarse_weight', type=float, default=0.01)
    parser.add_argument('--struct_var', dest='struct_var', type=float, default=0.01)
    # confidence settings
    parser.add_argument('--conf_sigma', dest='conf_sigma', type=float, default=0.2)
    # torch backend options
    parser.add_argument('--torch', dest='use_torch', default=False, action='store_true')
    parser.add_argument('--device', dest='torch_device', default='auto', help='auto|cpu|cuda|cuda:0')
    parser.add_argument('--torch_interp', dest='torch_interp', default='bilinear', help='bilinear|bicubic|nearest')
    parser.add_argument('--torch_batch', dest='torch_batch', type=int, default=1)
    parser.add_argument('--torch_cache', dest='torch_cache', default=False, action='store_true')
    # speed/diagnostic options
    parser.add_argument('--sgm_dp', dest='sgm_only_dp', default=False, action='store_true')
    parser.add_argument('--no_conf', dest='compute_conf', default=True, action='store_false')
    parser.add_argument('--timing', dest='timing', default=False, action='store_true')
    parser.add_argument('--sgm_workers', dest='sgm_workers', type=int, default=0)
    parser.add_argument('--cost_workers', dest='cost_workers', type=int, default=0)
    # scene is used to differentiate between real and synthetic
    # 'synth' is used for synthetic scene from Blender
    # 'real' for scenes acquired with cameras
    # it is usually helpful to understand if there is a ground truth
    parser.add_argument('-scene', dest='scene_type', default='real')
    # set to true to calculate error against ground truth (so you need to provide ground truth),
    # so only for synthetic scenes
    parser.add_argument('--err', default=False, action='store_true')
    # additional parameter, set to False if you don't want to save the confidence map
    parser.add_argument('-conf', dest='save_conf', default=True)
    ## THE CONFIDENCE
    # confidence can be calculated using three methods
    # - 'mlm' - mlm confidence proposed in "A Quantitative Evaluation of Confidence Measures for Stereo Vision"
    #           available at https://ieeexplore.ieee.org/document/6143951
    # - 'oev' - (NOT WORKING AT THE MOMENT) pixelwise confidence measure proposed in "A NOVEL CONFIDENCE MEASURE FOR DISPARITY MAPS BY PIXEL-WISE COST FUNCTION ANALYSIS"
    #           available at https://ieeexplore.ieee.org/document/8451500
    # - 'rtvbf' - cost based confidence measure proposed in "Real-Time Visibility-Based Fusion of Depth Maps"
    #           availble at https://ieeexplore.ieee.org/document/4408984
    parser.add_argument('-tconf', dest='confidence_technique', default='mlm')
    # whether to save or not the parameters file (default yes)
    parser.add_argument('-savepars', dest='save_parameters', default=True)
    # add a sparse depth map
    parser.add_argument('-sparse', dest='save_sparse', default=False)
    # format for depth map
    parser.add_argument('-format', dest='disparity_format', default='png')
    
    args = parser.parse_args()

    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path)
        #raise OSError('Path {0} does not exist'.format(args.output_path))
                          
    if args.img_path is not None or args.xml_path is not None:
        if args.img_path is None or args.xml_path is None:
            raise ValueError("Both --img and --xml must be provided together")
        input_filename = args.img_path
    elif args.input_filename is not None:
        input_filename = args.input_filename
    else:
        raise ValueError("Please provide either the .xml file or --img/--xml")

    params = rtxmain.EvalParameters()
    params.filename = input_filename
    params.coarse = args.coarse
    params.technique = args.technique
    params.method = 'real_lut'
    params.min_disp = args.min_disp
    params.max_disp = args.max_disp
    params.num_disp = args.num_disp
    params.match_hws = args.match_hws
    params.penalty1 = args.penalty1
    params.penalty2 = args.penalty2
    params.max_cost = args.max_cost
    params.use_rings = args.use_rings
    params.lut_trade_off = args.lut_trade_off
    params.refine = args.refine
    params.coarse_penalty1 = args.coarse_penalty1
    params.coarse_penalty2 = args.coarse_penalty2
    params.coarse_weight = args.coarse_weight
    params.struct_var = args.struct_var
    params.conf_sigma = args.conf_sigma
    params.use_torch = args.use_torch
    params.torch_device = args.torch_device
    params.torch_interp = args.torch_interp
    params.torch_batch = args.torch_batch
    params.torch_cache = args.torch_cache
    params.sgm_only_dp = args.sgm_only_dp
    params.compute_conf = args.compute_conf
    params.timing = args.timing
    params.sgm_workers = args.sgm_workers
    params.cost_workers = args.cost_workers
    params.scene_type = args.scene_type
    params.analyze_err = args.err
    params.confidence_technique = args.confidence_technique
    if args.img_path is not None:
        params.differentNames = True
        params.configfilename = args.xml_path

    pic_name = os.path.splitext(os.path.basename(params.filename))[0]
    
    I, disp, Dwta, Dgt, Dconf, Dcoarse, disparities, ncomp, disp_avg, new_offset, error_measurements, central_lens = rtxmain.estimate_disp(params)

    img_name = "{0}/{1}.png".format(args.output_path, pic_name) 
    disp_name = "{0}/{1}_disp_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    disp_name_col = "{0}/{1}_disp_col_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    gt_name = "{0}/{1}_gt_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    gt_name_col = "{0}/{1}_gt_col_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    conf_name = "{0}/{1}_conf_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    conf_name_norm = "{0}/{1}_conf_norm_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
    json_name = "{0}/{1}_parameters.json".format(args.output_path, pic_name)
    xml_name = "{0}/{1}_config.xml".format(args.output_path, pic_name)

    #plt.subplot(121)
    #plt.title("Input Image")
    #plt.imshow(I)
    #plt.subplot(122)
    #plt.title("Disparity Image")
    #plt.imshow(disp, vmin=disparities[0], vmax=disparities[-1], cmap='jet')
    #plt.show()
    
    print("\nFinished, now saving everything... ")

    if Dgt is not None:
        print("Saving ground truth... ")
        plt.imsave(gt_name, Dgt, vmin=np.min(Dgt), vmax=np.max(Dgt), cmap='gray')
        plt.imsave(gt_name_col, Dgt, vmin=np.min(Dgt), vmax=np.max(Dgt), cmap='jet')

    if error_measurements is not None:
        #save in a file the errors
        error_analysis = dict()
        error_analysis['avg_error'] = error_measurements[0]
        error_analysis['mask_error'] = error_measurements[1]
        error_analysis['mse_error'] = error_measurements[2]
        badPix1, badPix2, badPixGraph = error_measurements[3]
        error_analysis['badpix1_avg'] = np.mean(badPix1)
        error_analysis['badpix2_avg'] = np.mean(badPix2)
        plotting = np.mean(badPixGraph, axis=0)
        error_analysis['err_threshold'] = plotting.tolist()
        error_analysis['bumpiness'] = error_measurements[4]
        depth_disc, depth_smooth, badPix1Disc, badPix2Disc, badPix1Smooth, badPix2Smooth, badPixGraphDisc, badPixGraphSmooth = error_measurements[5]
        error_analysis['badpix1disc'] = np.mean(badPix1Disc)
        error_analysis['badpix1smooth'] = np.mean(badPix1Smooth)
        error_analysis['badpix2disc'] = np.mean(badPix2Disc)
        error_analysis['badpix2smooth'] = np.mean(badPix2Smooth)
        plottingdisc = np.mean(badPixGraphDisc, axis=0)
        error_analysis['err_thr_disc'] = plottingdisc.tolist()
        plottingsmth = np.mean(badPixGraphSmooth, axis=0)
        error_analysis['err_thr_smooth'] = plottingsmth.tolist()
        error_analysis['disc_err'] = depth_disc
        error_analysis['smooth_err'] = depth_smooth     
        err_ana_name = "{0}/{1}_error_analysis_{2}_{3}_{4}.json".format(args.output_path, pic_name, disparities[0], disparities[-1], params.technique) 
        err_ana_csv = "{0}/{1}_error_analysis_{2}_{3}_{4}.csv".format(args.output_path, pic_name, disparities[0], disparities[-1], params.technique) 
        err_arr_csv = "{0}/{1}_error_array_{2}_{3}_{4}.csv".format(args.output_path, pic_name, disparities[0], disparities[-1], params.technique)      
        imgIO.write_csv_file(error_analysis, err_ana_csv, params.technique)
        plotting_arrays = [plotting, plottingdisc, plottingsmth]
        imgIO.write_csv_array(plotting_arrays, err_arr_csv, params.technique)
    
    
    if args.save_parameters:
        print("Saving parameter file... ")
        parameters = dict()
        parameters['dmin'] = disparities[0]
        parameters['dmax'] = disparities[-1]
        parameters['disparities'] = disparities.tolist()
        parameters['technique'] = args.technique
        parameters['conf'] = args.confidence_technique
        parameters['filename'] = input_filename
        parameters['scene_type'] = args.scene_type
        parameters['method'] = params.method
        parameters['pic_name'] = pic_name
        parameters['image_path'] = img_name
        parameters['disp_path'] = disp_name
        parameters['disp_path_col'] = disp_name_col
        parameters['gt_path'] = gt_name
        parameters['gt_path_col'] = gt_name_col
        parameters['conf_path'] = conf_name
        parameters['config_path'] = xml_name
        parameters['output_path'] = args.output_path

        with open(json_name, 'w') as outfile:
            json.dump(parameters, outfile)

    print("Saving estimated disparity... ")
    plt.imsave(disp_name, disp, vmin=disparities[0], vmax=disparities[-1], cmap='gray')
    plt.imsave(disp_name_col, disp, vmin=disparities[0], vmax=disparities[-1], cmap='jet')
    if args.save_sparse:
        #pdb.set_trace()
        print("Saving sparse disparity..")
        sp_disp_name = "{0}/{1}_sparse_disp_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
        sp_disp_name_col = "{0}/{1}_sparse_disp_col_{2}_{3}_{4}_{5}.{6}".format(args.output_path, pic_name, params.method, disparities[0], disparities[-1], params.technique, args.disparity_format) 
        sparse_disp = disp * (Dconf > 0.7)
        plt.imsave(sp_disp_name, sparse_disp, vmin=disparities[0], vmax=disparities[-1], cmap='gray')
        plt.imsave(sp_disp_name_col, sparse_disp, vmin=disparities[0], vmax=disparities[-1], cmap='jet')

    print("Saving colored image... ")
    plt.imsave(img_name, I)

    # after the estimation, the image is rendered with the central lens in the center of the image, so
    # offset is [0,0], rotation is 0, border pixel should be taken from .xml file (it's usually one)
    lens_border = 1
    angle = 0
    print("Saving xml file... ")
    config_ = imgIO.save_only_xml(xml_name, I.shape, central_lens, lens_border, angle)
    if args.save_conf:
        print("Saving confidence map... ")
        plt.imsave(conf_name, Dconf, vmin=np.min(Dconf), vmax=np.max(Dconf), cmap='gray')
        plt.imsave(conf_name_norm, Dconf, vmin=np.min(Dconf), vmax=np.max(Dconf), cmap='jet')

    print("\n******************\n")
