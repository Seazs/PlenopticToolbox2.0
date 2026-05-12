"""
Batch disparity estimation for a dataset of plenoptic images.
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt

import disparity.disparity_methods as rtxmain


def _resolve_xml(dataset_dir, xml_path):
    if xml_path:
        return xml_path
    candidates = sorted(glob.glob(os.path.join(dataset_dir, "*.xml")))
    if len(candidates) == 0:
        raise OSError("No .xml found in dataset folder")
    if len(candidates) > 1:
        raise OSError("Multiple .xml files found; pass --xml to select one")
    return candidates[0]


def _build_params(args, img_path, xml_path):
    params = rtxmain.EvalParameters()
    params.filename = img_path
    params.differentNames = True
    params.configfilename = xml_path
    params.coarse = args.coarse
    params.technique = args.technique
    params.method = "real_lut"
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
    params.scene_type = args.scene_type
    params.analyze_err = args.err
    params.confidence_technique = args.confidence_technique
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
    return params


def _disp_names(output_dir, pic_name, params, disparities, disparity_format):
    disp_name = "{0}/{1}_disp_{2}_{3}_{4}_{5}.{6}".format(
        output_dir,
        pic_name,
        params.method,
        disparities[0],
        disparities[-1],
        params.technique,
        disparity_format,
    )
    disp_name_col = "{0}/{1}_disp_col_{2}_{3}_{4}_{5}.{6}".format(
        output_dir,
        pic_name,
        params.method,
        disparities[0],
        disparities[-1],
        params.technique,
        disparity_format,
    )
    return disp_name, disp_name_col


def main():
    parser = argparse.ArgumentParser(description="Batch disparity estimation")
    parser.add_argument("dataset_name", help="Dataset folder name under data/")
    parser.add_argument("--data-root", dest="data_root", default="./data")
    parser.add_argument("--output-root", dest="output_root", default="./results")
    parser.add_argument("--xml", dest="xml_path", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save-color", action="store_true", default=False)
    parser.add_argument("--coarse", default=False, action="store_true")
    parser.add_argument("-t", dest="technique", default="sad")
    parser.add_argument("-dmin", dest="min_disp", default="1")
    parser.add_argument("-dmax", dest="max_disp", default="9")
    parser.add_argument("-nd", dest="num_disp", default="16")
    parser.add_argument("--hws", dest="match_hws", type=int, default=1)
    parser.add_argument("--p1", dest="penalty1", type=float, default=0.1)
    parser.add_argument("--p2", dest="penalty2", type=float, default=0.4)
    parser.add_argument("--max_cost", dest="max_cost", type=float, default=10.0)
    parser.add_argument("--use_rings", dest="use_rings", default="0,1")
    parser.add_argument("--lut_trade_off", dest="lut_trade_off", type=float, default=1.0)
    parser.add_argument("--no_refine", dest="refine", default=True, action="store_false")
    parser.add_argument("--coarse_p1", dest="coarse_penalty1", type=float, default=0.01)
    parser.add_argument("--coarse_p2", dest="coarse_penalty2", type=float, default=0.03)
    parser.add_argument("--coarse_weight", dest="coarse_weight", type=float, default=0.01)
    parser.add_argument("--struct_var", dest="struct_var", type=float, default=0.01)
    parser.add_argument("--conf_sigma", dest="conf_sigma", type=float, default=0.2)
    parser.add_argument("--torch", dest="use_torch", default=False, action="store_true")
    parser.add_argument("--device", dest="torch_device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--torch_interp", dest="torch_interp", default="bilinear", help="bilinear|bicubic|nearest")
    parser.add_argument("--torch_batch", dest="torch_batch", type=int, default=1)
    parser.add_argument("--torch_cache", dest="torch_cache", default=False, action="store_true")
    parser.add_argument("--sgm_dp", dest="sgm_only_dp", default=False, action="store_true")
    parser.add_argument("--no_conf", dest="compute_conf", default=True, action="store_false")
    parser.add_argument("--timing", dest="timing", default=False, action="store_true")
    parser.add_argument("--sgm_workers", dest="sgm_workers", type=int, default=0)
    parser.add_argument("--cost_workers", dest="cost_workers", type=int, default=0)
    parser.add_argument("-scene", dest="scene_type", default="real")
    parser.add_argument("--err", default=False, action="store_true")
    parser.add_argument("-tconf", dest="confidence_technique", default="mlm")
    parser.add_argument("-format", dest="disparity_format", default="png")

    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset_name)
    if not os.path.isdir(dataset_dir):
        raise OSError("Dataset folder not found: {0}".format(dataset_dir))

    xml_path = _resolve_xml(dataset_dir, args.xml_path)
    images = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    if len(images) == 0:
        raise OSError("No .png images found in dataset folder")

    output_dir = os.path.join(args.output_root, args.dataset_name, "disparities")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Dataset: {0}".format(args.dataset_name))
    print("Images: {0}".format(len(images)))
    print("XML: {0}".format(xml_path))
    print("Output: {0}".format(output_dir))

    for img_path in images:
        pic_name = os.path.splitext(os.path.basename(img_path))[0]
        params = _build_params(args, img_path, xml_path)
        _, disp, _, _, _, _, disparities, _, _, _, _, _ = rtxmain.estimate_disp(params)

        disp_name, disp_name_col = _disp_names(
            output_dir, pic_name, params, disparities, args.disparity_format
        )
        if os.path.exists(disp_name) and not args.overwrite:
            print("Skipping existing: {0}".format(disp_name))
            continue

        print("Saving disparity: {0}".format(disp_name))
        plt.imsave(disp_name, disp, vmin=disparities[0], vmax=disparities[-1], cmap="gray")
        if args.save_color:
            plt.imsave(
                disp_name_col,
                disp,
                vmin=disparities[0],
                vmax=disparities[-1],
                cmap="jet",
            )

    print("Done")


if __name__ == "__main__":
    main()
