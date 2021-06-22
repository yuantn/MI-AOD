from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import os
import argparse
import numpy

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on single images')
    parser.add_argument('config_file', help='train config file path')
    parser.add_argument('ckpt_file', help='model checkpoint file path')
    parser.add_argument('img_file', help='image file path')
    parser.add_argument('out_file', help='output image file path')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    model = init_detector(args.config_file, args.ckpt_file, device='cuda:0')
    result, uncertainty = inference_detector(model, args.img_file)
    # uncertainty = calculate_uncertainty_single(cfg, model, args.img_file, return_box=False)
    model.show_result(args.img_file, result, out_file=args.out_file)
    print('Image uncertainty is: ' + str(uncertainty.cpu().numpy()))

if __name__ == '__main__':
    main()

