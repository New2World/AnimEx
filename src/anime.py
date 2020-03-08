import os
import cv2
import argparse

import skimage.metrics

import fixer.fix_image as f_img

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inp_path', required=True)
    parser.add_argument('-o', dest='outp_path', default=None)
    parser.add_argument('-g', dest='gpu', action='store_true', default=False)
    parser.add_argument('-s', dest='block_size', type=int, default=-1)
    parser.add_argument('-m', dest='measure', default=None)
    return parser.parse_args()

def main():
    argv = parse_arg()
    outp_path = argv.outp_path
    if outp_path is None:
        outp_path = os.path.dirname(argv.inp_path)
        outp_path = os.path.join(outp_path, 'output.jpg')

    solver = f_img.ImageFixer()
    outp = solver.fix(argv.inp_path, outp_path, size=argv.block_size, gpu=argv.gpu)
    if argv.measure is not None:
        gt = cv2.imread(argv.measure)
        if not gt.shape == outp.shape:
            print('UserWarning: the ground truth has different dimensions, and to fit the scale of output the ground truth will be resized')
            if gt.shape[0] < outp.shape[0]:
                inter_method = cv2.INTER_CUBIC
            else:
                inter_method = cv2.INTER_LANCZOS4
            gt = cv2.resize(gt, (outp.shape[1],outp.shape[0]), interpolation=inter_method)
        psnr = skimage.metrics.peak_signal_noise_ratio(gt, outp)
        print(f'PSNR: {psnr}')

if __name__ == '__main__':
    main()