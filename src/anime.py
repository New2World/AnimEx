import os
import argparse

import skimage.io
import skimage.metrics

import fixer.fix_image as f_img

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inp_path', required=True)
    parser.add_argument('-o', dest='outp_path', default=None)
    parser.add_argument('-g', dest='gpu', action='store_true', default=False)
    parser.add_argument('-s', dest='block_size', type=int, default=512)
    parser.add_argument('-m', dest='measure', action='store_true', default=False)
    parser.add_argument('-t', dest='target', default=None)
    return parser.parse_args()

def main():
    argv = parse_arg()
    outp_path = argv.outp_path
    if outp_path is None:
        outp_path = os.path.dirname(argv.inp_path)
        outp_path = os.path.join(outp_path, 'output.jpg')

    solver = f_img.ImageFixer(size=argv.block_size)
    outp = solver.fix(argv.inp_path, outp_path, gpu=argv.gpu)
    if argv.measure:
        if argv.target is None:
            raise ValueError('Ground truth needed')
        gt = skimage.io.imread(argv.target)
        psnr = skimage.metrics.peak_signal_noise_ratio(gt, outp)
        print(f'PSNR: {psnr}')

if __name__ == '__main__':
    main()