import math
import numpy as np
import skimage
import skimage.io as skio

from fixer import fixer

class ImageFixer(fixer._Fixer):
    def __init__(self, size):
        super(ImageFixer, self).__init__()
        self.__size = size

    def __cut_image(self, image):
        h_max, w_max, _ = image.shape
        nh, nw = math.ceil(h_max/self.__size), math.ceil(w_max/self.__size)
        batch = np.asarray([image[h*self.__size:min((h+1)*self.__size,h_max),w*self.__size:min((w+1)*self.__size,w_max)] for h in range(nh) for w in range(nw)])
        return batch, nh, nw

    def __merge_image(self, batch, nh, nw):
        if nh == nw == 1:
            return batch[0]
        image = np.vstack([np.hstack(batch[h*nw:(h+1)*nw]) for h in range(nh)])
        return image

    def _fix(self, image, gpu):
        batch, nh, nw = self.__cut_image(image)
        batch_fix = []
        for img in batch:
            out_img = self._run(img[np.newaxis], gpu).squeeze()
            batch_fix.append(out_img)
        outp = self.__merge_image(batch_fix, nh, nw)
        return outp
    
    def fix(self, inp_path, outp_path, gpu):
        image = skio.imread(inp_path)
        image = image/255.
        outp = self._fix(image, gpu)
        if outp_path.endswith('.jpg'):
            skio.imsave(outp_path, outp, quality=100)
        else:
            skio.imsave(outp_path, outp)
        return outp

