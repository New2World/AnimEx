import cv2
import math
import numpy as np

from fixer import fixer

class ImageFixer(fixer._Fixer):
    def __init__(self):
        super(ImageFixer, self).__init__()

    def __cut_image(self, image, size):
        h_max, w_max, _ = image.shape
        h_size, w_size = size
        nh, nw = math.ceil(h_max/h_size), math.ceil(w_max/w_size)
        batch = [image[h*h_size:min((h+1)*h_size,h_max),w*w_size:min((w+1)*w_size,w_max)] for h in range(nh) for w in range(nw)]
        return batch, nh, nw

    def __merge_image(self, batch, nh, nw):
        if nh == nw == 1:
            return batch[0]
        image = np.vstack([np.hstack(batch[h*nw:(h+1)*nw]) for h in range(nh)])
        return image

    def _fix(self, image, size, gpu):
        batch, nh, nw = self.__cut_image(image, size)
        batch_fix = []
        for img in batch:
            out_img = self._run(img[np.newaxis], gpu).squeeze()
            batch_fix.append(out_img)
        outp = self.__merge_image(batch_fix, nh, nw)
        return outp
    
    def fix(self, inp_path, outp_path, size, gpu):
        image = cv2.imread(inp_path)
        image = image/255.
        if size <= 0:
            size = image.shape[:2]
        else:
            size = (size,size)
        outp = self._fix(image, size, gpu)
        if outp_path.endswith('.jpg'):
            cv2.imwrite(outp_path, outp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            cv2.imwrite(outp_path, outp)
        return outp

