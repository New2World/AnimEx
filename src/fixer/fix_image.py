import cv2
import math
import numpy as np

from fixer import fixer

class ImageFixer(fixer._Fixer):
    def __init__(self):
        super(ImageFixer, self).__init__()

    def __cut_image(self, image, size, cut_edge):
        h_max, w_max, _ = image.shape
        h_size, w_size = size
        if cut_edge:
            h_size_k, w_size_k = h_size-20, w_size-20
            nh, nw = math.ceil(h_max/(h_size_k)), math.ceil(w_max/(w_size_k))
            batch = [image[max(0,h*h_size_k-10):min((h+1)*h_size_k+10,h_max),max(0,w*w_size_k-10):min((w+1)*w_size_k+10,w_max)] for h in range(nh) for w in range(nw)]
        else:
            nh, nw = math.ceil(h_max/h_size), math.ceil(w_max/w_size)
            batch = [image[h*h_size:min((h+1)*h_size,h_max),w*w_size:min((w+1)*w_size,w_max)] for h in range(nh) for w in range(nw)]
        return batch, nh, nw

    def __cut_edge(self, batch, nh, nw):
        blen = len(batch)
        block_pos = lambda x: (x//nw,x%nw)
        for i in range(blen):
            xy = block_pos(i)
            if not xy[0] == 0:
                batch[i] = batch[i][10:,:]
            if not xy[0] == nh-1:
                batch[i] = batch[i][:-10,:]
            if not xy[1] == 0:
                batch[i] = batch[i][:,10:]
            if not xy[1] == nw-1:
                batch[i] = batch[i][:,:-10]
        return batch

    def __merge_image(self, batch, nh, nw, cut_edge):
        if nh == nw == 1:
            return batch[0]
        if cut_edge:
            self.__cut_edge(batch, nh, nw)
        image = np.vstack([np.hstack(batch[h*nw:(h+1)*nw]) for h in range(nh)])
        return image

    def _fix(self, image, size, cut_edge, gpu):
        batch, nh, nw = self.__cut_image(image, size, cut_edge)
        batch_fix = []
        for img in batch:
            out_img = self._run(img[np.newaxis], gpu).squeeze()
            batch_fix.append(out_img)
        outp = self.__merge_image(batch_fix, nh, nw, cut_edge)
        return outp
    
    def fix(self, inp_path, outp_path, size, gpu):
        image = cv2.imread(inp_path)
        image = image/255.
        if size < 0:
            size = image.shape[:2]
            cut_edge = False
        elif size <= 20:
            raise ValueError("block size must be larger than 20 in order to get a better quality")
        else:
            size = (size,size)
            cut_edge = True
        outp = self._fix(image, size, cut_edge, gpu)
        if outp_path.endswith('.jpg') or outp_path.endswith('.jpeg'):
            cv2.imwrite(outp_path, outp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            cv2.imwrite(outp_path, outp)
        return outp