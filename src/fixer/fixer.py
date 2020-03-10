import torch
import numpy as np

from models import fsrcnn

class _Fixer:
    def __init__(self):
        self.model = fsrcnn.FSRCNN(3,3,scale=1)
        model_state_dict = torch.load('../model_param/FSRCNN1x.pt')
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
    
    def _run(self, batch, gpu):
        batch = batch.transpose((0,3,1,2))
        batch = torch.from_numpy(batch).type(torch.FloatTensor)
        if gpu:
            batch = batch.cuda()
            self.model.cuda()
        else:
            self.model.cpu()
        outp = self.model(batch)
        outp = torch.clamp(outp, 0., 1.)
        outp_img = outp.detach().cpu().numpy()
        outp_img = outp_img.transpose((0,2,3,1))
        outp_img = (outp_img * 255.).astype(np.uint8)
        return outp_img