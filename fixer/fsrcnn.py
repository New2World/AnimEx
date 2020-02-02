import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1):
        super(FSRCNN, self).__init__()
        self.first_part = self.__first(in_channels)
        self.mid_part = self.__mid()
        self.last_part = self.__last(out_channels, scale)

    def __first(self, in_channels):
        first = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        for m in first.modules():
            if type(m) is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        return first

    def __mid(self):
        mid = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.PReLU()
        )
        for m in mid.modules():
            if type(m) is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        return mid

    def __last(self, out_channels, scale):
        last = nn.ConvTranspose2d(128, out_channels, kernel_size=5, padding=2, stride=scale, output_padding=scale-1)
        nn.init.kaiming_normal_(last.weight)
        return last

    def forward(self, x):
        x1 = self.first_part(x)
        x2 = self.mid_part(x1)
        y = self.last_part(torch.cat((x2,x1), dim=1))
        return y