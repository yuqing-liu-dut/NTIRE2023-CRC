import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import cv2, numpy as np, sys


def make_model(args, parent=False):
    return XSWL(args)

class XSWL(nn.Module):
    def __init__(self, args):
        super(XSWL, self).__init__()
        self.conv  = nn.Sequential(
            nn.Conv2d(3, 3*args.scale[0]*args.scale[0]//2, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(3*args.scale[0]*args.scale[0]//2, 3*args.scale[0]*args.scale[0], 3, padding=3//2),
        )      
        self.scale = args.scale[0]
        
        self.ps = nn.PixelShuffle(args.scale[0])

    def forward(self, x):
        y = self.conv(x)
        y = self.ps(y)
        x = y + F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return x 


if __name__ == "__main__":
    net = XSWL().cuda()
    net.load_state_dict(torch.load("model.pth"))

    I = cv2.imread(sys.argv[1]).transpose(2, 0, 1)[np.newaxis, ...]
    I = torch.FloatTensor(I).cuda()

    I = net(I).detach().cpu().numpy()[0].transpose(1, 2, 0)
    I[I<0]=0
    I[I>255]=255
    
    cv2.imwrite(sys.argv[2], I)
    
