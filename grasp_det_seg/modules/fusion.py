import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, min_level, levels):
        super(FusionModule, self).__init__()
        
        self.min_level = min_level
        self.levels = levels

        # SFModule components
        self.conv1x1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)
        self.conv_d1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_d3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.conv_d5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
    
    def forward(self, x, sp):
        x = x[self.min_level:self.min_level + self.levels]
        #tensors = [torch.stack(i._tensors).detach() for i in sp]
        tensors = [torch.stack(i._tensors) for i in sp]

        fp = []
        for tensor in tensors:
            f_one_hot = F.one_hot(tensor.long(), num_classes=32)
            f_final = f_one_hot.permute(0, 3, 1, 2).float()
            fp.append(f_final)

        fx = [torch.cat((a, b), dim=1) for a, b in zip(x, fp)]

        # SFModule forward pass
        x = self.conv1x1(fx[0])
        x_d1 = self.conv_d1(x)
        x_d3 = self.conv_d3(x)
        x_d5 = self.conv_d5(x)
        f1 = x_d1 + x_d3 + x_d5

        x = self.conv1x1(fx[1])
        x_d1 = self.conv_d1(x)
        x_d3 = self.conv_d3(x)
        x_d5 = self.conv_d5(x)
        f2 = x_d1 + x_d3 + x_d5

        x = self.conv1x1(fx[2])
        x_d1 = self.conv_d1(x)
        x_d3 = self.conv_d3(x)
        x_d5 = self.conv_d5(x)
        f3 = x_d1 + x_d3 + x_d5

        x = self.conv1x1(fx[3])
        x_d1 = self.conv_d1(x)
        x_d3 = self.conv_d3(x)
        x_d5 = self.conv_d5(x)
        f4 = x_d1 + x_d3 + x_d5

        out = [f1, f2, f3, f4]
        return out