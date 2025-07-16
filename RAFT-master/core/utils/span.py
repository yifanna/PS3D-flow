import torch
from torch import nn as nn

class SPAB(nn.Module):
    def __init__(self,e_lambda=1e-4,):
        super(SPAB, self).__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        weight = self.act(y)

        out3 = x * weight

        sim_att = torch.sigmoid(out3) - 0.5

        out2 = out3 + x
        x_minus_mu_square2 = (out2 - out2.mean(dim=[2,3],keepdim = True)).pow(2)
        y2 = x_minus_mu_square2 / (4 * (x_minus_mu_square2.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        weight2 = self.act(y2)

        out = out2 * weight2
        out = out * sim_att

        return out

if __name__ == '__main__':
    input = torch.randn(1, 64, 128, 128).cuda()
    model = SPAB(64).cuda()
    output = model(input)
    print(output.shape)