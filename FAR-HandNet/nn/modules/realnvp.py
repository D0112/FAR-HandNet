import torch
import torch.nn as nn
__all__ = 'Sigma_linear','RealNVP'


class Sigma_linear(nn.Module):
    def __init__(self, kpts, bias=True, device='cpu'):
        super(Sigma_linear, self).__init__()
        in_channel = kpts*2
        out_channel = kpts*2
        self.bias = bias
        self.linear = nn.Linear(in_channel, out_channel, bias,device=device)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
    # def forward(self, x):
    #     device = x.device
    #     y = x.float().matmul(self.linear.weight.t().float())
    #
    #     if self.norm:
    #         x_norm = torch.norm(x, dim=1, keepdim=True)
    #         y = y / x_norm
    #
    #     if self.bias:
    #         y = y + self.linear.to(device).bias
    #     return y
    def forward(self, x):
        device = x.device
        y = self.linear(x)
        return y



class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = nett
        # self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.s = nets

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        device = x.device
        dtype = x.dtype
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t.block))):
            z_ = self.mask[i].to(device,dtype) * z
            s = self.s[i](z_) * (1 - self.mask[i].to(device,dtype))
            t = self.t[i](z_) * (1 - self.mask[i].to(device,dtype))
            z = (1 - self.mask[i].to(device,dtype)) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z).sigmoid() + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)
