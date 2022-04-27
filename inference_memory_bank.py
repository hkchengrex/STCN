import math
import torch

def make_gaussian(y_idx, x_idx, height, width, sigma):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

def softmax_w_top(x, top=None, kernel=None):
    if kernel is not None:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        x_exp = torch.exp(x - maxes)*kernel
        x_exp, indices = torch.topk(x_exp, k=top, dim=1)
    else:
        values, indices = torch.topk(x, k=top, dim=1)
        x_exp = torch.exp(values - values[:,0])

    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    x_exp /= x_exp_sum
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x

class MemoryBank:
    def __init__(self, k, top_k=20, kmn_sigma=None):
        self.top_k = top_k
        self.kmn_sigma = kmn_sigma

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k

    def _global_matching(self, mk, qk, h, w):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a+b-c) / math.sqrt(CK)  # B, NE, HW
        
        if self.kmn_sigma is not None:
            # Make a bunch of Gaussian distributions
            argmax_idx = affinity.max(2)[1]
            y_idx, x_idx = argmax_idx//w, argmax_idx%w
            g = make_gaussian(y_idx, x_idx, h, w, sigma=self.kmn_sigma)
            g = g.view(B, NE, h*w)

            affinity = softmax_w_top(affinity, top=self.top_k, kernel=g)  # B, NE, HW
        else:
            affinity = softmax_w_top(affinity, top=self.top_k, kernel=None)  # B, NE, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk, h, w)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)