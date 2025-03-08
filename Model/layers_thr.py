import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from spikingjelly.clock_driven import functional, layer, surrogate
import math

### 阈值改进

steps = 6
dt = 5
simwin = dt * steps
a = 0.25
aa = 0.5    # 梯度近似项


class PLIFNode_org(BaseNode):
    def __init__(self, tau=2.0, v_threshold=0.0, v_reset=0.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(tau - 1.0)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        # self.v = self.v_threshold / 2  # init
        if self.v_reset is None:
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()

            if len(dv.size()) == 5:
                self.v_threshold = dv.mean([0, 2, 3, 4])[None, :, None, None, None]
            elif len(dv.size()) == 3:
                self.v_threshold = dv.mean([0, 2])[None, :, None]
            else:
                print('dv shape error!')

        return self.neuronal_fire()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'



class SeqSNN_Spatial(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input: torch.Tensor):
        return self.seq_snn_forward(input, self)

    def seq_snn_forward(self, input: torch.Tensor, stateless_module: nn.Module or list or tuple or nn.Sequential):
        y_shape = [input.shape[0], input.shape[1]]
        y = input.flatten(0, 1)
        if isinstance(stateless_module, (list, tuple, nn.Sequential)):
            for m in stateless_module:
                y = m(y)
        else:
            y = stateless_module(y)
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)

