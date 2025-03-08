import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from Model.layers_thr import *
from Model.layers import *

if_lateral = True

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)

try:
    import cupy


    class S_TCNTauNet(nn.Module):
        def __init__(self, input_channels, channels: int, arg):
            super().__init__()
            # Spiking Encoder

            self.conv = nn.Sequential(
                SeqSNN_Spatial(
                    nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False, dilation=1),
                    nn.BatchNorm2d(channels)),
                PLIFNode_org(),
                SeqSNN_Spatial(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, dilation=2),
                    nn.BatchNorm2d(channels)),
                PLIFNode_org(),
                SeqSNN_Spatial(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, dilation=1),
                    nn.BatchNorm2d(channels)),
                PLIFNode_org(),
                SeqSNN_Spatial(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, dilation=2),
                    nn.BatchNorm2d(channels)),
                PLIFNode_org()
            )

            # Spiking Classifier
            self.fl2 = nn.Flatten(2)
            self.fc = nn.Sequential(
                layer.MultiStepDropout(0.5),
                SeqSNN_Spatial(nn.Linear(channels * arg.channels_FC1 * arg.channels_FC2, channels * 2 * 2, bias=False)),
                PLIFNode_org(),
                layer.MultiStepDropout(0.5),
                SeqSNN_Spatial(nn.Linear(channels * 2 * 2, arg.num_classes, bias=False)),
                PLIFNode_org()
            )
            self.vote = VotingLayer(1)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)

            out_spikes = self.conv(x)

            out_spikes = self.fl2(out_spikes)

            out_spikes = self.fc(out_spikes)

            return self.vote(out_spikes.mean(0))



except ImportError as e:
    print(e)
    print('-----Cupy is not installed.-----')