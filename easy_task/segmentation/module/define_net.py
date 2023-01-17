import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
from . import network


# import yaml
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
beta = 0.95
parm_learn = False
spike_grad = surrogate.atan()
net= network.ConvDense0(beta=beta, spike_grad=spike_grad, device=device, parm_lean=parm_learn)
