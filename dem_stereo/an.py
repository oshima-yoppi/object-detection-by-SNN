import torch
from module.const import *
from module.network import *

seed = 46
torch.manual_seed(seed)
print(torch.seed())

net1 = RoughConv3(
    beta=BETA,
    spike_grad=SPIKE_GRAD,
    device=DEVICE,
    input_height=SPLITED_INPUT_HEIGHT,
    input_width=SPLITED_INPUT_WIDTH,
    rough_pixel=3,
    beta_learn=BETA_LEARN,
    threshold_learn=THRESHOLD_LEARN,
    input_channel=INPUT_CHANNEL,
    power=True,
    reset=RESET,
    repeat_input=REPEAT_INPUT,
    time_aware_loss=TIME_AWARE_LOSS,
)
net2 = RoughConv3(
    beta=BETA,
    spike_grad=SPIKE_GRAD,
    device=DEVICE,
    input_height=SPLITED_INPUT_HEIGHT,
    input_width=SPLITED_INPUT_WIDTH,
    rough_pixel=3,
    beta_learn=BETA_LEARN,
    threshold_learn=THRESHOLD_LEARN,
    input_channel=INPUT_CHANNEL,
    power=True,
    reset=RESET,
    repeat_input=REPEAT_INPUT,
    time_aware_loss=TIME_AWARE_LOSS,
)


def are_weights_equal(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1.data, param2.data):
            return False
    return True


print(are_weights_equal(net1, net2))
