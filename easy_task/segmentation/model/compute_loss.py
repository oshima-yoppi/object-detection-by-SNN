import torch
import torch.nn as nn


def spike_mse_loss(input, target, rate=0.8):
    # input shape:[time, batch, neuron]?
    print(target.shape)
    batch = target.shape[0]
    target = target.reshape(batch, -1)
    criterion = nn.MSELoss()
    num_steps = input.shape[0]
    input_spike_count = torch.sum(input, dim=0)
    target_spike_count = num_steps*rate * target
    target_spike_count = torch.round(target_spike_count)
    loss = criterion(input_spike_count, target_spike_count)
    return loss