
import torch
import torch.nn as nn
def compute_loss(input: torch.Tensor, label:torch.Tensor):
    """
    :param input: tensor of shape (batch, time , 1)
    :param target: tensor of shape (batch, 3)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
  
    input = input[:, int(len(input[-1]) *0.8 ):]
    input = torch.mean(input, dim = 1)

    label = torch.sqrt(label[:,0]**2 + label[:,1]**2 + label[:,2]**2)
    # print(f'labelだよ{label}')
    loss = torch.mean((input - label) ** 2)
    # print('aaaaaaaaaaaaaaaaaaaaaaa')
    # print(loss)
    return loss


def vector_loss(input: torch.Tensor, label:torch.Tensor):
    """
    :param input: tensor of shape (batch, time , 3)
    :param target: tensor of shape (batch, 3)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
    eps = 1e-10
    input = input[:,:, int(len(input[-1]) *0.8 ):]
    input = torch.mean(input, dim = 2)+eps
    # pred_omemga = torch.sqrt(torch.pow(input[:,0], 2) + torch.pow(input[:,1], 2) + torch.pow(input[:,2], 2))
    # label_omega = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2) + torch.pow(label[:,2], 2))
    # print('##################')
    # print(pred_omemga.size())
    # print(label_omega.size())
    # print(pred_omemga)
    # print("asf")
    # print(label_omega)
    # loss_func = nn.MSELoss()
    # loss = loss_func(pred_omemga, label_omega)
    # loss = torch.mean(torch.pow(torch.sub(pred_omemga, label_omega), 2))
    # loss = torch.mean(torch.pow(torch.sub(torch.sqrt(label_omega), torch.sqrt(pred_omemga)), 2))
    #loss = {(wx-wx)^2 + **** + (wz-wz)^2} 平均してる
    loss = torch.mean(torch.pow(torch.sub(input[:,0], label[:,0]), 2) + torch.pow(torch.sub(input[:,1], label[:,1]), 2) + torch.pow(torch.sub(input[:,2], label[:,2]), 2))  
    return loss


def analysis_loss(input, label):
    """
    compute vector loss for analysis
    """
    eps = 1e-10
    input = input[:,:, int(len(input[-1]) * 0.8):]
    input = torch.mean(input, dim=2) + eps
    loss_x = torch.abs(torch.sub(input[:,0], label[:,0]))
    loss_y = torch.abs(torch.sub(input[:,1], label[:,1]))
    loss_z = torch.abs(torch.sub(input[:,2], label[:,2]))
    # print(loss_x.shape)
    # print(loss_x)

    pred_omemga = torch.sqrt(torch.pow(input[:,0], 2) + torch.pow(input[:,1], 2) + torch.pow(input[:,2], 2))
    label_omega = torch.sqrt(torch.pow(label[:,0], 2) + torch.pow(label[:,1], 2) + torch.pow(label[:,2], 2))
    loss_omega = torch.abs(torch.sub(pred_omemga, label_omega))

    # same train_loss
    same_loss = torch.mean(torch.pow(torch.sub(input[:,0], label[:,0]), 2) + torch.pow(torch.sub(input[:,1], label[:,1]), 2) + torch.pow(torch.sub(input[:,2], label[:,2]), 2))      
    
    return loss_x, loss_y, loss_z, loss_omega, same_loss

if __name__ == "__main__":
    a = torch.zeros(3,5
    )
    print(len(a[0]))