import torch.nn as nn
import torch

class Loss_Function(nn.Module):
    def __init__(self, normal_weight=1.0, aolp_weight=0.05):
        super(Loss_Function, self).__init__()
        self.cosine_loss = nn.CosineSimilarity()
        self.normal_weight = normal_weight
        self.aolp_weight = aolp_weight

    def forward(self, predict, ground_truth, aolp, confidence_map, mask, train_loader):
        # Normal Loss
        predict = predict * mask
        # predict = normalize(predict, dim=1)
        ground_truth = ground_truth * mask
        ground_truth_n = (ground_truth * 2.0 -1.0) * mask
        cosine = 1 - self.cosine_loss(predict, ground_truth_n)
        num_cosine = torch.sum(torch.sum(torch.sum(cosine, dim=1), dim=1))
        M = torch.sum(torch.sum(torch.sum(mask, dim=1), dim=1))  # Foreground object pixels
        back_ground = (train_loader.batch_size * 256 * 256) - M  # Background region pixels
        loss_cosine = num_cosine - back_ground
        L_normal = loss_cosine / M

        # # Physics-based Prior Confidence (From TransSfP)
        # L_aolp =loss_aolp(predict,aolp,mask,confidence_map)

        # Total Loss
        total_loss = self.normal_weight * L_normal #+ self.aolp_weight * L_aolp
        return total_loss

def loss_aolp(input_vec,aolp,mask_tensor,confidence_map):
    confidence_map = confidence_map.squeeze(1)
    aolp = aolp.squeeze(1) * torch.pi
    aolp_0 = aolp + torch.pi / 2
    aolp_1 = aolp - torch.pi / 2
    aolp_0 = torch.remainder(aolp_0,torch.pi * 2)
    aolp_1 = torch.remainder(aolp_1,torch.pi * 2)

    mask_invalid_pixels = torch.all(mask_tensor < 255, dim=1)
    y = input_vec[:,1,:,:]
    x = input_vec[:,0,:,:]
    phi = torch.atan2(y,x) # (batchSize,H,W) (-pi,pi)
    phi = torch.remainder(phi,torch.pi * 2)
    # aolp[mask_invalid_pixels] = 0.0
    error_0 = torch.min(torch.abs(phi - aolp_0),
                              torch.pi * 2 - torch.abs(phi - aolp_0)) # (bs,H,W)
    error_1 = torch.min(torch.abs(phi - aolp_1),
                              torch.pi * 2 - torch.abs(phi - aolp_1))
    error = torch.min(error_0,error_1)
    error = error * (confidence_map)
    error[mask_invalid_pixels] = 0.0
    loss = torch.sum(error)
    total_valid_pixels = (~mask_invalid_pixels).sum()
    # loss = loss / total_valid_pixels
    return loss