#%%
from .unet import UNet
from .transunet import TransUnet
import torch.nn as nn
import torch
import numpy as np

def get_model(args):
    if not hasattr(args, "channel"):
        channel = 11
    else:
       channel =  args.channel

    model = TransUnet(channel, 3, dim=64, residual_num=8, norm='in', dropout=0.0, skip_res=True)
    return model
#%%
if __name__ == "__main__":
    class Dict2Class(object):
        def __init__(self, my_dict):
            
            for key in my_dict:
                setattr(self, key, my_dict[key])
    
