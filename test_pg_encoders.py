import os
from functools import reduce
from typing import Union
import time
import sys
import logging

from timm.layers import create_conv2d
from pangaea.encoders.base import Encoder
from pangaea.encoders.croma_encoder import CROMA_OPTICAL_Encoder
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from utils.misc import get_model_size 



# Function to recursively iterate over all layers
def iterate_layers(module):
    for name, layer in module.named_children():
        print(f"Layer Name: {name}, Layer Type: {type(layer)}")
        iterate_layers(layer)  # Recursively iterate over nested layers

def get_first_module(model: torch.nn.Module):
    """
    modified from https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    """
    # get children form model
    children = list(model.named_children())
    if not children:
        # if model has no children; model is last child
        return model
    else:
       # look for children from children, to the last child
       for name, child in children:
            try:
                child_name, child = get_first_module(child)
                name = f'{name}.{child_name}'
            except TypeError:
                pass
            return name, child

def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def modify_first_conv2d(model, in_chans=1):
    # Find the first Conv2d layer
    first_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv_layer = (name, module)
            break

    if first_conv_layer is None:
        raise ValueError("No Conv2d layer found in the model.")

    layer_name, conv_layer = first_conv_layer
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    embed_dim = conv_layer.out_channels
    og_in_channels = conv_layer.in_channels

    # Create a new Conv2d layer with the desired number of input channels
    new_conv = torch.nn.Conv2d(
        in_chans,
        out_channels=embed_dim,
        kernel_size=kernel_size,
        stride=stride
    )

    # Copy weights and biases from the original Conv2d layer
    weight = conv_layer.weight.clone()
    bias = conv_layer.bias.clone() if conv_layer.bias is not None else None

    with torch.no_grad():
        for i in range(in_chans):
            j = i % og_in_channels  # cycle every n bands from the original weights
            new_conv.weight[:, i, :, :] = weight[:, j, :, :]  # band i takes old band j (blue) weights
        if bias is not None:
            new_conv.bias[:] = bias[:]

    # Replace the original Conv2d layer with the new one
    parent_module = model
    layer_names = layer_name.split('.')
    for name in layer_names[:-1]:
        parent_module = getattr(parent_module, name)
    setattr(parent_module, layer_names[-1], new_conv)

    return model


# cfg_path = './pangaea/configs/encoder/spectralgpt.yaml'
# cfg_path = './pangaea/configs/encoder/dofa.yaml'
# cfg_path = './pangaea/configs/encoder/croma_optical.yaml'
cfg_dir = './pangaea/configs/encoder/'

cfgs = [os.path.join(cfg_dir,f) for f in os.listdir(cfg_dir)]
# cfgs = ['./pangaea/configs/encoder/ssl4eo_dino.yaml']

no_pb = []

for cfg_path in cfgs:

    try :
        cfg = OmegaConf.load(cfg_path)
        encoder: Encoder = instantiate(cfg)
        logger = logging.getLogger()
        encoder.load_encoder_weights(logger)
        # print(encoder)
        modalities = list(encoder.input_bands.keys())
    except Exception as e:
        # print(e)
        pass

    if 'optical' in modalities:

        try:
            print(cfg_path)
            # print(get_model_size(encoder))
            time.sleep(1)
            input = {}
            input['optical'] = torch.rand(1,len(encoder.input_bands['optical']),encoder.input_size, encoder.input_size)
            # print(input'optical'].shape)
            # iterate_layers(encoder)
            # get_children(encoder)
            # print(get_first_module(encoder))
            # name, module = get_first_module(encoder)
            # print(name, module)
            # first = get_module_by_name(encoder, name)
            # print(first.in_channels)
            # model = modify_first_conv2d(encoder, in_chans=1)
            # print(model)
            # for name, param in encoder.named_parameters():
            #     print(f"Parameter Name: {name}, Parameter Shape: {param.shape}, Parameter type: {type(param)}")
            #     break

            feat = encoder(input)
            print(type(feat))
            print(len(feat))
            print(feat[0].shape)
            print(feat[1].shape)
            print(feat[2].shape)
            print(feat[3].shape)
            print("\n"*2)
            no_pb.append(cfg_path)
        except Exception as e:
            # print(e)
            pass

print(no_pb)
sys.exit(1)
feat = torch.Tensor(feat)
# print(len(feat))
# print(feat)
print(feat.shape)
