import torch
import torch.nn as nn


def quantize_model(model, device):
    ## Dynamique quantization is not supported on CUDA, hence static conversion
    if "cuda" in device:
        # set quantization config for server (x86)
        model.qconfig = torch.quantization.get_default_config("fbgemm")

        # insert observers
        torch.quantization.prepare(model, inplace=True)
        # Calibrate the model and collect statistics

        # convert to quantized version
        torch.quantization.convert(model, inplace=True)

    else:
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    return model

def vit_first_layer_with_nchan(model, in_chans=1):

    kernel_size = model.patch_embed.proj.kernel_size
    stride = model.patch_embed.proj.stride
    embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # copy the original patch_embed.proj config 
    # except the number of input channels
    new_conv = torch.nn.Conv2d(
            in_chans, 
            out_channels=embed_dim,
            kernel_size=kernel_size, 
            stride=stride
            )
    # copy weigths and biases
    weight = model.patch_embed.proj.weight.clone()
    bias = model.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model
