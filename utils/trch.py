import torch
import torch.nn as nn


def quantize_model(model, device):
    ## Dynamique quantization is not supported on CUDA, hence static conversion
    if "cuda" in device:
        # set quantization config for server (x86)
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

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


def vit_first_layer_with_nchan(model, in_chans=1):

    kernel_size = model.patch_embed.proj.kernel_size
    stride = model.patch_embed.proj.stride
    embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    og_in_channels = model.patch_embed.proj.in_channels
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
            j = i%og_in_channels # cycle every n bands from the original weights
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model


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
