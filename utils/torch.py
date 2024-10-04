import torch
import torch.nn as nn


def quantize_model(model, device):

    ## Dynamique quantization is not supported on CUDA, hence static conversion
    if 'cuda' in device:
        # set quantization config for server (x86)
        model.qconfig = torch.quantization.get_default_config('fbgemm')

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

