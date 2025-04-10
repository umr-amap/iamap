import os
import time
import sys
import logging
from pangaea.encoders.base import Encoder
from pangaea.encoders.croma_encoder import CROMA_OPTICAL_Encoder
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from utils.misc import get_model_size 



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
        print(encoder)
        modalities = list(encoder.input_bands.keys())
    except Exception as e:
        print(e)
        pass

    if 'optical' in modalities:

        try:
            print(cfg_path)
            print(get_model_size(encoder))
            time.sleep(1)
            input = {}
            input['optical'] = torch.rand(1,len(encoder.input_bands['optical']),encoder.input_size, encoder.input_size)
            print(input['optical'].shape)

            feat = encoder(input)
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
