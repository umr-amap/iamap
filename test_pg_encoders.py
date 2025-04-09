import sys
import logging
from pangaea.encoders.base import Encoder
from pangaea.encoders.croma_encoder import CROMA_OPTICAL_Encoder
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

# cfg_path = './pangaea/configs/encoder/spectralgpt.yaml'
# cfg_path = './pangaea/configs/encoder/dofa.yaml'
cfg_path = './pangaea/configs/encoder/croma_optical.yaml'
cfg = OmegaConf.load(cfg_path)
# print(cfg)
# sys.exit(1)
encoder: Encoder = instantiate(cfg)
logger = logging.getLogger()
encoder.load_encoder_weights(logger)
# print(encoder.input_size)
# print(encoder.input_bands)
# print(len(encoder.input_bands['optical']))
modalities = list(encoder.input_bands.keys())
# print(modalities)

input = {}
input['optical'] = torch.rand(1,len(encoder.input_bands['optical']),encoder.input_size, encoder.input_size)
print(input['optical'].shape)

feat = encoder(input)
print(feat[0].shape)
print(feat[1].shape)
print(feat[2].shape)
print(feat[3].shape)
print(type(feat[0]))
sys.exit(1)
feat = torch.Tensor(feat)
# print(len(feat))
# print(feat)
print(feat.shape)
