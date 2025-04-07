import sys
import logging
from pangaea.encoders.base import Encoder
from pangaea.encoders.croma_encoder import CROMA_OPTICAL_Encoder
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# cfg_path = './pangaea/configs/encoder/spectralgpt.yaml'
# cfg_path = './pangaea/configs/encoder/dofa.yaml'
cfg_path = './pangaea/configs/encoder/croma_optical.yaml'
cfg = OmegaConf.load(cfg_path)
print(cfg)
# sys.exit(1)
encoder: Encoder = instantiate(cfg)
logger = logging.getLogger()
encoder.load_encoder_weights(logger)
print(encoder)
