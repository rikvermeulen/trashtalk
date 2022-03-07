import nemo.collections.asr as nemo_asr
from torch.utils.data import DataLoader
from audio_data_layer import AudioDataLayer
from omegaconf import OmegaConf

import config

model = config.asr_model
cfg = config.cfg

## Make config overwrite-able
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0

# spectrogram normalization constants
cfg.preprocessor.normalize = config.normalization

## Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)

model.preprocessor = model.from_config_dict(cfg.preprocessor)

# Set model to inference mode
model.eval();
model= config.asr_model.to(model.device)

data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)


#test
config.mbn_model.eval();
mbn_data_layer = AudioDataLayer(sample_rate=config.mbn_cfg.train_ds.sample_rate)
mbn_data_loader = DataLoader(mbn_data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
