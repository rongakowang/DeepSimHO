import torch
import torch.nn as nn
from anakin.opt import cfg as config
from anakin.utils.logger import logger

class PhysicsMeshMLP(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_set_up = config["PHYSICS_MLP"]["LAYER_ENCODER"]
        decoder_set_up = config["PHYSICS_MLP"]["LAYER_DECODER"]
        self.input_dim = 778 * 3 + 1000 * 3 + 1778
        self.out_channel = config["PHYSICS_MLP"]["OUT_DIM"]
        self.use_batch_norm = config["PHYSICS_MLP"]["USE_BATCH_NORM"] # fix here
        self.use_dropout = config["PHYSICS_MLP"]["USE_DROPOUT"]

        layers_encoder = nn.ModuleList()
        layers_decoder = nn.ModuleList()
        self.layers_input = nn.Linear(self.input_dim, encoder_set_up[0])
        logger.info(f"creating MLP with {self.input_dim} input and {self.out_channel} output")
        for i in range(len(encoder_set_up) - 1):
            layers_encoder.append(nn.Linear(encoder_set_up[i], encoder_set_up[i+1]))
            if self.use_batch_norm:
                layers_encoder.append(nn.BatchNorm1d(encoder_set_up[i+1]))
            layers_encoder.append(nn.ReLU())
            if self.use_dropout:
                layers_encoder.append(nn.Dropout(p=0.2))

        layers_decoder.append(nn.Linear(encoder_set_up[-1], decoder_set_up[0]))
        if self.use_batch_norm:
            layers_decoder.append(nn.BatchNorm1d(decoder_set_up[0]))
        layers_decoder.append(nn.ReLU())
        if self.use_dropout:
            layers_encoder.append(nn.Dropout(p=0.2))
        
        for i in range(len(decoder_set_up) - 1):
            layers_decoder.append(nn.Linear(decoder_set_up[i], decoder_set_up[i+1]))
            if self.use_batch_norm:
                layers_decoder.append(nn.BatchNorm1d(decoder_set_up[i+1]))
            layers_decoder.append(nn.ReLU())
            if self.use_dropout:
                layers_encoder.append(nn.Dropout(p=0.2))
        layers_decoder.append(nn.Linear(decoder_set_up[-1], self.out_channel))

        self.en = nn.Sequential(*layers_encoder)
        self.de = nn.Sequential(*layers_decoder)

    def forward(self, x, cp):
        x = torch.cat([x, cp], dim=-1) # simple concatenation
        x0 = self.layers_input(x)
        x1 = self.en(x0)
        xr = x0 + x1 # residual connection, x2 should be (B, 1024)
        x2 = self.de(xr)
        return x2