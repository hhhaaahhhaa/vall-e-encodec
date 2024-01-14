import math
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
import transformers.models.bart.modeling_bart as modeling_bart


class CustomModel(modeling_bart.BartPreTrainedModel):
    # Wrapper designed for loading pretrain parameters
    # Forward should not be directly called.
    def __init__(self, config: BartConfig):
        super().__init__(config)

        # remove tied embedding
        self.encoder = modeling_bart.BartEncoder(config)
        self.decoder = modeling_bart.BartDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
