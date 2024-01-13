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


# class BartEncodecEncoder(BartEncoder):
#     def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
#         super().__init__(config)

#         self.dropout = config.dropout
#         self.layerdrop = config.encoder_layerdrop

#         embed_dim = config.d_model
#         self.padding_idx = config.pad_token_id
#         self.max_source_positions = config.max_position_embeddings
#         self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

#         self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

#         if embed_tokens is not None:
#             self.embed_tokens.weight = embed_tokens.weight

#         self.embed_positions = BartLearnedPositionalEmbedding(
#             config.max_position_embeddings,
#             embed_dim,
#         )
#         self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
#         self.layernorm_embedding = nn.LayerNorm(embed_dim)
#         learning_weight_init = torch.arange(8, 0, step=-1).float().view(8, 1, 1, 1)
#         self.learning_weight = nn.Parameter(learning_weight_init)
#         self.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutput]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input = input_ids
#             input_ids = input_ids.view(-1, 8, input_ids.shape[-1])
#         elif inputs_embeds is not None:
#             input = inputs_embeds[:, :, -1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         ENCODEC_RANGE = 8
#         if inputs_embeds is None:
#             inputs_embeds = []
#             for i in range(ENCODEC_RANGE):
#                 input_scale = self.embed_tokens(input_ids[:, i, :]) * self.embed_scale
#                 inputs_embeds.append(input_scale)
#             weighted_inputs_embeds = torch.mul(torch.stack(inputs_embeds, dim=0), F.softmax(self.learning_weight, dim=0))
#             inputs_embeds = torch.sum(weighted_inputs_embeds, dim=0)
#             # inputs_embeds = torch.sum(torch.stack(inputs_embeds, dim=0), dim=0)
#             embed_pos = self.embed_positions(inputs_embeds)
#             embed_pos = embed_pos.to(input.device)
#             inputs_embeds = inputs_embeds + embed_pos
#         hidden_states = inputs_embeds
#         hidden_states = self.layernorm_embedding(hidden_states)
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         # expand attention_mask
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         # check if head_mask has a correct number of layers specified if desired
#         if head_mask is not None:
#             if head_mask.size()[0] != (len(self.layers)):
#                 raise ValueError(
#                     f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
#                     f" {head_mask.size()[0]}."
#                 )

#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             dropout_probability = random.uniform(0, 1)
#             if self.training and (dropout_probability < self.layerdrop):  # skip the layer
#                 layer_outputs = (None, None)
#             else:
#                 if self.gradient_checkpointing and self.training:

#                     def create_custom_forward(module):
#                         def custom_forward(*inputs):
#                             return module(*inputs, output_attentions)

#                         return custom_forward

#                     layer_outputs = torch.utils.checkpoint.checkpoint(
#                         create_custom_forward(encoder_layer),
#                         hidden_states,
#                         attention_mask,
#                         (head_mask[idx] if head_mask is not None else None),
#                     )
#                 else:
#                     layer_outputs = encoder_layer(
#                         hidden_states,
#                         attention_mask,
#                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                         output_attentions=output_attentions,
#                     )

#                 hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )
    


class CustomBartModel(modeling_bart.BartPreTrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        # remove tied embedding
        self.encoder = modeling_bart.BartEncoder(config)
        self.decoder = modeling_bart.BartDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = modeling_bart.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

