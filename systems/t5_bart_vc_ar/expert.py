"""
Customized Bart, autoregressive decoder, flexible max length, no tied embeddings.
"""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput, CausalLMOutputWithCrossAttentions
import transformers.models.bart.modeling_bart as modeling_bart
import transformers.models.t5.modeling_t5 as modeling_t5
import copy

from .model import CustomModel


class System(modeling_bart.BartPreTrainedModel):
    """
    Implemented as a decoder-only model but able to be conditioned.
    """

    def __init__(self, config):
        # replace with custom bart and do not tie any embedding.
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)

        self.model = CustomModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, config.vocab_size)))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # tie decoder embedding only
        self.lm_head.weight = self.model.decoder.embed_tokens.weight

        # freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    # Modified methods
    def init_from_hf(self):
        self.load_bart_decoder()
    
    def load_bart_decoder(self):
        # fit weights from bart-base into custom architecture brutally
        model = modeling_bart.BartForConditionalGeneration.from_pretrained("voidful/bart-base-unit")
        state_dict = model.model.decoder.state_dict()
        model_state_dict = self.model.decoder.state_dict()
        state_dict_pop_keys = []
        for k in state_dict:
            if k in model_state_dict:
                # Untie
                if k in ["embed_tokens.weight"]:
                    state_dict[k] = copy.deepcopy(model.state_dict()["model.shared.weight"])
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
            else:
                print(f"Dropping parameter {k}")
                state_dict_pop_keys.append(k)

        # modify state_dict format into model_state_dict format
        for k in state_dict_pop_keys:
            state_dict.pop(k)
        for k in model_state_dict:
            if k not in state_dict:
                print("Reinitialize: ", k)
                state_dict[k] = model_state_dict[k]
        
        # become required shape
        self.model.decoder.load_state_dict(state_dict)

    def forward(
        self,
        # Condition (from encoder)
        e_input_ids: torch.LongTensor = None,
        e_attention_mask: Optional[torch.Tensor] = None,
        e_head_mask: Optional[torch.Tensor] = None,
        e_inputs_embeds: Optional[torch.FloatTensor] = None,
        # Decoder input
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encoder forward
        # print(encoder_outputs is None)
        # print("forward: ", len(input_ids[0]))
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=e_input_ids,
                attention_mask=e_attention_mask,
                head_mask=e_head_mask,
                inputs_embeds=e_inputs_embeds,
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
        
        # Decoder forward
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=e_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])
        logits = logits + self.final_logits_bias.to(logits.device)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,
            "e_input_ids": kwargs["e_input_ids"],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # Copied methods from BartCasualLM
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
