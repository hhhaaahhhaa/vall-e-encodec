import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments


class CodecCollate(object):
    """
    encoder input: instruction + src_codec(0)
    decoder input: tgt_codec(0)
    decoder output: tgt_codec(0)
    """
    def __init__(self, data_config: dict, model_config, train_config: Seq2SeqTrainingArguments):
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config

        self.codec_name = data_config["codec_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(model_config["encoder_tokenizer_name"], model_max_length=1024)

    def collate_fn(self, batch):
        # print(len(batch))
        # print(batch[0])
        # input()
        all_encoder_input_ids = [b["e_input_ids"] for b in batch]
        all_decoder_input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        # decoder only model need to apply left-padding!
        all_encoder_input_ids, attention_mask = pad_sequences_and_create_masks(all_encoder_input_ids, max_length=None,
                                                                padding_value=self.tokenizer.pad_token_id)
        all_decoder_input_ids, _ = pad_sequences_and_create_masks(all_decoder_input_ids, max_length=None,
                                                            padding_value=self.tokenizer.pad_token_id, is_left=True)
        labels, _ = pad_sequences_and_create_masks(labels, max_length=None,
                                                padding_value=-100, is_left=True)
        
        return {
            "e_input_ids": torch.LongTensor(all_encoder_input_ids),
            "e_attention_mask": torch.LongTensor(attention_mask),
            "input_ids": torch.LongTensor(all_decoder_input_ids),
            "labels": torch.LongTensor(labels),
        }

    def map_fn(self, batch):
        tokenizer = self.tokenizer

        all_encoder_input_ids = []
        all_decoder_input_ids = []
        labels = []  # all decoder output ids

        max_length = self.model_config["max_length"]
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        # sep_token_id = tokenizer.sep_token_id
        sep_token_id = 35  # Use ":", currently sep token id == eos token id
        pad_token_id = tokenizer.pad_token_id

        for b in range(len(batch["instruction"])):
            instruction_ids = self.encoder_tokenizer(batch["instruction"][b])["input_ids"][1 : -1]

            # Encoder input
            encoder_input_ids = [bos_token_id] + instruction_ids + [eos_token_id]
            
            # Decoder input
            src_ids = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u}" for u in batch[f"src_{self.codec_name}_0"][b]])
            tgt_ids = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u}" for u in batch[f"tgt_{self.codec_name}_0"][b]])
            decoder_input_ids = [bos_token_id] + src_ids + [sep_token_id] + tgt_ids
            decoder_output_ids = src_ids + [sep_token_id] + tgt_ids + [eos_token_id]

            # Filter inputs
            if len(encoder_input_ids) > max_length or len(decoder_input_ids) > max_length:
                continue

            all_encoder_input_ids.append(encoder_input_ids)
            all_decoder_input_ids.append(decoder_input_ids)
            labels.append(decoder_output_ids)

        return {
            "e_input_ids": all_encoder_input_ids,
            "input_ids": all_decoder_input_ids,
            "labels": labels
        }


def pad_sequences_and_create_masks(sequences, max_length, padding_value, is_left=False):
    if max_length is None:
        max_length = max([len(sequence) for sequence in sequences])
    if is_left:
        padded_sequences = [[padding_value] * (max_length - len(sequence)) + sequence for sequence in sequences]
    else:
        padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
    attention_masks = [[1 if token != padding_value else 0 for token in sequence] for sequence in padded_sequences]
    return padded_sequences, attention_masks
