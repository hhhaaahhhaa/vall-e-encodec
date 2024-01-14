from argparse import ArgumentParser, Namespace

# import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, BatchEncoding
import os
import torch

import Define
from systems.bart_vc_ar.expert import System


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import Define


def pack_inputs(tokenizer, instruction_ids, src_ids):
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = 35
    pad_token_id = tokenizer.pad_token_id
    
    encoder_input_ids = [bos_token_id] + instruction_ids + [eos_token_id]
    decoder_input_ids = [bos_token_id] + src_ids + [sep_token_id]
    
    inputs = BatchEncoding(tensor_type="pt")
    inputs["e_input_ids"] = torch.tensor([encoder_input_ids])
    inputs["input_ids"] = torch.tensor([decoder_input_ids])

    return inputs


def ar_inference(ar_model, ar_tokenizer, dataset, bad_words_ids):
    instruction_ids = ar_tokenizer(dataset["instruction"][0])["input_ids"][1 : -1]
    src_ids = ar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u}" for u in dataset[f"src_speech_tokenizer_0"][0]])
    # print(instruction_ids, len(instruction_ids))
    # print(src_ids, len(src_ids))
    inputs = pack_inputs(ar_tokenizer, instruction_ids, src_ids)
    inputs = inputs.to("cuda")
    decode_ar = ar_model.generate(**inputs, max_length=4096, num_beams=1,
                                  do_sample=True, use_cache=True, bad_words_ids=bad_words_ids)
    
    # remove <bos>, prefix, <sep>, and <eos> from output
    print(decode_ar[0])
    n_prefix = len(src_ids) + 2
    print(decode_ar[0, n_prefix:-1])


if __name__ == "__main__":
    data_config = {
        "name": "kuanhuggingface/google_tts_speech_tokenizer",
        "splits": ["validation"]
    }
    dataset = load_dataset(
        data_config["name"],
        split="+".join(data_config["splits"]),
        cache_dir=Define.HF_CACHE_DIR
    )
    
    dataset = dataset.filter(lambda x : len(x[f"src_speech_tokenizer_0"]) <= 700)
    dataset = dataset.shuffle(seed=666).select(range(1))
    
    checkpoint = "training_output/speech-chatgpt-base-ar-debug/checkpoint-1600/"
    ar_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ar_model = System.from_pretrained(checkpoint)
    ar_model.to("cuda")

    ar_tokens = [f"v_tok_{i}" for i in range(1024)]
    ar_tokens.extend(["</s>"])  # since prefix is given, <eos> becomes the only special token allowed.
    allowed_word_ids = [ar_tokenizer.convert_tokens_to_ids(x) for x in ar_tokens]
    ar_bad_words_ids = [[i] for i in range(ar_tokenizer.vocab_size) if i not in allowed_word_ids]
    ar_inference(ar_model, ar_tokenizer, dataset, ar_bad_words_ids)
