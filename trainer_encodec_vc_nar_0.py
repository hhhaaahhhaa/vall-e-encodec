from argparse import ArgumentParser, Namespace

import torch
import wandb
from datasets import load_dataset
from encodec_model.nar_bart_model import NARBartForConditionalGeneration
from jiwer import wer
from transformers import (AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


wandb.init(project="encodec_vc", 
           name="speech-chatpgpt-base-nar-0",
)


TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output/speech-chatpgpt-base-nar-0",
    num_train_epochs=2,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    warmup_ratio=0.08,
    weight_decay=1e-2,
    logging_dir="./logs/speech-chatpgpt-base-nar-0",
    logging_steps=500,
    save_steps=10000,
    save_total_limit=5,
    evaluation_strategy="steps",
    eval_steps=10000,
    predict_with_generate=False,
    fp16=True,
    learning_rate=1e-5,
    push_to_hub=True,
    hub_model_id="lca0503/speech-chatpgpt-base-nar-0",
    report_to="wandb",
)


def pad_sequences_and_create_masks(sequences, max_length, padding_value):
    padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
    attention_masks = [[1 if token != padding_value else 0 for token in sequence] for sequence in padded_sequences]
    return padded_sequences, attention_masks


def process_data_to_model_inputs(batch, tokenizer):
    input_ids = []
    decoder_input_ids = []
    labels = []

    max_length = 1023  
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    for b in range(len(batch["instruction"])):
        instruction_ids = tokenizer(batch["instruction"][b])["input_ids"][1 : -1]
        transcription_ids = tokenizer(batch["transcription"][b])["input_ids"][1 : -1]

        curr_src_encodec_ids = tokenizer.convert_tokens_to_ids(
            [f"v_tok_{u}" for u in batch["src_encodec_0"][b]])
        encoder_input_ids = [bos_token_id] + \
            instruction_ids + [sep_token_id] + \
            transcription_ids + [sep_token_id] + \
            curr_src_encodec_ids + [eos_token_id]

        # Filter inputs
        if len(encoder_input_ids) > max_length:
            continue
        
        for i in range(1, 8):
            prev_tgt_encodec_ids = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (i - 1) * 1024}" for u in batch[f"tgt_encodec_{i - 1}"][b]])
            curr_tgt_encodec_ids = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + i * 1024}" for u in batch[f"tgt_encodec_{i}"][b]])

            # Filter inputs
            if len(prev_tgt_encodec_ids) > max_length:
                break
            
            input_ids.append(encoder_input_ids)
            decoder_input_ids.append(prev_tgt_encodec_ids)
            labels.append(curr_tgt_encodec_ids)

    # Pad decoder_input_ids and labels
    input_ids, attention_mask = pad_sequences_and_create_masks(input_ids, max_length=max_length,
                                                               padding_value=pad_token_id)
    decoder_input_ids, decoder_attention_mask = pad_sequences_and_create_masks(decoder_input_ids, max_length=max_length,
                                                                               padding_value=pad_token_id)
    labels, _ = pad_sequences_and_create_masks(labels, max_length=max_length,
                                               padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels
    }


def get_dataset(tokenizer, args):
    train_dataset = load_dataset(args.dataset, "train", split="+".join(args.train_splits))
    eval_dataset = load_dataset(args.dataset, "eval", split="+".join(args.eval_splits))

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_train_batch_size,
        fn_kwargs={"tokenizer": tokenizer}
    )
    eval_dataset = eval_dataset.map(
        process_data_to_model_inputs,
        remove_columns=eval_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_eval_batch_size,
        fn_kwargs={"tokenizer": tokenizer}
    )

    return train_dataset, eval_dataset


def preprocess_logits_for_metrics(ret, labels):
    logits, encoder_last_hidden_state = ret
    predictions = torch.argmax(logits, axis=-1)

    return predictions


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    predictions = [prediction[:len(label)] for prediction, label in zip(predictions, labels)]
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer_value = wer([" ".join(filter(None, i.split("v_tok_"))) for i in decoded_labels],
                    [" ".join(filter(None, i.split("v_tok_"))) for i in decoded_preds])
    
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:", labels[i])
        print("pred:", predictions[i])
        print("-----------------")
    print("=================================")
    
    return {"wer": wer_value}


def main(args):
    model = NARBartForConditionalGeneration.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    
    train_dataset, eval_dataset = get_dataset(tokenizer, args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="lca0503/GPTspeech_encodec")
    parser.add_argument("-t", "--train_splits", type=str, nargs="+",
                        default=["train"])
    parser.add_argument("-e", "--eval_splits", type=str, nargs="+",
                        default=["validation"])
    parser.add_argument("-m", "--model_name", type=str, default="/work/b08902123/SpeechChatGPT/previous_ckpt/tts_nar_fixed/checkpoint-105000/")

    args = parser.parse_args()    
    return args
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
