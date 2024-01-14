from argparse import ArgumentParser, Namespace

import wandb
from jiwer import wer
from transformers import (AutoTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, BartConfig, GenerationConfig)
import os

import Define
from systems.bart_vc_ar.datamodule import DataModule
from systems.bart_vc_ar.expert import System


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# TODO: Change training arguments
wandb.init(project="encodec_vc", 
        name="speech-chatgpt-base-ar",
)

TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output/speech-chatgpt-base-ar-debug2",
    num_train_epochs=10,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    warmup_ratio=0.08,
    weight_decay=1e-2,
    logging_dir="./logs/speech-chatgpt-base-ar",
    logging_steps=500,
    save_steps=100,
    save_total_limit=5,
    evaluation_strategy="steps",
    eval_steps=400,
    predict_with_generate=True,
    fp16=True,
    learning_rate=1e-5,
    generation_max_length=4096,
    generation_num_beams=1,
    # push_to_hub=True,
    # hub_model_id="lca0503/speech-chatgpt-base-ar",
    report_to="wandb",
)

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    predictions = [i[i != -100] for i in predictions]
    labels = [i[i != -100] for i in labels]

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
    config = BartConfig.from_pretrained("voidful/bart-base-unit")
    config.update({"max_position_embeddings": 4096, "tie_word_embeddings": False})
    model = System(config)
    model.load_hf_pretrained_bart()
    
    tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
    if Define.DEBUG:
        print(model)
        print("System prepared.")
        input()

    # TODO: load config from yaml instead of args
    data_config = {
        "name": args.dataset,
        "codec_name": "speech_tokenizer",  # change this for different codec
        "train_splits": args.train_splits,
        "eval_splits": args.eval_splits,
    }
    model_config = {
        "tokenizer_name": "voidful/bart-base-unit",
        "max_length": config.max_position_embeddings,
    }
    datamodule = DataModule(data_config, model_config, TRAIN_ARGS)
    train_dataset, eval_dataset = datamodule.train_dataset(), datamodule.eval_dataset()
    data_collator = datamodule.data_collator()
    if Define.DEBUG:
        print("Dataset prepared.")
        input()
    
    # Set generation config for evaluation
    generation_config = GenerationConfig.from_model_config(model.config)
    ar_tokens = [f"v_tok_{i}" for i in range(1024)]
    ar_tokens.extend(["<s>", ":", "</s>", "<pad>"])
    allowed_word_ids = [tokenizer.convert_tokens_to_ids(x) for x in ar_tokens]
    ar_bad_words_ids = [[i] for i in range(tokenizer.vocab_size) if i not in allowed_word_ids]
    generation_config.do_sample = True
    generation_config.use_cache = True
    generation_config.bad_words_ids = ar_bad_words_ids
    TRAIN_ARGS.generation_config = generation_config
    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer),
    )

    trainer.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="lca0503/GPTspeech_encodec")
    parser.add_argument("-t", "--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("-e", "--eval_splits", type=str, nargs="+", default=["validation"])
    parser.add_argument("-m", "--model_name", type=str, default="/work/b08902123/SpeechChatGPT/previous_ckpt/tts_ar/checkpoint-75000/")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()    
    return args


if __name__ == "__main__":
    args = parse_args()
    args.dataset = "kuanhuggingface/google_tts_speech_tokenizer"
    args.model_name = "voidful/bart-base-unit"
    Define.DEBUG = args.debug
    main(args)
