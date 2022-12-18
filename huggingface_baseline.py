from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, trainer
from modules.dataset import ChatGPT, DataCollatorForSeq2Seq
import utils


if __name__ == "__main__":
    model_name = "google/mt5-small"
    model_saved_name = model_name.split("/")[-1]

    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    model, percent_reset = utils.search_and_reset_layers(model, tokenizer, scale_down_factor=10, revert_old=False, device='cuda')
    print(percent_reset)
    model.resize_token_embeddings(len(tokenizer))

    args = Seq2SeqTrainingArguments( 
        output_dir=f"{model_name}-finetuned", 
        fp16=True,
        num_train_epochs=40, 
        warmup_steps=1000,
        learning_rate=2e-5,
        # gradient_checkpointing=True,
        gradient_accumulation_steps=15,
        per_device_train_batch_size=9,
        per_device_eval_batch_size=12,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=4,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=1000,
        report_to="tensorboard"
    )
    # must pass in model otherwise no decoder_input_ids
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model, max_length=402)
    train_dataset = ChatGPT('dataset/sharegpt_train', tokenizer)
    val_dataset = ChatGPT('dataset/sharegpt_val', tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset= train_dataset,
        eval_dataset = val_dataset,
        data_collator=seq2seq_data_collator,
        tokenizer=tokenizer
    )
    trainer.train()