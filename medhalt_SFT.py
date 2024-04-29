from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTTrainer, setup_chat_format
import trl
import datasets
import ipdb

def main():

    model = AutoModelForCausalLM.from_pretrained("gpt2-large",  cache_dir="./data_generation/")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir="./data_generation/")

    # model, tokenizer = setup_chat_format(model, tokenizer)

    medhalt_data = datasets.load_from_disk("/scratch/km6276/DPO-RL/medhalt_generations/few_shots/med_halt_few_shot")
    train_test_split = medhalt_data.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    training_args = TrainingArguments(
        output_dir="SFT_output",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=20,
        logging_steps=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # log_level="debug",
        # save_steps=100,
        learning_rate=1e-4,
        num_train_epochs=1,
        warmup_steps=50,
        # gradient_checkpointing=True,
        # max_grad_norm=0.3,
        lr_scheduler_type="reduce_lr_on_plateau"
    )

    def formatting_prompts_func(data):
        try:
            output_texts = []
            for i in range(len(data['question'])):
                option = f"output{data['correct_idx'][i]}"
                text = f"### Question: {data['prompt'][i]}\n ### Answer: {data[option][i]}"
                output_texts.append(text)
            return output_texts
        except:
            ipdb.set_trace(context=5)
            print("")

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
        max_seq_length=512,
        # dataset_batch_size = 10000,
    )

    trainer.train()

    trainer.save_model("medhalt_sft_gp2_large_lr1e-6")

if __name__ == "__main__":
    main()