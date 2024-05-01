import sys
sys.path.append('..')
import torch
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTTrainer
import trl
import wandb
from datasets import Dataset, load_from_disk
import numpy as np
import pandas as pd
import hydra
import torch.optim as optim
from eval_mcq import EvalMCQ
from transformers.integrations import WandbCallback
from sklearn.metrics import accuracy_score



# def prompt_format(example):
#     correct_idx = example["correct_idx"]
#     correct_option = f"output{example['correct_idx']}"
#     return {"formated_prompt":f"### Question: {example['prompt']}\n### Answer: {example[correct_option]}"}

class MMLUEvalCallback(WandbCallback):
    def __init__(self, config_dict):
        super().__init__()
        self.eval_pipeline = EvalMCQ()
        self.config_dict = config_dict
        self.max_accuracy = 0.0

    def on_evaluate(self, args, state, control, model, **kwargs):
        self.eval_pipeline.eval(model)
        accuracy = accuracy_score(
            list(self.eval_pipeline.id_gt_map.values()), 
            list(self.eval_pipeline.id_pred_map.values())
            )

        self._wandb.log({"mmlu_accuracy": accuracy}, step=state.global_step)

        if accuracy>self.max_accuracy:
            print ("saving model!")
            print ("prev:", self.max_accuracy)
            print ("current:", accuracy)

            self.max_accuracy = accuracy

            save_path = os.path.join("/scratch/as14661/dpo_base/sft/checkpoints", "lr_"+str(self.config_dict["lr"])+"_warmup_"+str(self.config_dict["warmup_steps"])+"_alpha_"+str(self.config_dict["neft_alpha"]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)



def filter_short_entries(example):
    # Check each column in the example
    for value in example.values():
        if (value is None) or (str(value)=="None") or (isinstance(value, str) and len(value) < 5):
            return False  # Filter out this example
    return True


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    else:
        return 1

class SFT:
    def __init__(self, config):
        self.config = config

    def config_tokenizer(self):
        # self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

    def load_model(self):
        print('loading the model')
        self.model_nm = self.config.model_nm
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_nm, cache_dir=".")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_nm, cache_dir=".") #, device_map = "cuda")
        self.config_tokenizer()
        print('model loading completed!')

    def process_example(self, example):
        correct_idx = example["correct_idx"]
        correct_option = f"output{example['correct_idx']}"
        example[correct_option] = example[correct_option].strip()
        input_text = f"### Question: {example['prompt']}\n### Answer:{example[correct_option]}"
        encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        offset = len(self.tokenizer(f"### Question: {example['prompt']}\n### Answer:")["input_ids"])
        labels = encoding.input_ids.clone().detach()
        labels[:, :offset] = -100
        labels[labels==self.tokenizer.pad_token_id] = -100
        labels = labels.squeeze().tolist()

        return {'input_ids': encoding['input_ids'].squeeze().tolist(), 
                'attention_mask': encoding['attention_mask'].squeeze().tolist(), 
                'labels': labels}

@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/sft/", config_name="sft_config")
def main(config):
    global warmup_steps
    lr = config.lr
    warmup_steps = config.warmup
    neft_alpha = config.alpha

    config_dict = {"lr":lr, "warmup_steps":warmup_steps, "neft_alpha":neft_alpha}

    print ("learning rate set", lr)
    print ("warmup set", warmup_steps)
    print ("NEFT Alpha", neft_alpha)

    policy = SFT(config)
    policy.load_model()
    wandb.init(project="mmlu_supervised_finetuning", entity="as14661", name="lr_"+str(lr)+"_warmup_"+str(warmup_steps)+"_alpha_"+str(neft_alpha))

    optimizer = optim.RMSprop(policy.model.parameters(), lr=lr) 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    medhalt_data = load_from_disk(config.train_dataset)
    medhalt_data = medhalt_data.filter(filter_short_entries)
    medhalt_data = medhalt_data.map(lambda x: policy.process_example(x))

    if config.train_sample:
        train_dataset = medhalt_data.select(list(range(1000)))
        eval_dataset = medhalt_data.select(list(range(10)))
        print (train_dataset)
    else:
        # do a train-eval split
        train_eval_split = medhalt_data.train_test_split(test_size=0.1, seed=42)  # 10% for evaluation, 90% for training
        train_dataset = train_eval_split['train']
        eval_dataset = train_eval_split['test'].select(list(range(200)))

    training_args = TrainingArguments(
        output_dir = "sft_outputs",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        num_train_epochs=1,
    )

    

    trainer = SFTTrainer(
        policy.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers = (optimizer, scheduler),
        formatting_func=policy.process_example,
        callbacks = [MMLUEvalCallback(config_dict)],
        args=training_args,
        neftune_noise_alpha = neft_alpha,
        max_seq_length=512,
    )

    trainer.train()


if __name__ == '__main__':
    main()