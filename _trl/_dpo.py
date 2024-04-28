import torch
import gdown
import hydra
import copy
import random
from collections import defaultdict
from huggingface_hub import hf_hub_download
from datasets import Dataset, load_from_disk
import transformers
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)

def models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True

class DPO:
    def __init__(self, config):
        self.config = config

        self.training_args = TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            evaluation_strategy=config.evaluation_strategy,
            num_train_epochs=config.num_train_epochs,
            output_dir=config.output_dir,
            report_to=config.report_to,
        )

    def load_model(self):
        print('loading the model')
        self.model_nm = self.config.model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_nm, cache_dir=".")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_nm, cache_dir=".") #, device_map = "cuda")
        self.config_tokenizer()
        print('model loading completed!')

    def config_tokenizer(self):
        self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

    def load_checkpoint(self):
        print('loading the checkpoint')
        try:
            self.model.load_state_dict(torch.load(self.config.checkpoint), strict=False)
        except:
            gdown.download("https://drive.google.com/uc?id=1ZPlfmfkCindqJfD8eNrl8kwtMJ2f1Nqv", self.config.checkpoint, quiet=False)
            self.model.load_state_dict(torch.load(self.config.checkpoint), strict=False)

        # self.configure_model()
        print('checkpoint loading completed!')

    def get_reference_model(self):
        self.model_ref = copy.deepcopy(self.model)
        assert(models_equal(self.model_ref, self.model))

@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/_trl/", config_name="dpo_config")
def main(config):
    # get the model and reference model
    policy = DPO(config)
    policy.load_model()
    policy.load_checkpoint()
    policy.get_reference_model()

    # load the data
    train_dataset = load_from_disk(config.train_dataset)
    # TODO: figure out what eval dataset to have (like what shape and elements required) and how can i define an eval pipeline?
    if config.train_sample:
        train_dataset = train_dataset.select(list(range(1000)))
        eval_dataset = train_dataset.select(list(range(10)))

    trainer = DPOTrainer(
        policy.model,
        policy.model_ref,
        args=policy.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=policy.tokenizer,
        max_target_length=policy.max_target_length,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)

if __name__ == '__main__':
    main()
