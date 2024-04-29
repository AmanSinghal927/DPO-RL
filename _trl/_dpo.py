import sys
sys.path.append('..')
import os
import wandb
import copy
import torch
import gdown
import hydra
import nltk

# from nltk.util import ngrams
# from nltk.tokenize import word_tokenize
from collections import Counter
from copy import deepcopy
import random
from collections import defaultdict
from huggingface_hub import hf_hub_download
from datasets import Dataset, load_from_disk
import transformers
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from trl import DPOTrainer
from data_generation.inference import LM
from eval.eval_inference import SentInference
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, AutoModelForSequenceClassification
import deepspeed
from transformers.integrations import WandbCallback

os.environ["WANDB_PROJECT"]="imdb_preference_optimization"
world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

def get_unique_ngrams(sentences, n):
    all_ngrams = []
    for sentence in sentences:
        tokens = sentence.split()
        sentence_ngrams = zip(*[tokens[i:] for i in range(n)])
        all_ngrams.extend(sentence_ngrams)
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams)

class SentEvalCallback(WandbCallback):
    def __init__(self, generator_tokenizer, generator_dict):
        super().__init__()
        self.eval_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english").to("cuda")
        self.eval_tokenizer = AutoTokenizer.from_pretrained('siebert/sentiment-roberta-large-english')
        eval_dataset = load_from_disk("/scratch/as14661/dpo_base/data_generation/data/beam_search_imdb_test")
        self.generator_tokenizer = generator_tokenizer
        self.generator_dict = generator_dict
        self.eval_dataset = eval_dataset.remove_columns('inferences')
        self.eval_dataset = self.eval_dataset.select(list(range(256)))
        self.mean_reward = 0

    def configure_model(self, model):
        model = model.to(local_rank)
        model = deepspeed.init_inference(model, 
                                        tensor_parallel={"tp_size": world_size}, 
                                        dtype=torch.half, 
                                        replace_with_kernel_inject=True)
        return model
        

    def on_evaluate(self, args, state, control, model, **kwargs):
        generator_model = self.configure_model(deepcopy(model))
        generation_pipline = LM(self.generator_tokenizer, generator_model)
        generation_pipline.get_inference_from_dict(self.eval_dataset, self.generator_dict)
        generated_dataset = generation_pipline.generated_dataset
        sent_inference = SentInference('inferences', self.eval_model, self.eval_tokenizer, 512)
        mean_rewards, std_rewards = sent_inference.inference(generated_dataset)
        list_sentences = [i[0] for i in generated_dataset["inferences"]]
        self._wandb.log({"mean_eval_rewards": mean_rewards}, step=state.global_step)
        self._wandb.log({"eval_unigrams": get_unique_ngrams(list_sentences, 1)}, step=state.global_step)
        self._wandb.log({"eval_bigrams": get_unique_ngrams(list_sentences, 2)}, step=state.global_step)
        self._wandb.log({"eval_trigrams": get_unique_ngrams(list_sentences, 3)}, step=state.global_step)
        self._wandb.log({"eval_ngrams": get_unique_ngrams(list_sentences, 4)}, step=state.global_step)



        if mean_rewards>self.mean_reward:
            save_path = os.path.join(self.generator_dict["save_path"], "beta_"+ str(self.generator_dict["beta"]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)

        # log their average sentiment score to wandb or somewhere

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
        # self.tokenizer.padding_side="left"
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

def lr_lambda(current_step):
    if current_step < 150:
        return current_step / 150
    else:
        return 1

@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/_trl/", config_name="dpo_config")
def main(config):
    # get the model and reference model
    policy = DPO(config)
    policy.load_model()
    policy.load_checkpoint()
    policy.get_reference_model()

    wandb.init(project="imdb_preference_optimization", entity="as14661", name="beta_"+str(config.beta))

    optimizer = optim.RMSprop(policy.model.parameters(), lr=1e-6) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # load the data
    train_dataset = load_from_disk(config.train_dataset)
    if config.train_sample:
        train_dataset = train_dataset.select(list(range(1000)))
        eval_dataset = train_dataset.select(list(range(10)))
    else:
        # do a train-eval split
        train_eval_split = train_dataset.train_test_split(test_size=0.1)  # 10% for evaluation, 90% for training

        # Access the splits
        train_dataset = train_eval_split['train']
        eval_dataset = train_eval_split['test'].select(list(range(200)))

    # for evaluation
    generator_dict = {"eval_generation_batch_size":config.eval_generation_batch_size, 
                      "eval_generation_max_new_tokens": config.eval_generation_max_new_tokens,
                      "eval_generation_top_k": config.eval_generation_top_k,
                      "eval_generation_top_p":config.eval_generation_top_p, 
                      "beta":config.beta,
                      "save_path":config.save_path}

    trainer = DPOTrainer(
        policy.model,
        policy.model_ref,
        beta = config.beta,
        args=policy.training_args,
        optimizers = (optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=policy.tokenizer,
        callbacks = [SentEvalCallback(policy.tokenizer, generator_dict)],
        max_target_length=config.max_target_length,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)

if __name__ == '__main__':
    main()
