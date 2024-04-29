import torch
import wandb
import hydra
import pandas as pd
from itertools import chain
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.utils.data import DataLoader

# really I just want to iterate on a column and then generate its sentiment scores
class SentInference:
    def __init__(self, col, model, tokenizer, max_length):
        self.col = col
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length 


    def tokenize_function(self, examples):
        if isinstance(examples[self.col][0], list):
            flat_list = list(chain.from_iterable(examples[self.col]))
        else:
            flat_list = examples[self.col]
        return self.tokenizer(flat_list, padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt')


    def inference(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        data_loader = DataLoader(tokenized_datasets, batch_size=512, shuffle=False) 
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        rewards = []
        for batch in tqdm(data_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                # Collect results
                batch_rewards = probabilities[:, 1].tolist()  # Assuming the reward is the second column in logits
                rewards.extend(batch_rewards)

        print ("Mean reward: ", np.mean(rewards), "Reward DEV: ", np.std(rewards))
        return (np.mean(rewards), np.std(rewards))


