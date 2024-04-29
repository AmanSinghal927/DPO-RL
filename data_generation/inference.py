import torch
import time
import logging
from .utils import get_time

class LM:
    def __init__(self, tokenizer, model, local_rank='cuda'):
       self.model = model
       self.tokenizer = tokenizer
       self.local_rank = local_rank
       print ("Created LM object for generation")
       
    def perform_inference(self, samples):
        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.local_rank) for k, v in inputs.items()}
        with torch.no_grad():
            if self.dp and torch.cuda.device_count() > 1:
                generated_ids = model.module.generate(**inputs, do_sample=True,
                                                max_new_tokens=self.max_new_tokens,
                                                pad_token_id=tokenizer.pad_token_id,
                                                top_k=self.top_k,
                                                top_p=self.top_p,)
                                                #  eos_token_id=eos_token_id,) # full-stop criteria
            else:
                generated_ids = model.generate(**inputs, do_sample=True,
                                                max_new_tokens=self.max_new_tokens,
                                                pad_token_id=tokenizer.pad_token_id,
                                                top_k=self.top_k,
                                                top_p=self.top_p,)
                                                #  eos_token_id=eos_token_id,) # full-stop criteria

        generated_texts = [tokenizer.decode(g_id, skip_special_tokens=True) for g_id in generated_ids]
        return generated_texts
    
    def batched_inference(self, batch):
        return {"inferences": [self.perform_inference(sample) for sample in batch['samples']]}
    
    def min_sample(self, sample_list):
        min_length = min([len(s) for s in sample_list])
        min_length_string = [s for s in sample_list if len(s) == min_length][0]
        return min_length_string
    
    def get_inference(self, dataset, cfg):
        print ("Starting inference")
        start_time = time.time()
        self.batch_size = cfg.batch_size
        self.max_new_tokens = cfg.max_new_tokens
        self.top_k = cfg.top_k
        self.top_p = cfg.top_p
        self.dp = cfg.DP
        generated_dataset = dataset.map(self.batched_inference, batched=True, batch_size=self.batch_size)
        generated_dataset = generated_dataset.map(lambda x:{'samples':self.min_sample(x['samples'])})
        self.generated_dataset = generated_dataset
        end_time = time.time()
        get_time(start_time, end_time)
        # logging.info(f"DP is set to: {self.dp}")
        print ("Inference finished!")

    def get_inference_from_dict(self, dataset, dict):
        print ("Starting inference")
        start_time = time.time()
        self.batch_size = dict["eval_generation_batch_size"]
        self.max_new_tokens = dict["eval_generation_max_new_tokens"]
        self.top_k = dict["eval_generation_top_k"]
        self.top_p = dict["eval_generation_top_p"]
        self.dp = False
        generated_dataset = dataset.map(self.batched_inference, batched=True, batch_size=self.batch_size)
        generated_dataset = generated_dataset.map(lambda x:{'samples':self.min_sample(x['samples'])})
        self.generated_dataset = generated_dataset
        end_time = time.time()
        get_time(start_time, end_time)
        # logging.info(f"DP is set to: {self.dp}")
        print ("Inference finished!")
        