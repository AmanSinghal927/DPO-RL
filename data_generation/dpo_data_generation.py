# reading a dataset
# getting samples of data from it 
# calling a model to do generations 
import transformers
import torch
from transformers import AutoTokenizer
import gc
from datasets import load_dataset
import random
import hydra
import _datasets
from _datasets import imdb
from inference import LM
import logging

class Generator:
    def __init__(self, cfg):
        self.cfg = cfg

    def config_tokenizer(self):
        if self.cfg.model.pad_left == True:
            self.tokenizer.padding_side="left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token=self.tokenizer.eos_token


    def configure_model(self):
        if self.cfg.inference.DP and torch.cuda.device_count() > 1:
            print ("Configuring data parallelism for model")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

    def load_model(self):
        print('loading the model')
        self.model_nm = self.cfg.model.name_or_path
        self.device_map = self.cfg.generator.device_map
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_nm, cache_dir=".")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_nm, cache_dir=".", device_map = "cuda")
        self.configure_model()
        self.config_tokenizer()
        print('model loading completed!')
    
    def load_checkpoint(self):
        print('loading the checkpoint')
        self.model.load_state_dict(torch.load(self.cfg.model.checkpoint), strict=False)
        print('checkpoint loading completed!')


@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/config/", config_name="config")
def main(config):
    generator = Generator(config)
    generator.load_model()
    generator.load_checkpoint()
    raw_data = imdb.IMDB()
    raw_data.get_samples(generator.tokenizer, config.generator_dataset)
    

    indices = range(1000)
    first_100_rows = raw_data.sampled_dataset.select(indices)
    lm = LM(generator.tokenizer, generator.model)
    lm.get_inference(first_100_rows, config.inference)
    lm.generated_dataset.save_to_disk("generated_data_without_dp_1000_imdb_4_4_8")


if __name__ == '__main__':
    main()