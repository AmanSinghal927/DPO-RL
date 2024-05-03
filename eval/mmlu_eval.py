import hydra
import os
import sys
sys.path.append('..')
import transformers
import torch
from transformers import AutoTokenizer
import gc
from datasets import load_dataset, load_from_disk, Dataset
import random
import hydra
import deepspeed
from data_generation import LM
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


def formatting_prompts_test(batch, prompt="prompt"):
    return {"prompt":[f"### Question: {prompt}\n### Answer:" for prompt in batch[prompt]]}

class Generator:
    def __init__(self, cfg):
        self.cfg = cfg

    def config_tokenizer(self):
        if self.cfg.pad_left == True:
            self.tokenizer.padding_side="left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token=self.tokenizer.eos_token


    def configure_model(self):
        if self.cfg.deep_speed:
            self.model = self.model.to(local_rank)
            self.model = deepspeed.init_inference(self.model, 
                                                  tensor_parallel={"tp_size": world_size}, 
                                                  dtype=torch.half, 
                                                  replace_with_kernel_inject=True)
            self.local_rank = local_rank
        # self.model = self.model.cuda()

    def load_model(self):
        print('loading the model')  
        if self.cfg.safetensors == False:
            self.model_nm = self.cfg.name_or_path
            self.tokenizer_nm = self.cfg.name_or_path
        else:
            self.model_nm = self.cfg.direct_path
            self.tokenizer_nm = self.cfg.name_or_path

        self.device_map = self.cfg.device_map
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_nm, cache_dir=".")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_nm, cache_dir=".") #, device_map = "cuda")
        self.config_tokenizer()
        print('model loading completed!')
    
    def load_checkpoint(self):
        if self.cfg.safetensors == False:
            print('loading the checkpoint')
            try:
                self.model.load_state_dict(torch.load(self.cfg.model.checkpoint), strict=False)
            except:
                gdown.download("https://drive.google.com/uc?id=1ZPlfmfkCindqJfD8eNrl8kwtMJ2f1Nqv", self.cfg.model.checkpoint, quiet=False)
                self.model.load_state_dict(torch.load(self.cfg.model.checkpoint), strict=False)

        self.configure_model()
        print('checkpoint loading completed!')

@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/eval/", config_name="mmlu_eval_config.yaml")
def main(config):
    # load model checkpoint and tokenzier
    generator = Generator(config)
    generator.load_model()
    generator.load_checkpoint()
    mmlu = load_from_disk(config.datapath)
    mmlu = mmlu.select(list(range(100)))
    mmlu = mmlu.map(lambda x: formatting_prompts_test(x, config.prompt), batched = True)
    lm = LM(generator.tokenizer, generator.model)
    gen_dict = {}
    gen_dict["eval_generation_batch_size"] = config.eval_generation_batch_size
    gen_dict["eval_generation_max_new_tokens"] = config.eval_generation_max_new_tokens
    gen_dict["eval_generation_top_k"] = config.eval_generation_top_k
    gen_dict["eval_generation_top_p"] = config.eval_generation_top_p
    lm.get_inference_from_dict(mmlu, gen_dict)
    lm.generated_dataset.save_to_disk(os.path.join("/scratch/as14661/dpo_base/eval/data_for_api_eval", "gpt2-large"))



if __name__ == "__main__":
    main()
