import os
import glob
import random
from datasets import load_dataset, load_from_disk

class IMDB:
    def __init__(self):
        # download data if it's not there
        self.raw_path = "/scratch/as14661/dpo_base/data_generation/downloads"

        if 'imdb' not in [os.path.basename(x) for x in glob.glob(os.path.join(self.raw_path, "*"))]:
            print ("Downloading dataset")
            self.dataset = load_dataset("stanfordnlp/imdb", split="train", trust_remote_code=True)
            self.dataset.save_to_disk(os.path.join(self.raw_path, 'imdb'))
            print ("Downloading complete!")
        else:
            print("Loading saved dataset")
            self.dataset = load_from_disk(os.path.join(self.raw_path, 'imdb'))
            print("Loading complete!")


    def prompt_sampling(self, text):
        """
        tokenize, then get random number of tokens between 2,8 and then detokenize
        padding and truncation not needed as data length is going to be between 2,8
        """
        return [self.tokenizer.decode(self.tokenizer(text)["input_ids"][0:random.randint(self.start, self.end)]) for _ in range(self.num_samples)]
    
    def get_samples(self, tokenizer, cfg):
        sample_cached = cfg.sample_cached
        num_samples = cfg.num_samples
        start = cfg.start
        end = cfg.end
        print ("Generating samples")


        if sample_cached == False:
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.start = start
            self.end = end
            sampled_dataset = self.dataset.map(lambda x:{'samples':self.prompt_sampling(x['text'])})
            sampled_dataset.save_to_disk(os.path.join(self.raw_path, 'imdb_'+str(num_samples)+"_"+str(start)+"_"+str(end)))
            self.sampled_dataset = sampled_dataset
        else:
            self.sampled_dataset = load_from_disk(os.path.join(self.raw_path, 'imdb_'+str(num_samples)+"_"+str(start)+"_"+str(end)))
        print ("Finished generating samples!")



        


