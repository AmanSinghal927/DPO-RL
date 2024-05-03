import transformers
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import formatting_prompts_test, get_ids, get_index, flatten_list_columns
from datasets import load_dataset, load_from_disk, Dataset
import random
from sklearn.metrics import accuracy_score
import tqdm




class EvalMCQ:
    def __init__(self, tokenizer='gpt2-large', dataset_pth = "/scratch/as14661/dpo_base/sft/data/mmlu/content/mmlu"):
        self.max_length = 256
        self.batch_size = 32
        self.prompt = "prompt"
        self.options = "options"
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=".",  device_map="auto")
        self.config_tokenizer()
        self.dataset = load_from_disk(dataset_pth)
        self.config_dataset()
    
    def config_tokenizer(self):
        self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

    def tokenize_function(self, examples, formated_prompt = "formated_prompt"):
        return self.tokenizer(examples[formated_prompt] + examples[self.options], padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt')

    def config_dataset(self):
        self.dataset = self.dataset.map(lambda x: formatting_prompts_test(x, self.prompt), batched = True)
        self.dataset = get_ids(self.dataset)
        self.dataset = self.dataset.map(lambda x: get_index(x, self.options))
        self.flat_dataset = flatten_list_columns(self.dataset, [self.options, "idx"])
        self.tokens_flat_dataset = self.flat_dataset.map(self.tokenize_function)
        self.tokens_flat_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "id", "idx", "correct_idx"])

    def eval(self, model_or_checkpoint):

        if isinstance(model_or_checkpoint, str):
            model = AutoModelForCausalLM.from_pretrained(model_or_checkpoint, cache_dir=".").to(self.device)
        else:
            model = model_or_checkpoint.to(self.device)

        data_loader = DataLoader(self.tokens_flat_dataset, batch_size=self.batch_size, shuffle=False)
        self.id_ce_map = {}
        self.id_pred_map = {}
        self.id_gt_map = {}

        for batch in tqdm.tqdm(data_loader):
            batch_ids = batch["id"]
            indices = batch["idx"]
            truth = batch["correct_idx"]

            inputs = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            inputs_for_model =  {k: v[:, :, :-1].contiguous() for k, v in inputs.items()}
            # huggingface does this shifting internally & automatically
            labels_for_model = inputs['input_ids'][:, :, 1:].contiguous()

            # generate rewards
            with torch.no_grad():
                outputs = model(**inputs_for_model)
            losses = torch.nn.functional.cross_entropy(outputs.logits.permute(0,3,1,2), labels_for_model, reduction = 'none')
            losses = losses.mean(dim=-1)
            for id, ce, idx, gt in zip(batch_ids, losses, indices, truth):
                id = id.item()
                if id not in self.id_ce_map or ce < self.id_ce_map[id]:
                    self.id_ce_map[id] = ce.item()
                    self.id_pred_map[id] = idx.item()
                self.id_gt_map[id] = gt.item()


# if __name__=="__main__":
#     checkpoint = "/scratch/as14661/dpo_base/sft/checkpoints/medhalt_sft_gp2_large_512/medhalt_sft_gp2_large_512"
#     mcq = EvalMCQ(checkpoint, tokenizer='gpt2-large', dataset_pth = "/scratch/as14661/dpo_base/sft/data/mmlu/content/mmlu")
#     mcq.eval()
#     accuracy = accuracy_score(list(mcq.id_gt_map.values()), list(mcq.id_pred_map.values()))
#     print("Accuracy:", accuracy)



        



    




