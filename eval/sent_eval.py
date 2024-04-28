import torch
import hydra
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset

tokenizer = AutoTokenizer.from_pretrained('siebert/sentiment-roberta-large-english')

def tokenize_function(examples, config):
    return tokenizer(examples[config.inferences], padding="max_length", truncation=True, max_length=config.max_length, return_tensors='pt')

def get_ids(dataset, config):
    df = dataset.to_pandas()
    df[config.id] = list(range(len(df)))
    dataset = Dataset.from_pandas(df)
    return dataset

def group_by_id(data, config):
    """
    function to collect rewards
    """
    grouped_data = {}
    for item in data:
        id = item[config.id]
        if id not in grouped_data:
            grouped_data[id] = []
        grouped_data[id].append({config.inference: item[config.inference], config.reward: item[config.reward]})
    
    result = []
    for id, values in grouped_data.items():
        result.append({
            config.id: id,
            config.inferences: values
        })
    return result

def flatten_list_column(dataset, list_column_name):
    def expand_lists(example):
        new_examples = []
        for item in example[list_column_name]:
            new_example = {key: value if key != list_column_name else item for key, value in example.items()}
            new_examples.append(new_example)
        return new_examples
    new_dataset = dataset.map(lambda example: {'expanded': expand_lists(example)}, batched=False, remove_columns=dataset.column_names)
    flat_dataset = Dataset.from_dict({k: [dic[k] for batch in new_dataset['expanded'] for dic in batch] for k in new_dataset['expanded'][0][0]})
    return flat_dataset


def merge_details(original, preference_data, config):
    """
    merge the details back into the original dataset
    """
    original_df = original.to_pandas()
    preference_data_df = pd.DataFrame(preference_data)
    preference_data_df = preference_data_df.merge(original_df[[config.id, "samples", "text"]], on=config.id, how="left")
    preference_data_df.columns = [config.id, config.chosen, config.rejected, config.chosen_reward, config.rejected_reward, config.prompt, config.benchmark]
    print (preference_data_df.head()) 
    enriched_dataset = Dataset.from_pandas(preference_data_df)
    return enriched_dataset

def create_comparison_results(data, config):
    results = []
    for item in data:
        id = item[config.id]
        inferences = item[config.inferences]
        # Generate all pairs of inferences
        for i in range(len(inferences)):
            for j in range(i + 1, len(inferences)):
                inf1 = inferences[i]
                inf2 = inferences[j]
                if inf1[config.reward] > inf2[config.reward]:
                    chosen, rejected = inf1[config.inference], inf2[config.inference]
                    chosen_reward, rejected_reward = inf1[config.reward], inf2[config.reward]
                else:
                    chosen, rejected = inf2[config.inference], inf1[config.inference]
                    chosen_reward, rejected_reward = inf2[config.reward], inf1[config.reward]
                results.append({config.id: id, config.chosen: chosen, config.rejected: rejected, config.chosen_reward:chosen_reward, config.rejected_reward:rejected_reward})
    return results
    
@hydra.main(version_base=None, config_path="/scratch/as14661/dpo_base/eval/", config_name="eval_config")
def main(config):
    if config.testing:
        dataset = load_from_disk(config.path).select(list(range(1024)))
    else:
        dataset = load_from_disk(config.path)
    dataset = get_ids(dataset, config) # add ids to the dataset
    model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
    # flatten dataset and generate tokens
    new_dataset = flatten_list_column(dataset, config.inferences) # flatten the inferences
    tokenized_datasets = new_dataset.map(tokenize_function, batched=True, fn_kwargs = {"config":config})
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', config.id])
    data_loader = DataLoader(tokenized_datasets, batch_size=512, shuffle=False)  
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rewards = []
    results = []
    for batch in tqdm(data_loader):
        batch_ids = batch[config.id]
        inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

        # generate rewards
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # Collect results
            batch_rewards = probabilities[:, 1].tolist()  # Assuming the reward is the second column in logits
            rewards.extend(batch_rewards)
            for id, inference, reward in zip(batch_ids, inputs['input_ids'], batch_rewards):
                # Convert input_ids back to text if necessary
                text_inference = tokenizer.decode(inference, skip_special_tokens=True)
                results.append({config.id:id.item(), config.inference: text_inference, config.reward: reward})

    # collect all the results by their ids
    if config.save == True:
        grouped_results = group_by_id(results, config)
        preference_data = create_comparison_results(grouped_results, config)
        enriched_dataset = merge_details(dataset, preference_data, config)
        enriched_dataset.save_to_disk(config.path+"_rewards")

    print ("Completed ", len(rewards), " examples")
    print ("Mean reward: ", np.mean(rewards), "Reward DEV: ", np.std(rewards))



if __name__ == "__main__":
    main()