from datasets import load_dataset, load_from_disk, Dataset


def formatting_prompts_test(batch, prompt="prompt"):
    return {"formated_prompt":[f"### Question: {prompt}\n### Answer:" for prompt in batch[prompt]]}

def get_ids(dataset, id = "id"):
    df = dataset.to_pandas()
    df[id] = list(range(len(df)))
    dataset = Dataset.from_pandas(df)
    return dataset

def get_index(example, options="options"):
    return {"idx":list(range(len(example[options])))}

def flatten_list_columns(dataset, list_column_names):
    column1, column2 = list_column_names

    def expand_lists(example):
        new_examples = []
        for item1, item2 in zip(example[column1], example[column2]):
            new_example = {key: value for key, value in example.items()}
            new_example[column1] = item1
            new_example[column2] = item2
            new_examples.append(new_example)
        return new_examples
    
    new_dataset = dataset.map(lambda example: {'expanded': expand_lists(example)}, batched=False, remove_columns=dataset.column_names)
    flat_dataset = Dataset.from_dict({
        k: [dic[k] for batch in new_dataset['expanded'] for dic in batch] 
        for k in new_dataset['expanded'][0][0]
    })

    return flat_dataset