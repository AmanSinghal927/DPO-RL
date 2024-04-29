import json
import glob
from datasets import Dataset
import ipdb
import os

os.chdir('/scratch/km6276/DPO-RL/medhalt_generations/few_shots/')

files = [f"_{n}.json" for n in range(1,16961,10)]

data = []
error_files = []
for filename in files:
    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)
            data.extend(json_data)  # Assuming each file contains a list of records
    except:
        # ipdb.set_trace(context=5)
        error_files.append(filename)


print(f"{error_files=}")

dataset = Dataset.from_list(data)

dataset.save_to_disk('med_halt_few_shot')


