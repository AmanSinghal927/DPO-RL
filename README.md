## DPO
- SFT model: https://drive.google.com/file/d/1ZPlfmfkCindqJfD8eNrl8kwtMJ2f1Nqv/view
- Train vs test for generating prefence data? - Train; test is used for benchmarking model performance against say just the SFTed model e.g. given prompt, complete the generation
   - Open questions:
   - Should I use all the data from IMDB train/test to do this? or Should I just use the positive reviews? - yes all data, yes even the negative ones as starting prompts
   - Should I generate 4 generations for this? - yes
   - How should I get variations? - temprature, different from lengths, what? - experiment with this (only length, length & temprature) -> does avg reward difference variance matter 
- What is the goal with IMDb sentiment analysis? - Given the starting few tokens of a review, write a positive review for a movie e.g. starting tokens could be which movie to write about
-  What do we need to do for this? - Ultimately we need preference data, i.e. given a prompt, we need a response with less +ve sentiment value and a response with more +ve sentiment values
-  How are we going to do this? - We need to create a model that is already good at what it does (Prof Pinto's point). So we must first fine-tune on the domain i.e. +ve sentiment analysis by taking 2-8 tokens of Imdb and then generating the +ve reviews from that
-  Then, to get preference data, we get the model to generate 4 completions given a prompt (not clear if this is the evaluation prompt or the training ones), but essentially these completions are used to generate 6 pairs of data and we get a notion of good or better by using our sentiment classifer
 
![alt text](https://github.com/AmanSinghal927/DPO-RLAIF/blob/main/imdb_exp_1.png)

![alt text](https://github.com/AmanSinghal927/DPO-RLAIF/blob/main/imdb_exp_2.png)


## How to Run
- srun --cpus-per-task=2 --mem=32GB --gres=gpu:rtx8000:2 --time=04:00:00 --pty /bin/bash

## Resources
- https://github.com/eric-mitchell/direct-preference-optimization/issues/45
- https://github.com/QiyaoWei/Reproduce-DPO → starter code: maynot be the best choice
    - https://github.com/eric-mitchell/direct-preference-optimization/issues/28
    - Read more of this repo’s issues section to figure this out
    - Checkout other author’s repositories as well
    - LORA & LLAMA: https://github.com/eric-mitchell/direct-preference-optimization/issues/43

## Notes
- concatenated_batch: Concatenates chosen and rejected together into a single batch (2, 1, 512) -> (4, 1, 512), where 512 is max length
- all_logits: torch.Size([4, 1, 512, 50257]) (distribution over vocabulary)


## Open questions
- Why does greedy decoding lead to degeneration?
- How can I determine if my generations are more human?
https://huggingface.co/blog/how-to-generate
should i use temprature > 0.6 when using top-p and top-k?

## Inference with speedup
Create singularity config in a folder /scratch/user_id/deep
https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda

Changes:
mkdir /scratch/<NetID>/deep
mv overlay-15GB-500K.ext3 my_deep.ext3
conda install conda-forge/label/gcc7::libstdcxx-ng

```
srun --cpus-per-task=1 --mem=32GB --gres=gpu:rtx8000:1 --time=01:00:00 --pty /bin/bash

singularity exec --nv --overlay /scratch/as14661/deep/my_deep.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash

source /ext3/env.sh

deepspeed --num_gpus 2 dpo_data_generation.py

OR

python dpo_data_generation.py

(SINGLE GPU)

Baseline: 23 minutes/1000 examples
Deepspeed: 7 minutes/1000 examples

```

## Datasets
https://drive.google.com/drive/folders/1w0_wujhXpVsptOAAZZyMcyZWndAPseQJ?usp=drive_link
