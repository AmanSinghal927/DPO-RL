#!/bin/bash

# Define arrays of parameters
learning_rates=(1e-6 5e-5 5e-7)
warmups=(50 150 300)
alphas=(0 5 10 15)

# Loop over each combination of parameters
for lr in "${learning_rates[@]}"
do
  for warmup in "${warmups[@]}"
  do
    for alpha in "${alphas[@]}"
    do
      # Submit a job with specific parameters
      sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=training_lr${lr}_warmup${warmup}_alpha${alpha}
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --time=15:25:00
#SBATCH --mail-type=END
#SBATCH --output=${lr}_${warmup}_${alpha}_%j.out
#SBATCH --error=${lr}_${warmup}_${alpha}_%j.err

module purge
singularity exec --nv --bind /scratch/as14661 --overlay /scratch/as14661/deep/my_deep.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; python _sft.py lr=${lr} warmup=${warmup} alpha=${alpha}"
EOT
    done
  done
done
