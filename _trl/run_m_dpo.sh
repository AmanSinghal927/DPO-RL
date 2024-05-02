#!/bin/bash

# Define arrays of parameters
learning_rates=(1e-6 5e-5)
warmups=(50 150 300)
betas=(0.01 0.1 0.3 0.6 0.9)

# Loop over each combination of parameters
for lr in "${learning_rates[@]}"
do
  for warmup in "${warmups[@]}"
  do
    for beta in "${betas[@]}"
    do
      # Submit a job with specific parameters
      sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=training_lr${lr}_warmup${warmup}_beta${beta}
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --time=15:25:00
#SBATCH --mail-type=END
#SBATCH --output=${lr}_${warmup}_${beta}_%j.out
#SBATCH --error=${lr}_${warmup}_${beta}_%j.err

module purge
singularity exec --nv --bind /scratch/as14661 --overlay /scratch/km6276/deeppytorch-example/my_deep.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; python modified_dpo.py lr=${lr} warmup=${warmup} beta=${beta}"
EOT
    done
  done
done
