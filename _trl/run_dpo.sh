#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=2
#SBATCH --nodes 1
#SBATCH --mem=32GB
#SBATCH --time=14:25:00
#SBATCH --mail-type=END
#SBATCH --output=%jmain.out
#SBATCH --error=%jmaintester.err
module purge
singularity exec --nv --bind /scratch/as14661 --overlay /scratch/as14661/deep/my_deep.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; python _dpo.py"