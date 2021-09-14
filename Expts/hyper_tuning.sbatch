#!/bin/bash
#
#SBATCH --job-name=hyper
#SBATCH --output=out_hyper.txt  # output file
#SBATCH -e err_hyper.txt       # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 

#SBATCH --time=4-00:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=1			# Number of tasks
#SBATCH --cpus-per-task=1

module add python/3.7.3 
which python /cm/shared/apps/python/3.7.3/bin/python
which virtualenv /cm/shared/apps/python/3.7.3/bin/virtualenv

virtualenv ~/mypython
source ~/mypython/bin/activate

pip install numpy

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python hyper_tuning.py 50 55 0 