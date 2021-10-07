#!/bin/bash
#
#SBATCH --job-name=criticalvul
#SBATCH --output=out_criticalvul.txt  # output file
#SBATCH -e err_criticalvul.txt       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 

#SBATCH --time=0-12:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=4		# Number of tasks
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

python criticalvul_BR.py 0 0 2
python criticalvul_BR.py 0 0 5
python criticalvul_BR.py 0 0 10
python criticalvul_BR.py 0 0 15
wait