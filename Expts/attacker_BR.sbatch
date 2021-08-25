#!/bin/bash
#
#SBATCH --job-name=BR
#SBATCH --output=out_BR.txt  # output file
#SBATCH -e err_BR.txt       # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to 

#SBATCH --time=4-00:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=5			# Number of tasks
#SBATCH --cpus-per-task=1

module add python/3.7.3 
which python /cm/shared/apps/python/3.7.3/bin/python
which virtualenv /cm/shared/apps/python/3.7.3/bin/virtualenv

virtualenv ~/mypython
source ~/mypython/bin/activate

pip install numpy
pip install docplex
pip install gurobipy

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python attacker_BR.py 0 0 &
python attacker_BR.py 1 1 &
python attacker_BR.py 2 2 &
python attacker_BR.py 3 3 &
python attacker_BR.py 4 4 &
wait