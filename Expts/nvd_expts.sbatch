#!/bin/bash
#
#SBATCH --job-name=nvd
#SBATCH --output=out_nvd.txt  # output file
#SBATCH -e err_nvd.txt       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#SBATCH --nodelist=swarm041

#SBATCH --time=0-10:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=4		# Number of tasks
#SBATCH --cpus-per-task=1

module add python/3.7.3 
which python /cm/shared/apps/python/3.7.3/bin/python
which virtualenv /cm/shared/apps/python/3.7.3/bin/virtualenv

virtualenv ~/mypython
source ~/mypython/bin/activate

pip install numpy
pip install docplex

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python attacker_BR.py 0 0 &
python attacker_fplue.py 0 0 &
python attacker_stackelberg.py 0 0 &
python attacker_random.py 0 0 &
python attacker_QR.py 0 0 &
python attacker_probs.py 0 0 &

wait