#!/bin/bash
#
#SBATCH --job-name=BR
#SBATCH --output=out_BR.txt  # output file
#SBATCH -e err_BR.txt       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#SBATCH --nodelist=swarm041

#SBATCH --time=0-12:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=41			# Number of tasks
#SBATCH --cpus-per-task=1

module add python/3.7.3 
which python /cm/shared/apps/python/3.7.3/bin/python
which virtualenv /cm/shared/apps/python/3.7.3/bin/virtualenv

virtualenv ~/mypython
source ~/mypython/bin/activate

pip install numpy
pip install gurobipy

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python attacker_BR.py 0 0 &
python attacker_BR.py 1 1 &
python attacker_BR.py 2 2 &
python attacker_BR.py 3 3 &
python attacker_BR.py 4 4 &
python attacker_BR.py 5 5 &
python attacker_BR.py 6 6 &
python attacker_BR.py 7 7 &
python attacker_BR.py 8 8 &
python attacker_BR.py 9 9 &
python attacker_BR.py 10 10 &
python attacker_BR.py 11 11 &
python attacker_BR.py 12 12 &
python attacker_BR.py 13 13 &
python attacker_BR.py 14 14 &
python attacker_BR.py 15 15 &
python attacker_BR.py 16 16 &
python attacker_BR.py 17 17 &
python attacker_BR.py 18 18 &
python attacker_BR.py 19 19 &
python attacker_BR.py 20 20 &
python attacker_BR.py 1 1 0 &
python attacker_BR.py 2 2 0 &
python attacker_BR.py 3 3 0 &
python attacker_BR.py 4 4 0 &
python attacker_BR.py 5 5 0 &
python attacker_BR.py 6 6 0 &
python attacker_BR.py 7 7 0 &
python attacker_BR.py 8 8 0 &
python attacker_BR.py 9 9 0 &
python attacker_BR.py 10 10 0 &
python attacker_BR.py 11 11 0 &
python attacker_BR.py 12 12 0 &
python attacker_BR.py 13 13 0 & 
python attacker_BR.py 14 14 0 &
python attacker_BR.py 15 15 0 &
python attacker_BR.py 16 16 0 &
python attacker_BR.py 17 17 0 &
python attacker_BR.py 18 18 0 &
python attacker_BR.py 19 19 0 &
python attacker_BR.py 20 20 0 &
wait