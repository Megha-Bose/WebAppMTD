#!/bin/bash
#
#SBATCH --job-name=BR
#SBATCH --output=out_BR.txt  # output file
#SBATCH -e err_BR.txt       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#SBATCH --nodelist=swarm041

#SBATCH --time=0-12:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=60		# Number of tasks
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
python attacker_BR.py 101 101 &
python attacker_BR.py 102 102 &
python attacker_BR.py 103 103 &
python attacker_BR.py 104 104 &
python attacker_BR.py 105 105 &
python attacker_BR.py 106 106 &
python attacker_BR.py 107 107 &
python attacker_BR.py 108 108 &
python attacker_BR.py 109 109 &
python attacker_BR.py 110 110 &
python attacker_BR.py 111 111 &
python attacker_BR.py 112 112 &
python attacker_BR.py 113 113 &
python attacker_BR.py 114 114 &
python attacker_BR.py 115 115 &
python attacker_BR.py 116 116 &
python attacker_BR.py 117 117 &
python attacker_BR.py 118 118 &
python attacker_BR.py 119 119 &
python attacker_BR.py 120 120 &
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