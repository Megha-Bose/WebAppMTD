#!/bin/bash
#
#SBATCH --job-name=nvdlarge
#SBATCH --output=out_nvdlarge.txt  # output file
#SBATCH -e err_nvdlarge.txt       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#SBATCH --nodelist=swarm041

#SBATCH --time=0-12:00        # Maximum runtime in D-HH:MM

#SBATCH --ntasks=80		# Number of tasks
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
python attacker_fplue.py 101 101 &
python attacker_fplue.py 102 102 &
python attacker_fplue.py 103 103 &
python attacker_fplue.py 104 104 &
python attacker_fplue.py 105 105 &
python attacker_fplue.py 106 106 &
python attacker_fplue.py 107 107 &
python attacker_fplue.py 108 108 &
python attacker_fplue.py 109 109 &
python attacker_fplue.py 110 110 &
python attacker_fplue.py 111 111 &
python attacker_fplue.py 112 112 &
python attacker_fplue.py 113 113 &
python attacker_fplue.py 114 114 &
python attacker_fplue.py 115 115 &
python attacker_fplue.py 116 116 &
python attacker_fplue.py 117 117 &
python attacker_fplue.py 118 118 &
python attacker_fplue.py 119 119 &
python attacker_fplue.py 120 120 &
python attacker_stackelberg.py 101 101 &
python attacker_stackelberg.py 102 102 &
python attacker_stackelberg.py 103 103 &
python attacker_stackelberg.py 104 104 &
python attacker_stackelberg.py 105 105 &
python attacker_stackelberg.py 106 106 &
python attacker_stackelberg.py 107 107 &
python attacker_stackelberg.py 108 108 &
python attacker_stackelberg.py 109 109 &
python attacker_stackelberg.py 110 110 &
python attacker_stackelberg.py 111 111 &
python attacker_stackelberg.py 112 112 &
python attacker_stackelberg.py 113 113 &
python attacker_stackelberg.py 114 114 &
python attacker_stackelberg.py 115 115 &
python attacker_stackelberg.py 116 116 &
python attacker_stackelberg.py 117 117 &
python attacker_stackelberg.py 118 118 &
python attacker_stackelberg.py 119 119 &
python attacker_stackelberg.py 120 120 &
python attacker_random.py 101 101 &
python attacker_random.py 102 102 &
python attacker_random.py 103 103 &
python attacker_random.py 104 104 &
python attacker_random.py 105 105 &
python attacker_random.py 106 106 &
python attacker_random.py 107 107 &
python attacker_random.py 108 108 &
python attacker_random.py 109 109 &
python attacker_random.py 110 110 &
python attacker_random.py 111 111 &
python attacker_random.py 112 112 &
python attacker_random.py 113 113 &
python attacker_random.py 114 114 &
python attacker_random.py 115 115 &
python attacker_random.py 116 116 &
python attacker_random.py 117 117 &
python attacker_random.py 118 118 &
python attacker_random.py 119 119 &
python attacker_random.py 120 120 &
python attacker_QR.py 101 101 &
python attacker_QR.py 102 102 &
python attacker_QR.py 103 103 &
python attacker_QR.py 104 104 &
python attacker_QR.py 105 105 &
python attacker_QR.py 106 106 &
python attacker_QR.py 107 107 &
python attacker_QR.py 108 108 &
python attacker_QR.py 109 109 &
python attacker_QR.py 110 110 &
python attacker_QR.py 111 111 &
python attacker_QR.py 112 112 &
python attacker_QR.py 113 113 &
python attacker_QR.py 114 114 &
python attacker_QR.py 115 115 &
python attacker_QR.py 116 116 &
python attacker_QR.py 117 117 &
python attacker_QR.py 118 118 &
python attacker_QR.py 119 119 &
python attacker_QR.py 120 120 &
python attacker_probs.py 101 101 &
python attacker_probs.py 102 102 &
python attacker_probs.py 103 103 &
python attacker_probs.py 104 104 &
python attacker_probs.py 105 105 &
python attacker_probs.py 106 106 &
python attacker_probs.py 107 107 &
python attacker_probs.py 108 108 &
python attacker_probs.py 109 109 &
python attacker_probs.py 110 110 &
python attacker_probs.py 111 111 &
python attacker_probs.py 112 112 &
python attacker_probs.py 113 113 &
python attacker_probs.py 114 114 &
python attacker_probs.py 115 115 &
python attacker_probs.py 116 116 &
python attacker_probs.py 117 117 &
python attacker_probs.py 118 118 &
python attacker_probs.py 119 119 &
python attacker_probs.py 120 120 &

wait