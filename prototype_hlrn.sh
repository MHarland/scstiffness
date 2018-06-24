#!/bin/bash -x
#PBS -N {%jobname%}
#PBS -l nodes={%n_nodes%}:ppn=24
#PBS -l walltime={%walltime%}
#PBS -l feature=mpp1
#PBS -A {%accountname%}
#PBS -d {%cwd%}
#PBS -j oe
#PBS -q mpp1q
#PBS -o {%cwd%}/{%name%}.log
#PBS -v MKL_NUM_THREADS=1

source /home/h/hhpmharl/.bashrc
triqsenv2
PYTHONPATH="{%cwd%}:$PYTHONPATH" aprun -n {%n_tasks%} {%pyname%}
