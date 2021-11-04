#!/bin/sh
#$ -cwd
#PBS -N testpy
#PBS -q workq
#PBS -V
# A PBS script for running adapters code on qsub based clusters
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bt1/18CS10050/.mujoco/mujoco200/bin
export PATH="$PATH:$HOME/rpm/usr/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64"
export LDFLAGS="-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64"
export CPATH="$CPATH:$HOME/rpm/usr/include"
export PATH="/home/bt1/18CS10050/anaconda3/bin:$PATH"
# anaconda environment path.
export PATH="/home/bt1/18CS10050/anaconda3/envs/py3.7/bin:$PATH"

CUDA_VISIBLE_DEVICES=0,1 python job_script.py