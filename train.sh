#!/bin/sh

#SBATCH --job-name=train_clip
#SBATCH -p bioe
#SBATCH --time=1-00:00:00
#SBATCH -G 1
#SBATCH --output=/home/users/jihyeonj/clip/run.log

source ~/.bashrc
source activate jjenv
export WANDB_API_KEY=fc5ac49fdfa453bfea20126fe268c37ba348d4cd
python run_train.py

