#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc030
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --output=./output/%J.out
#SBATCH --error=./error/%J.err
#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#SBATCH --mail-type=BEGIN,END,FAIL

#### notes

# execute train CLI
#source "$project_dir"/venv/bin/activate
#python "$project_dir"/cli_train.py \


python3 dl-assignment-7.ipynb
