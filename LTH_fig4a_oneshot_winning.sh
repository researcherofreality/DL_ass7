#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc030
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=./output/%J.out
#SBATCH --error=./error/%J.err
#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#SBATCH --mail-type=BEGIN,END,FAIL

echo "start run" 

source "/scratch/IMC070_deeplearning/venv/bin/activate"
python3 LTH_fig4a_oneshot_winning.py

echo "done running"
