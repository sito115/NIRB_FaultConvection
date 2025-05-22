#!/usr/bin/zsh 

### Job Parameters 
####   SBATCH --ntasks=1              # Ask for 8 MPI tasks
#SBATCH --cpus-per-task=5              # Optional: allow multi-threaded libs
#SBATCH --time=00:05:00         # Run time 
#SBATCH --job-name=nirb_job  # Sets the job name
#SBATCH --output=/home/kt346075/Documents/logs/stdout_%J.log # redirects stdout and stderr to stdout.txt

### Program Code
cd "$HOME/Documents/NIRB_FaultConvection"
source .venv/bin/activate
python scripts/E_sweep.py