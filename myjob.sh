#!/bin/bash -l

## Working dir
#SBATCH -D /users/qj2004/ScaleNet
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J gpu-coraml
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=10
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=80240
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#flight env activate conda@Apr15
#conda activate myenv

#===========================
#  Create results directory
#---------------------------
#RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
#echo "Your results will be stored in: $RESULTS_DIR"
#mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------
echo "1Executing job commands, current working directory is $(pwd)"

echo "Output file has been generated, please check $RESULTS_DIR/test.output"
net_values="scalenet  "
layer_values="4      "

echo "2Executing job commands, current working directory is $(pwd)"
echo "Output file has been generated, please check $RESULTS_DIR/test.output"

Direct_dataset=('ogbn-arxiv'   'directed-roman-empire'   'snap-patents'   'arxiv-year'    )  # Update your Direct_dataset value

generate_timestamp() {
  date +"D%dH%H_%M%S"
}
timestamp=$(generate_timestamp)

for Didataset in "${Direct_dataset[@]}"; do
  Direct_dataset_filename=$(echo $Didataset | sed 's/\//_/g')
for layer in $layer_values; do    # --IsDirectedData --to_undirected
  logfile="${Didataset}outforlayer${layer}.log"  # Adjust log file name with layer number
    exec > $logfile 2>&1  # Redirect stdout and stderr to log file
  for net in $net_values; do
    nohup python3 -m src.run  --model=$net  --dataset="$Didataset" --seed=1  \
      > ${Direct_dataset_filename}_${timestamp}${net}_layer${layer}GPU.log &
    pid=$!

    wait $pid
  done
done
done

