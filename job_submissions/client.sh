#!/bin/bash

# Default values
numparties=""
numepoch=""
session=""
pid=""
period=""
worker=4
distribution=""
label_distribution=""

# Parse command-line options
while getopts ":n:e:s:p:f:w:d:l:" opt; do
  case $opt in
    n)
      numparties="$OPTARG"
      ;;
    e)
      numepoch="$OPTARG"
      ;;
    p)
      pid="$OPTARG"
      ;;
    s)
      session="$OPTARG"
      ;;
    f)
      period="$OPTARG"
      ;;
    w)
      worker="$OPTARG"
      ;;
    l)
      label_distribution="$OPTARG"
      ;;
    d)
      distribution="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Function to check if argument is an integer
is_integer() {
    local s="$1"
    if [[ "$s" =~ ^-?[0-9]+$ ]]; then
        return 0
    else
        return 1
    fi
}

# Check if -n option is provided
if [ -z "$numparties" ]; then
  echo "Error: -n flag is required." >&2
  exit 1
fi

# Check if numparties is greater than 1
if (( numparties != 4 )) && (( numparties != 8 )) && (( numparties != 16)); then
  echo "Error: Number of parties must be 4 or 8 or 16." >&2
  exit 1
fi

# Check if -e option is provided
if [ -z "$numepoch" ]; then
  echo "Error: -e flag is required." >&2
  exit 1
fi

# Check if -s option is provided
if [ -z "$session" ]; then
  echo "Error: -s flag is required." >&2
  exit 1
fi

# Check if -p option is provided
if [ -z "$pid" ]; then
  echo "Error: -p flag is required." >&2
  exit 1
fi

# Check if -f option is provided
if [ -z "$period" ]; then
  echo "Error: -f flag is required." >&2
  exit 1
fi

# Check if period is greater than 1
if ! is_integer "$period" || (( period < 1 )); then
  echo "Error: Period must be an integer greater than or equal to 1." >&2
  exit 1
fi

# Check if worker is integer and greater than 1
if ! is_integer "$worker" || (( worker < 1 )); then
  echo "Error: Worker must be an integer greater than or equal to 1." >&2
  exit 1
fi

echo "Number of parties: $numparties"
echo "Party id: $pid"
echo "Number of epochs: $numepoch"
echo "Session: $session"
echo "Period: $period"
echo "Number of workers: $worker"
echo "Label distribution: $label_distribution"
echo "Distribution: $distribution"

# Create the SBATCH script with dynamic output and error filenames
sbatch_script=$(mktemp)

cat << EOF > $sbatch_script
#!/bin/bash
#SBATCH --job-name=s${session}_federated_gestaltmatcher_service
#SBATCH --cpus-per-task=${worker}
#SBATCH --partition=day
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=02:00:00
#SBATCH --error=job.s${session}.%J.err
#SBATCH --output=job.s${session}.%J.out

echo "${SLURM_JOB_ID}\n" >> ${session}_job_ids

start=\$SECONDS

echo "Training encoder model for party ${pid}..."
singularity exec gm_server.simg python3 ../client.py --encoding_dir ../encodings/ --num_classes 204 --num_workers $worker --epochs $numepoch --session $session --dataset gmdb --in_channels 3 --img_size 112 --use_tensorboard --data_dir ../data --weight_dir ../saved_models --model_dir ../models --lookup_table_save_path ../lookup_table --federated_metadata_path federated_metadata --num_parties $numparties --party_id $pid --aggregation_period $period --metadata_file_prefix $label_distribution$distribution

end=\$SECONDS

echo "Federated encoder models are trained for party ${pid}. Time to run: \$((end - start)) seconds"
EOF

# Submit the job
sbatch $sbatch_script

# Cleanup
rm $sbatch_script

