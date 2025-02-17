#!/bin/bash

# Default values
numparties=""
numepoch=""
session=""
period=""
aggregation=""

# Parse command-line options
while getopts ":n:e:s:f:a:" opt; do
  case $opt in
    n)
      numparties="$OPTARG"
      ;;
    e)
      numepoch="$OPTARG"
      ;;
    s)
      session="$OPTARG"
      ;;
    f)
      period="$OPTARG"
      ;;
    a)
      aggregation="$OPTARG"
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

# Check if -a option is provided
if [ -z "$aggregation" ]; then
  echo "Error: -a flag is required." >&2
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

echo "Number of parties: $numparties"
echo "Number of epochs: $numepoch"
echo "Session: $session"
echo "Period: $period"
echo "Aggregation: $aggregation"

# Cleanup the session's id list if exists any
rm ${session}_job_ids

# Create the SBATCH script with dynamic output and error filenames
sbatch_script=$(mktemp)

cat << EOF > $sbatch_script
#!/bin/bash
#SBATCH --job-name=s${session}_federated_gestaltmatcher_service
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=02:00:00
#SBATCH --error=job.s${session}.%J.err
#SBATCH --output=job.s${session}.%J.out

echo "${SLURM_JOB_ID}\n" >> ${session}_job_ids

start=\$SECONDS

echo "Starting the aggregator..."
singularity exec gm_server.simg python3 ../aggregator.py --session $session --epochs $numepoch --aggregation_method $aggregation --data_dir ../data --federated_metadata_path federated_metadata --dataset gmdb --in_channels 3 --img_size 112 --weight_dir ../saved_models --model_dir ../models --num_parties $numparties --aggregation_period $period

end=\$SECONDS

echo "Aggregator is finished. Time to run: \$((end - start)) seconds"
EOF

# Submit the job
sbatch $sbatch_script

# Cleanup
rm $sbatch_script

