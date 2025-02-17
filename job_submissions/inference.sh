#!/bin/bash

# Default values
numparties=""
session=""
distribution=""
label_distribution="subject-wise_complete_uniform_random_"

# Parse command-line options
while getopts ":n:s:d:l:" opt; do
  case $opt in
    n)
      numparties="$OPTARG"
      ;;
    s)
      session="$OPTARG"
      ;;
    l)
      label_distribution="$OPTARG"
      ;;
    d)
      echo "parsing d..."
      distribution="$OPTARG"
      echo "parsed: $distribution"
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

# Check if -s option is provided
if [ -z "$session" ]; then
  echo "Error: -s flag is required." >&2
  exit 1
fi

# Check if distribution is integer and in the correct range
if [ -n "$distribution" ]; then
  if ! is_integer "$distribution" || (( distribution < 1 )) || (( distribution > 5 )); then
    echo "Error: Distribution must be an integer in range [1,5]"
    exit 1
  else
    echo "Updating distribution..."
    distribution="dist${distribution}_"
    echo "Updated: $distribution"
  fi
fi

echo "Number of parties: $numparties"
echo "Session: $session"
echo "Label distribution: $label_distribution"
echo "Distribution: $distribution"

# Create the SBATCH script with dynamic output and error filenames
sbatch_script=$(mktemp)

cat << EOF > $sbatch_script
#!/bin/bash
#SBATCH --job-name=s${session}_inference_federated_gestaltmatcher_service
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --mem-per-cpu=24G
#SBATCH --time=01:20:00
#SBATCH --error=job.s${session}.inference.%J.err
#SBATCH --output=job.s${session}.inference.%J.out

start=\$SECONDS

echo "Starting the inference..."

singularity exec gm_server.simg python3 ../flake_integration_proof_of_concept.py --session $session --num_parties $numparties --num_classes 204 --metadata_file_prefix $label_distribution$distribution

end=\$SECONDS

echo "Evaluation is completed! Time to run: \$((end - start)) seconds"
EOF

# Submit the job
sbatch $sbatch_script

# Cleanup
rm $sbatch_script
