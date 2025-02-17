#!/bin/bash

# Default values
numparties=""
numepoch=""
session=""
period=""
aggregation=""
worker=4
distribution=""
label_distribution="subject-wise_complete_uniform_random_"

# Function to display usage information
usage() {
  echo "Usage: $0 -n numparties -e numepoch -s session -f period -a aggregation [-w worker]"
  echo
  echo "Options:"
  echo "  -n    Number of parties (required, valid values: 4, 8, 16)"
  echo "  -e    Number of epochs (required)"
  echo "  -s    Session (required)"
  echo "  -f    Aggregation period (required, must be an integer >= 1)"
  echo "  -a    Aggregation method (required, valid options: mean, weighted_average)"
  echo "  -w    Number of workers (optional, default: 4, must be an integer >= 1)"
  echo "  -l    Distribution name of the experiment (default: 'subject-wise_complete_uniform_random_'"
  echo "  -d    Distribution id of the samples that will be used (current valid values 1 <= x <= 5)"
  echo "  -h    Display this help message"
  exit 1
}

# Parse command-line options
while getopts ":n:e:s:f:a:w:hd:l:" opt; do
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
    w)
      worker="$OPTARG"
      ;;
    l)
      label_distribution="$OPTARG"
      ;;
    d)
      distribution="$OPTARG"
      ;;
    h)
      usage
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

# Check if worker is integer and greater than 1
if ! is_integer "$worker" || (( worker < 1 )); then
  echo "Error: Worker must be an integer greater than or equal to 1." >&2
  exit 1
fi

# Check if worker is integer and greater than 1
if [ -n "$distribution" ]; then
  if ! is_integer "$distribution" || (( distribution < 1 )) || (( distribution > 5 )); then
    echo "Error: Distribution must be an integer in range [1,5]"
    exit 1
  else
    distribution="dist${distribution}_"
  fi
fi

echo "Number of parties: $((numparties))"
echo "Number of epochs: $((numepoch))"
echo "Session: $session"
echo "Period: $period"
echo "Aggregation: $aggregation"
echo "Number of workers: $worker"
echo "Label distribution: $label_distribution"
echo "Distribution ID: $distribution"

#sbatch aggregator.sbatch -n $numparties -e $numepoch -s $session -f $period -a $aggregation
./aggregator.sh -n $numparties -e $numepoch -s $session -f $period -a $aggregation

for ((i=0; i<$numparties; i++))
do
	#sbatch client.sbatch -n $numparties -p $((i+1)) -e $numepoch -s $session -f $period
	./client.sh -n $numparties -p $((i+1)) -e $numepoch -s $session -f $period -w $worker -l $label_distribution -d $distribution
done 

echo "Aggregator and clients are called!"
