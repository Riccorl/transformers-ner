#!/bin/bash

# Pre-start
# checkmark font for fancy log
CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
# usage text
USAGE="$(basename "$0") LANG_MODEL_NAME [-h] [-d] [-p PRECISION] [-c] [-g DEVICES] [-n NODES] [-m GPU_MEM] [-s STRATEGY] [-o] OVERRIDES

where:
    LANG_MODEL_NAME   Language model name (one of the models from HuggingFace)
    -h            Show this help text
    -d            Run in debug mode (no GPU and wandb offline)
    -p            Training precision, default 16.
    -c            Use CPU instead of GPU.
    -g            How many GPU to use, default 1. If 0, use CPU.
    -n            How many nodes to use, default 1.
    -m            Minimum GPU memory required in MB (default: 8000). If less that this,
                  training will wait until there is enough space.
    -s            Strategy to use for distributed training, default NULL.
    -o            Run the experiment offline
    OVERRIDES     Overrides for the experiment, in the form of key=value.
                  For example, 'model_name=bert-base-uncased'
Example:
  ./script/train.sh bert-base-cased
  ./script/train.sh bert-base-cased -m 10000
"

# check for named params
while [ $OPTIND -le "$#" ]; do
  if getopts ":hdp:cgn:m:s:o" opt; then
    case $opt in
    h)
      printf "%s$USAGE" && exit 0
      ;;
    d)
      DEV_RUN="True"
      ;;
    p)
      PRECISION="$OPTARG"
      ;;
    c)
      USE_CPU="True"
      ;;
    g)
      DEVICES="$OPTARG"
      ;;
    n)
      NODES="$OPTARG"
      ;;
    m)
      GPU_MEM="$OPTARG"
      ;;
    s)
      STRATEGY="$OPTARG"
      ;;
    o)
      WANDB="offline"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
      ;;
    esac
  shift $((OPTIND-1))
  else
    LANG_MODEL_NAME=${*:$OPTIND:1}
    ((OPTIND++))
  fi
done

EXTRA_OVERRIDES="$@"

# PRELIMINARIES
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate ner

# Default device is GPU
ACCELERATOR="gpu"

## if LANG_MODEL_NAME is not specified, abort
if [ -z "$LANG_MODEL_NAME" ]; then
  printf "A configuration name must be specified.\n\n"
  printf "%s$USAGE"
  exit 0
fi

# if -d is not specified, DEV_RUN is False
if [ -z "$DEV_RUN" ]; then
  # default value
  DEV_RUN=False
fi

# if -p is not specified, PRECISION is 16
if [ -z "$PRECISION" ]; then
  # default value
  PRECISION=16
fi

# if -c is not specified, USE_CPU is False
if [ -z "$USE_CPU" ]; then
  # default value
  USE_CPU=False
fi

# if -g is not specified, DEVICES is 1
if [ -z "$DEVICES" ]; then
  # default value
  DEVICES=1
fi

# if -n is not specified, NODES is 1
if [ -z "$NODES" ]; then
  # default value
  NODES=1
fi

# if -m is not specified, GPU_MEM is not limited
if [ -z "$GPU_MEM" ]; then
  # default value
  GPU_MEM=0
fi

# if -o is not specified, WANDB is "online"
if [ -z "$WANDB" ]; then
  # default value
  WANDB="online"
fi

# CHECK FOR BOOLEAN PARAMS
# if -d then GPU is not required and no output dir
if [ "$DEV_RUN" = "True" ]; then
  WANDB="offline"
  DEVICES=1
  PRECISION=32
  ACCELERATOR="cpu"
  NODES=1
  USE_CPU="True"
fi

# if -s is not specified, STRATEGY is None
if [ -z "$STRATEGY" ]; then
  # default value
  STRATEGY="null"
fi

# if -g DEVICES is 0 (no GPU) and PRECISION is 32
if [ "$USE_CPU" = "True" ]; then
  # default value
  DEVICES=1
  PRECISION=32
  ACCELERATOR="cpu"
  STRATEGY="null"
fi

if type nvidia-smi >/dev/null 2>&1; then
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
else
  FREE_MEM=0
  GPU_MEM=0
  PRECISION=32
  ACCELERATOR="cpu"
  if [ $USE_CPU = "False" ]; then
    echo -e "GPU not found, fallback to CPU.\n"
  fi
  USE_CPU="True"
fi
GPU_RAM_MESSAGE=""

# echo configuration

cat <<EOF
Configuration:
------------------------------------------------
Language model name:  $LANG_MODEL_NAME
Run in debug mode:    $DEV_RUN
Requested VRAM:       $GPU_MEM MB
Available VRAM:       $FREE_MEM MB
Precision:            $PRECISION bit
Number of GPUs:       $DEVICES
Number of nodes:      $NODES
Use CPU:              $USE_CPU
W&B Mode:             $WANDB

EOF

# WAITING FOR VRAM STUFF
chars="/-\|"
if [ "$FREE_MEM" -lt $GPU_MEM ]; then
  # echo -n "$GPU_RAM_MESSAGE"
  GPU_RAM_MESSAGE="\\rWaiting for at least $GPU_MEM MB of VRAM "
  echo -ne "$GPU_RAM_MESSAGE"
fi
while [ "$FREE_MEM" -lt $GPU_MEM ]; do
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
  for ((i = 0; i < ${#chars}; i++)); do
    sleep 1
    echo -ne "${chars:$i:1} $GPU_RAM_MESSAGE"
  done
done

# if DEV_RUN then GPU is not required
if [ "$DEV_RUN" = "True" ]; then
  echo -n "Debug run started, ignoring GPU memory. "
  GPU_RAM_MESSAGE=""
fi

echo -e "$GPU_RAM_MESSAGE${CHECK_MARK} Starting.\n"

# if you use the `GenerativeDataset` class
# you may want to set `TOKENIZERS_PARALLELISM` to `false`
#export TOKENIZERS_PARALLELISM=false

if [ "$DEV_RUN" = "True" ]; then
  export HYDRA_FULL_ERROR=1
  python transformers_ner/train.py \
    "model.model.language_model=$LANG_MODEL_NAME" \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.devices=$DEVICES" \
    "train.pl_trainer.accelerator=$ACCELERATOR" \
    "train.pl_trainer.num_nodes=$NODES" \
    "train.pl_trainer.strategy=$STRATEGY" \
    "train.pl_trainer.precision=$PRECISION" \
    "hydra.run.dir=." \
    "hydra.output_subdir=null" \
    "hydra/job_logging=disabled" \
    "hydra/hydra_logging=disabled" \
    "$EXTRA_OVERRIDES"
else
  python transformers_ner/train.py \
    "model.model.language_model=$LANG_MODEL_NAME"  \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.devices=$DEVICES" \
    "train.pl_trainer.accelerator=$ACCELERATOR" \
    "train.pl_trainer.num_nodes=$NODES" \
    "train.pl_trainer.strategy=$STRATEGY" \
    "train.pl_trainer.precision=$PRECISION" \
    "logging.wandb_arg.mode=$WANDB" \
    "$EXTRA_OVERRIDES"
fi
