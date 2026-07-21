#!/bin/bash
# NOTE: run this script from the repository root, e.g. `bash src/scripts/finetune_com.sh`
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# CosyVoice / codec decoding needs the active conda env's shared libs on the path
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
export WANDB_MODE=offline

code_dir=src
num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))


###################################################################################################
project_name=slam-com

# Initialization checkpoint: a pretrained S2S base-model directory containing `model.pt`.
# The COM model is fine-tuned on top of this base. Set it to your own pretrained checkpoint.
# To train from scratch instead, remove the `++ckpt_path=...` line from hydra_args below.
ckpt_path=/path/to/pretrained_s2s_base_ckpt


# com settings
dataset_version=mix-com
model_file="src/model/slam_model_s2t.py:model_factory"  

# Dataset manifests (YAML), paths relative to the repo root (the run directory).
# Edit the jsonl paths inside these YAMLs to point at your built data (see data/README.md).
train_data_path="data/train_mix_dataset.yaml"
val_data_path="data/test_math_dataset.yaml"


###################################################################################################


# model settings
group_decode=true
group_decode_adapter_type=linear

llm_path="checkpoints/Qwen2.5-1.5B"    # base LLM (git-ignored link under checkpoints/)
llm_name=Qwen2.5-1.5b
llm_dim=1536                      # Qwen2.5

encoder_path="checkpoints/whisper-small/small.pt"   # Whisper encoder checkpoint
encoder_dim=768                     # 384 512 768 1024 1280
whisper_size=small                  # tiny small medium large-v3
mel_size=80                         # 80 128 ( only whisper-large-v3 supports 128 )


# vocabulary settings
code_layer=3                       # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=6625       # the vocab size of the codec token 6561+64 
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen3 151936+64)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))


# dataset settings
manifest_format=jsonl
load_from_cache_file=true           # set to true if you have already generated the cache file, otherwise set to false
code_type=CosyVoice                 # CosyVoice3 
num_latency_tokens=0                # number of delay tokens (in front of the generated audio tokens) default=0
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks


# training settings
batch_size_training=2 # 1 2
use_fp16=true # bf16: false
use_peft=false
num_epochs=10
lr=1e-5
task_type=s2s
warmup_steps=3000
total_steps=300000
validation_interval=6150 # pretrain 6390, post 6150
split_size=0.01


# log settings
exp_name="s2s_train-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-nofp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
if [ "$use_fp16" = true ]; then
    exp_name="s2s_train-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-fp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
fi
use_wandb=true
wandb_entity_name=your_wandb_entity        # only used if use_wandb=true and WANDB_MODE != offline
wandb_project_name=$project_name
wandb_exp_name=$exp_name
home_dir="./exp/${project_name}"           # where checkpoints/logs are written
output_dir=$home_dir/$exp_name


hydra_args="
hydra.run.dir=$output_dir \
++model_config.file=$model_file \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=linear \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.code_type=$code_type \
++model_config.group_decode=$group_decode \
++model_config.group_decode_adapter_type=$group_decode_adapter_type \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.dataset_version=$dataset_version \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=$mel_size \
++dataset_config.seed=42 \
++dataset_config.manifest_format=$manifest_format \
++dataset_config.split_size=$split_size \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.code_type=$code_type \
++dataset_config.num_latency_tokens=$num_latency_tokens \
++dataset_config.do_layershift=$do_layershift \
++train_config.model_name=s2s \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=$lr \
++train_config.validation_interval=$validation_interval \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$batch_size_training \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.task_type=$task_type \
++train_config.use_peft=$use_peft \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=100 \
++ckpt_path=$ckpt_path/model.pt \
"
# ↑ `++ckpt_path=$ckpt_path/model.pt` loads the pretrained base weights (init / resume).
#   Remove that line from hydra_args above to train from scratch.

sys_prompt="prompt.yaml"  # last version setting (useless now)

if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name $sys_prompt \
        $hydra_args

else
    torchrun \
        --nnodes $num_nodes \
        --nproc_per_node $num_gpus_per_node \
        --master_port=29503 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name $sys_prompt \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false\
        $hydra_args
fi

# for multi-machine training, you should add the following line to the torchrun command
# --node_rank=$node_rank \
# --master_addr=$master_addr \

# bash examples/s2s/scripts/finetune/finetune_s2s_group.sh