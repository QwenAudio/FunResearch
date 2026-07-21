#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


code_dir=src

##################################################################
# load the backbone model
dataset_version="mix-com" 


# Checkpoint to run — a git-ignored dir under checkpoints/ containing model.pt
# (see checkpoints/README.md). Switch to checkpoints/comthinksatst4-0310 for the other model.
ckpt_path=checkpoints/comthink-0302
output_dir=./exp/infer/$(basename "$ckpt_path")
model_file="src/model/slam_model_s2t.py:model_factory"

sys_prompt="prompt.yaml"
online_output_dir=$output_dir

##################################################################

# model settings
group_decode=true
group_decode_adapter_type=linear
llm_path="checkpoints/Qwen2.5-1.5B"                 # base LLM (same as training)
codec_decoder_path="checkpoints/Fun-CosyVoice3-0.5B" # CosyVoice codec/vocoder for audio decoding
encoder_path="checkpoints/whisper-small/small.pt"   # Whisper encoder checkpoint

encoder_dim=768                      # 384 512 768 896 1024 1280 
mel_size=80                          # 80 128 (128 for whisper-large only, 80 for others)
llm_dim=1536                        # 896 1536 2048 3584  -> 0.5B 1.5B 3B 7B

code_type=CosyVoice                 # CosyVoice or SNAC
codec_decoder_type=CosyVoice
num_latency_tokens=0                # number of latency tokens (same as the number in training)
do_layershift=false                

task_type=s2s
use_peft=false

# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=6625          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))


# decode config
text_repetition_penalty=1.0
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false
input_text=false

output_text_only=false
speech_sample_rate=24000          # 22050 for CosyVoice, 24000 for CosyVoice3
inference_online=true
audio_prompt_path=src/audio_prompt/en/prompt_1.wav     # replace this with your own audio prompt path or our provided audio prompt path

decode_log=$output_dir/s2s_decode_${split}_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_seed${dataset_sample_seed}_greedy
if [ "$do_sample" = true ] ; then
    decode_log=$output_dir/s2s_decode_${split}_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_seed${dataset_sample_seed}_sampling_topk${top_k}_topp${top_p}_temp${temperature}
fi

if [ "$decode_text_only" = true ] ; then
    decode_log=$decode_log"_text_only"
fi

# -m debugpy --listen 5678 --wait-for-client
mkdir -p "$output_dir"
python $code_dir/inference_s2s.py \
        --config-path "conf" \
        --config-name $sys_prompt \
        hydra.run.dir=$output_dir \
        ++model_config.file=$model_file \
        ++model_config.llm_name=qwen2-0.5b \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.dataset_version=$dataset_version \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.inference_mode=true \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.freeze_encoder_projector=true \
        ++train_config.freeze_group_decode_adapter=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++train_config.use_peft=$use_peft \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++log_config.online_output_dir=$online_output_dir \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path \
        # ++peft_ckpt_path=$peft_ckpt_path/model.pt

