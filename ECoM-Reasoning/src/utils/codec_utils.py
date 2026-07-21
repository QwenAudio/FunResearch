from slam_llm.utils.train_utils import print_module_size
import torch
import torchaudio
import os
import torch.nn as nn
import uuid
from utils.cosyvoice.utils.file_utils import load_wav

def setup_codec(train_config, model_config, **kwargs):
    if model_config.codec_decoder_type == "CosyVoice":
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        codec_decoder = AutoModel(model_dir=model_config.codec_decoder_path)
        codec_decoder_module = nn.ModuleList((codec_decoder.model.flow,codec_decoder.model.hift))
    else:
        raise NotImplementedError
    print_module_size(codec_decoder_module, model_config.codec_decoder_type + " Codec", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder


def get_single_layer_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio):
    audio_length = len(audio_tokens) + num_latency_tokens + 1  # 1 is due to end of audio token
    result = [padding_token] * num_latency_tokens + list(audio_tokens) + [end_of_audio]
    result_tensor = torch.tensor(result).unsqueeze(0)
    return result_tensor, audio_length


def get_group_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio, num_layers):
    padded_audio_tokens = audio_tokens + [end_of_audio]
    padding_needed = (num_layers - len(padded_audio_tokens) % num_layers ) % num_layers
    
    # Add padding to ensure even distribution across layers
    padded_audio_tokens = padded_audio_tokens + [padding_token] * padding_needed
    total_length = len(padded_audio_tokens)
    audio_length = total_length // num_layers + num_latency_tokens

    # Create the result for each layer
    result = []
    for layer in range(num_layers):
        layer_tokens = [padding_token] * num_latency_tokens
        layer_tokens.extend(padded_audio_tokens[layer::num_layers])
        result.append(torch.tensor(layer_tokens))
    
    result_tensor = torch.stack(result)
    return result_tensor, audio_length


def get_group_answer_token_v2(audio_tokens, num_latency_tokens, padding_token, end_of_audio, num_layers):
    assert num_latency_tokens == 0 , "num_latency_tokens must be 0 for get_group_answer_token_v2"
    padded_audio_tokens = audio_tokens + [end_of_audio]
    padding_needed = (num_layers - len(padded_audio_tokens) % num_layers ) % num_layers
    
    # Add padding to ensure even distribution across layers
    padded_audio_tokens = padded_audio_tokens + [end_of_audio] * padding_needed 
    total_length = len(padded_audio_tokens)
    audio_length = total_length // num_layers + num_latency_tokens 

    # Create the result for each layer
    result = []
    for layer in range(num_layers):
        layer_tokens = [padding_token] * num_latency_tokens
        layer_tokens.extend(padded_audio_tokens[layer::num_layers])
        result.append(torch.tensor(layer_tokens))
    
    result_tensor = torch.stack(result)
    return result_tensor, audio_length


def audio_decode_cosyvoice(audio_tokens, model_config, codec_decoder, tone_dir, audio_prompt_path=None, code_layer=1, num_latency_tokens=1, speed=1.0, replace_token=6560):
    # CosyVoice 3 model
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Generated audio waveform.
    """
    
    # Reshape audio tokens based on code_layer
    if code_layer > 1:
        audio_tokens_tensor = torch.stack(audio_tokens, dim=0)
        audio_tokens_permuted = audio_tokens_tensor.permute(1, 0)
        audio_tokens = audio_tokens_permuted.reshape(-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens * code_layer:]
    else:
        audio_tokens = torch.cat(audio_tokens, dim=-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens:]

    # Get vocabulary configuration for end of audio (EOA) and padding token
    eoa = model_config.vocab_config.eoa
    pad_a = model_config.vocab_config.pad_a

    # Truncate audio tokens at the EOA token 
    eoa_mask = (audio_tokens[0] == eoa)
    if not eoa_mask.any():
        # No EOA token found → no valid audio generated
        print("No end-of-audio (EOA) token found in generated tokens. Skipping audio decoding.")
        return None  # or return torch.zeros(...) if you need a placeholder
    else:
        end_index = torch.nonzero(eoa_mask, as_tuple=True)[0][0].item()
    audio_tokens = audio_tokens[..., :end_index]

    # Handle padding tokens if present # FIXME: this is a temporary fix for the padding issue, where the padding token may be included in the audio tokens
    if pad_a in audio_tokens:
        audio_tokens = audio_tokens.masked_fill(audio_tokens == pad_a, replace_token)

    # Generate a unique ID for this audio generation
    this_uuid = str(uuid.uuid1())

    # Set up the prompt speech features and speaker embedding
    if tone_dir == "default_tone":
        audio_prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "audio_prompt", "en", "prompt_1.wav"
        )

    ###### new path for cosyvoice3 ######
    flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(audio_prompt_path)
    prompt_speech_16k = load_wav(audio_prompt_path, 16000)
    prompt_speech_24k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
    prompt_speech_24k_dir = os.environ.get("COSYVOICE_TMP_DIR", "/tmp/cosyvoice_prompt")
    os.makedirs(prompt_speech_24k_dir, exist_ok=True)
    prompt_speech_24k_path = os.path.join(prompt_speech_24k_dir, "prompt_speech_24k.wav")
    torchaudio.save(prompt_speech_24k_path, prompt_speech_24k, 24000)
    prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_24k_path)
    flow_embedding = codec_decoder.frontend._extract_spk_embedding(audio_prompt_path)

    # Convert tokens to audio waveform
    audio_hat = codec_decoder.model.token2wav(
        token=audio_tokens,
        prompt_token=flow_prompt_speech_token,
        prompt_feat=prompt_speech_feat,
        embedding=flow_embedding,
        token_offset=0,
        uuid=this_uuid,
        finalize=True,
        speed=speed
    )

    return audio_hat
