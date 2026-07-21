import torch
import os
import logging
import torch.nn.functional as F
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional, Generator
from slam_llm.utils.metric import compute_accuracy
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
from utils.tts_adapter_utils import setup_tts_adapter
from utils.codec_utils import setup_codec
from utils.trick_utils import partial_freeze_weights, train_embedding_layer_only, train_embedding_layer
from utils.snac_utils import get_snac, generate_audio_data, simple_shift
from utils.snac_utils import layershift as layer_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

logger = logging.getLogger(__name__)

import json
import time
from datetime import datetime

class TokenTool:
    def __init__(self, log_path: str = None):
        if log_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_dir = os.environ.get("SLAM_REMARK_DIR", "/tmp/slam_remark")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"token_{ts}.jsonl")
        self.log_path = log_path
        self.reset()

    def reset(self):
        self.num_steps = 0       # number of update() calls (== generated tokens)
        self.text_tokens = 0     # cumulative text-token count
        self.first_audio_latency = None  # first-packet latency (seconds)
        self.input_complete_time = None  # timestamp when input finished

    def update(self):
        self.num_steps += 1

    def update_text(self, num_new_tokens: int = 1):
        """Call once per generated token; may also add several at once."""
        self.text_tokens += int(num_new_tokens)

    def mark_input_complete(self):
        """Mark the timestamp when input finished."""
        self.input_complete_time = time.perf_counter()

    def mark_first_audio(self):
        """Mark first-audio-token time and compute first-packet latency."""
        if self.input_complete_time is not None:
            self.first_audio_latency = time.perf_counter() - self.input_complete_time
            print("\nFirst audio latency:", self.first_audio_latency, "seconds")

    def log(self):
        record = {
            "steps": self.num_steps,
            "text_tokens": self.text_tokens,
            "first_audio_latency_sec": self.first_audio_latency,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.reset()

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    whisper_model = None
    if train_config.task_type == "s2s" :
        if not model_config.whisper_decode:
            encoder = setup_encoder(train_config, model_config, **kwargs)
        else:
            whisper_model = setup_encoder(train_config, model_config, **kwargs)
            encoder = whisper_model.encoder
    else:
        raise NotImplementedError

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    if encoder is not None:
        encoder_projector = setup_encoder_projector(
            train_config, model_config, **kwargs
        )
        if train_config.freeze_encoder_projector:
            for name, param in encoder_projector.named_parameters():
                param.requires_grad = False
            encoder_projector.eval()
    else:
        encoder_projector = None

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    tts_adapter = None
    if model_config.tts_adapter:
        adapter_config = model_config.tts_adapter_config
        tts_adapter = setup_tts_adapter(adapter_config, model_config, **kwargs)

    group_decode_adapter = None
    if model_config.group_decode:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_s2t(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.train_audio_embed_only:
        partial_freeze_weights(model, model_config.vocab_config.padded_text_vocabsize, model_config.vocab_config.total_vocabsize)

    if train_config.train_embed_only:
        train_embedding_layer_only(model)

    if train_config.train_embed:
        train_embedding_layer(model)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer



class slam_model_s2t(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer 
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)

            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.whisper_model = whisper_model
        self.tts_adapter = tts_adapter
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter
        
        # token_tool is only used at inference; None during training
        decode_log_dir = kwargs.get("decode_log", None)
        if decode_log_dir is not None:
            token_log_path = f"{decode_log_dir.rstrip('/')}/token_len.jsonl"
            os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
            if not os.path.exists(token_log_path):
                open(token_log_path, "a", encoding="utf-8").close()
            self.token_tool = TokenTool(log_path=token_log_path)
        else:
            self.token_tool = None


    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        ### input & config
        audio_mel = kwargs.get("audio_mel", None)
        audio_embedding = kwargs.get("audio_embedding", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper
        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        modality_mask = kwargs.get("modality_mask", None)
        text_labels = kwargs.get("text_labels", labels) # same length as labels, but masks everything after com_aat
        audio_labels = kwargs.get("audio_labels", None) # per-layer; same length as labels, masks com_aat and before
        audio_input_ids = kwargs.get("audio_input_ids", None) 
        do_layershift = kwargs.get("do_layershift", False)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        com_aa = self.model_config.vocab_config.com_aat    
        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a

        ### encode audio
        encoder_outs = None
        if audio_mel is not None or audio is not None:
            if audio_embedding is None:
                if self.train_config.freeze_encoder: # freeze encoder
                    self.encoder.eval()

                if self.model_config.encoder_name == "whisper":
                    encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim

                if self.encoder is None:
                    encoder_outs = audio_mel if audio_mel is not None else audio
            else:
                encoder_outs = audio_embedding

            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        ### encode text
        if input_ids is not None:
            input_ids[input_ids == -1] = 0  # [btz, seq_length]

            assert hasattr(self.llm.model, "embed_tokens"), "Please make sure that the llm model has embed_tokens"
            inputs_embeds = self.llm.model.embed_tokens(input_ids)  # [btz, seq_length, emb_dim]
            
            if audio_labels is not None:
                for i in range(input_ids.shape[0]):
                    mask = (input_ids[i] == com_aa)
                    if mask.any():
                        com_aa_positions = mask.nonzero(as_tuple=True)[0]
                        if len(com_aa_positions) == 0:
                            continue
                        pos = com_aa_positions[0].item()
                        start_replace = pos + 1

                        if start_replace >= input_ids.shape[1]:
                            continue 
                        
                        audio_tokens = audio_input_ids[i, :, start_replace:]

                        audio_emb = self.llm.model.embed_tokens(audio_tokens)
                        
                        inputs_embeds[i, start_replace:, :] = torch.mean(audio_emb, dim=0) # multi_layer mean

        ### modality concat serial
        if modality_mask is not None and encoder_outs is not None: 
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)  # [batch_size, seq_length]
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds) # [batch_size, seq_length, dim]
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outs[i][:modality_lengths[i]]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        ### infer
        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        ### llm infer
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        ### serial generation
        x_ori = model_outputs.logits  # [B, L, total_vocab]; unified labels: text ids [0, text_vocab) + audio ids [text_vocab, total_vocab)
        
        xt = x_ori[..., :text_vocab_size]
        

        ### group decode
        xa = []

        need_audio_loss = (
            audio_labels is not None 
            and self.group_decode_adapter.linear.weight.requires_grad 
            and not torch.all(audio_labels == -100)  # ensure some valid labels
        )

        if self.group_decode_adapter is not None and need_audio_loss:
            xa_ori = x_ori[..., text_vocab_size:]
            x_audio = self.group_decode_adapter(xa_ori)
            for i in range(self.code_layer):
                xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])

        # calculate loss
        loss_recorder = []
        if text_labels is None:
            raise NotImplementedError("text labels not provided")

        total_loss, loss_recorder = self.compute_loss(xt, text_labels, xa, audio_labels)
        model_outputs.loss = total_loss

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)]
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(xt, -1)
                text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                if len(xa) > 0: 
                    preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                    audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]


        return model_outputs, text_acc, audio_acc, loss_recorder

    def compute_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1) ]
        
        if text_labels is not None:
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100)
            layer_loss[self.code_layer] = text_loss
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels is not None and audio_labels[:,i] is not None and self.train_config.task_type != "asr":
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        total_loss = (text_loss + total_audio_loss) / (self.code_layer+1)
        return total_loss, layer_loss

    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        print("\n\n\ngenerate\n\n\n")

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        next_token = None
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", False)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        # com token
        com_qt = self.model_config.vocab_config.com_qt
        com_at = self.model_config.vocab_config.com_at
        com_aa = self.model_config.vocab_config.com_aat

        text_end = False     # whether the text stream ended (by special token)
        audio_end = True   # Track whether audio generation has ended
        first_non_pad = 0 


        if decode_text_only:
            print("Decoding text only")

        # NOTE: currently, we only support greedy decoding and sampling for parallel generation, no beam search
        for step in tqdm(range(max_new_tokens), desc="Generating"):

            if current_input_text is not None and (not text_end or current_audio_tokens[0] is None):
                combined_input_ids = current_input_text.unsqueeze(1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(combined_input_ids)

            elif current_audio_tokens[0] is not None and not audio_end:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(audio_tokens)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(audio_tokens)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(audio_tokens)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            

            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            if self.token_tool is not None:
                self.token_tool.update()

            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step


            # Split logits into text and audio layers based on vocab size
            next_token = None
            if not text_end:
                xt_logits = logits[..., :text_vocab_size]
                xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
                current_input_text = next_token_text
                next_token = next_token_text
            
            next_tokens_audio = []
            if not audio_end:
                if self.group_decode_adapter is not None:
                    xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                    xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
                else:
                    xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]
                
                for i in range(self.code_layer):
                    xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][first_non_pad:step], audio_repetition_penalty) # audio repetition penalty

                for i in range(self.code_layer):
                    if not audio_end and not decode_text_only and num_latency_tokens <= step:
                        next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                    else:
                        next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                    next_tokens_audio.append(next_token_audio)
            
                for i in range(self.code_layer):
                    current_audio_tokens[i] = next_tokens_audio[i]
                
           
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            if not text_end:
                generated_ids[self.code_layer][step] = next_token_text  # Text layer
                if self.token_tool is not None:
                    self.token_tool.update_text()
                first_non_pad += 1
            elif not audio_end:
                for i in range(self.code_layer):
                    generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
                if eoa in next_tokens_audio:
                    break
            else:
                break
            
            if next_token is not None: 
                if next_token == com_aa:
                    text_end = True
                    audio_end = False

        # Concatenate the generated tokens to form the complete sequence
        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][-1]] if eot in text_tokens else text_tokens 

        for i in range(self.code_layer):
            audio_tokens = generated_ids[i]
            generated_ids[i] = audio_tokens[first_non_pad: ]  # drop leading pad tokens

        if upsampling_factor > 1:
            generated_ids = generated_ids[::upsampling_factor]

        if self.token_tool is not None:
            self.token_tool.log()

        return generated_ids


    @torch.no_grad()
    def stream_generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ) -> Generator:
        """
        Streaming generation: yields one token at a time.
        Yields: {"text_token": token_id, "audio_tokens": [layer0, layer1, ...], "is_text_end": bool, "is_audio_end": bool}
        """
        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # mark input-processing done (for latency timing)
        if self.token_tool is not None:
            self.token_tool.mark_input_complete()

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        next_token = None
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", False)
        text_token_limit = kwargs.get("text_token_limit", None)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        # com token
        com_qt = self.model_config.vocab_config.com_qt
        com_at = self.model_config.vocab_config.com_at
        com_aa = self.model_config.vocab_config.com_aat

        text_end = False
        audio_end = True
        first_non_pad = 0
        first_audio_yielded = False

        for step in range(max_new_tokens):

            if current_input_text is not None and (not text_end or current_audio_tokens[0] is None):
                combined_input_ids = current_input_text.unsqueeze(1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(combined_input_ids)

            elif current_audio_tokens[0] is not None and not audio_end:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(audio_tokens)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(audio_tokens)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(audio_tokens)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if self.token_tool is not None:
                self.token_tool.update()

            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values

            # Split logits into text and audio layers based on vocab size
            next_token = None
            next_token_text = None
            if not text_end:
                if text_token_limit is not None and self.token_tool.text_tokens >= text_token_limit:
                    next_token_text = torch.tensor([com_aa], device=input_ids.device)
                else:
                    xt_logits = logits[..., :text_vocab_size]
                    xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
                    next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
                current_input_text = next_token_text
                next_token = next_token_text

            next_tokens_audio = []
            if not audio_end:
                if self.group_decode_adapter is not None:
                    xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                    xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
                else:
                    xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

                for i in range(self.code_layer):
                    xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][first_non_pad:step], audio_repetition_penalty)

                for i in range(self.code_layer):
                    if not audio_end and not decode_text_only and num_latency_tokens <= step:
                        next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                    else:
                        next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                    next_tokens_audio.append(next_token_audio)

                for i in range(self.code_layer):
                    current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            stream_output = {
                "text_token": next_token_text.item() if next_token_text is not None else None,
                "audio_tokens": [t.item() for t in next_tokens_audio] if next_tokens_audio else [],
                "is_text_end": text_end,
                "is_audio_end": audio_end,
                "step": step,
            }

            # Append generated tokens to the tensor
            if not text_end:
                generated_ids[self.code_layer][step] = next_token_text
                if self.token_tool is not None:
                    self.token_tool.update_text()
                first_non_pad += 1
                yield stream_output
            elif not audio_end:
                for i in range(self.code_layer):
                    generated_ids[i][step] = next_tokens_audio[i]
                
                # record first audio-token time (latency)
                if not first_audio_yielded:
                    if self.token_tool is not None:
                        self.token_tool.mark_first_audio()
                    first_audio_yielded = True
                
                yield stream_output
                
                if eoa in next_tokens_audio:
                    break
            else:
                break

            if next_token is not None:
                if next_token == com_aa:
                    text_end = True
                    audio_end = False

        #self.token_tool.log()


    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """

        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)
        
        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)


    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits