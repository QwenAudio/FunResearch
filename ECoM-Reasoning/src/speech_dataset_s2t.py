import json
import copy
import math
import os
import random

import numpy as np

import torch
import whisper
from utils.snac_utils import layershift, get_snac_answer_token, simple_shift
from utils.codec_utils import get_single_layer_answer_token, get_group_answer_token
import librosa

from dataclasses import dataclass,field
from typing import Optional, List, Any, Dict
import os
import yaml


################### Mix Dataset ######################
@dataclass
class CompressConfig:
    stage: Optional[str] = None
    rate: float = 1.0

@dataclass
class DatasetConfig:
    name: str
    dataset_path: str
    data_struct: str = "com-math"
    sampling_rate: float = 1.0
    compress: CompressConfig = field(default_factory=CompressConfig)
    system_prompt: str = ""

@dataclass
class DataConfigFile:
    datasets: List[DatasetConfig] = field(default_factory=list)

def _expand_path(p: str) -> str:
    # support ~/xx and $HOME/xx style paths
    return os.path.expandvars(os.path.expanduser(p))

def load_config_from_yaml(path: str, check_exists: bool = True) -> DataConfigFile:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_datasets = raw.get("datasets", [])
    if not isinstance(raw_datasets, list):
        raise ValueError("YAML field `datasets` must be a list.")

    datasets: List[DatasetConfig] = []
    seen_names = set()

    for i, d in enumerate(raw_datasets):
        if not isinstance(d, dict):
            raise ValueError(f"datasets[{i}] must be a dict.")

        name = d.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"datasets[{i}].name is required and must be a string.")
        if name in seen_names:
            raise ValueError(f"Duplicate dataset name: {name}")
        seen_names.add(name)

        dataset_path = d.get("dataset_path")
        if not dataset_path or not isinstance(dataset_path, str):
            raise ValueError(f"datasets[{i}].dataset_path is required and must be a string.")
        dataset_path = _expand_path(dataset_path)

        data_struct = d.get("data_struct", "com-math")

        sampling_rate = float(d.get("sampling_rate", 1.0))
        if not (0.0 < sampling_rate <= 1.0):
            raise ValueError(f"datasets[{i}].sampling_rate must be in (0, 1]. Got {sampling_rate}")

        compress_raw = d.get("compress", {}) or {}
        if not isinstance(compress_raw, dict):
            raise ValueError(f"datasets[{i}].compress must be a dict or null.")

        compress = CompressConfig(
            stage=compress_raw.get("stage", None),
            rate=float(compress_raw.get("rate", 1.0)),
        )
        if compress.rate <= 0:
            raise ValueError(f"datasets[{i}].compress.rate must be > 0. Got {compress.rate}")

        if check_exists and not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path} (name={name})")

        system_prompt = d.get("system_prompt", None)

        datasets.append(
            DatasetConfig(
                name=name,
                dataset_path=dataset_path,
                data_struct=data_struct,
                sampling_rate=sampling_rate,
                compress=compress,
                system_prompt=system_prompt,
            )
        )

    return DataConfigFile(datasets=datasets)



class SpeechDatasetJsonl_CoM(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "<SYSTEM>: {}\n "
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        self.manifest_format = dataset_config.get("manifest_format", "parquet")
        self.seed = dataset_config.get("seed", 42)
        self.rng = random.Random(self.seed)
        self.split_size = dataset_config.get("split_size", 0.1)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        assert self.manifest_format in ["parquet", "jsonl"], "manifest_format must be one of [parquet, jsonl]"

        # vocab config
        self.vocab_config = dataset_config.get("vocab_config", None)
        self.text_vocabsize = self.vocab_config.text_vocabsize
        self.text_specialtokens = self.vocab_config.text_specialtokens
        self.audio_vocabsize = self.vocab_config.audio_vocabsize
        self.audio_specialtokens = self.vocab_config.audio_specialtokens
        self.padded_text_vocabsize = self.vocab_config.padded_text_vocabsize
        self.padded_audio_vocabsize = self.vocab_config.padded_audio_vocabsize
        self.total_vocabsize = self.vocab_config.total_vocabsize
        self._eot = self.vocab_config.eot
        self._pad_t = self.vocab_config.pad_t
        self._input_t = self.vocab_config.input_t
        self._answer_t = self.vocab_config.answer_t
        self._asr = self.vocab_config.asr
        self._eoa = self.vocab_config.eoa
        self._pad_a = self.vocab_config.pad_a
        self._input_a = self.vocab_config.input_a
        self._answer_a = self.vocab_config.answer_a
        self._split = self.vocab_config.split
        self.code_layer = self.vocab_config.code_layer
        
        self._com_qt = self.vocab_config.com_qt
        self._com_at = self.vocab_config.com_at
        self._com_think = self.vocab_config.com_think
        self._com_aat= self.vocab_config.com_aat

        self.special_token_a = self._answer_a
        self.special_token_t = self._answer_t

        # task config 
        self.task_type = dataset_config.get("task_type", "s2s")

        # upsample config 
        self.upsample_text_tokens = dataset_config.get("upsample_text_tokens", False)
        self.upsampling_factor = dataset_config.get("upsampling_factor", 1)
        self.upsample_method = dataset_config.get("upsample_method", "repeat")

        # code type config
        self.code_type = dataset_config.get("code_type", "SNAC")
        if self.code_type != "SNAC" and self.code_type != "CosyVoice":
            raise ValueError("code_type must be one of [SNAC, CosyVoice]")
        
        # number of tokens for latency
        self.num_latency_tokens = dataset_config.get("num_latency_tokens", 1)

        # layershift config
        self.do_layershift = dataset_config.get("do_layershift", True)
        if self.do_layershift:
            self.layershift = layershift
        else:
            self.layershift = simple_shift

        # read dataset
        if split == "train":
            list_json_path = dataset_config.train_data_path
        elif split in ("val", "valid", "validation"):
            list_json_path = getattr(dataset_config, "val_data_path", None)
            if list_json_path is None:
                raise ValueError("dataset_config.val_data_path not found")
        elif split == "test":
            list_json_path = getattr(dataset_config, "test_data_path", None)
            if list_json_path is None:
                raise ValueError("dataset_config.test_data_path not found")
        else:
            raise ValueError(f"Unknown split: {split}")   
               

        self.cfg = load_config_from_yaml(list_json_path)
        self.data_list=[]
        self.data_struct_list=[]
        for ds in self.cfg.datasets:
            rate = float(ds.sampling_rate)
            with open(ds.dataset_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    # sample
                    if random.random() >= rate:
                        continue

                    data_dict = json.loads(line)
                    self.data_list.append(data_dict)
                    self.data_struct_list.append((ds.data_struct, ds.compress.stage, ds.compress.rate, ds.system_prompt))
        
        mix_flag = True
        if mix_flag: # sequential or mixed training
            rng = random.Random(self.seed)
            idx = list(range(len(self.data_list)))
            rng.shuffle(idx)
            self.data_list = [self.data_list[i] for i in idx]
            self.data_struct_list = [self.data_struct_list[i] for i in idx]


    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)

    def extract_audio_feature(self, audio_path):
        # audio path is a dictionary, resample the audio to 16kHz
        if self.manifest_format == "parquet" and isinstance(audio_path, dict):
            audio_raw = audio_path['array']
            audio_raw_sr = audio_path['sampling_rate']
            if not isinstance(audio_raw, np.ndarray):
                audio_raw = np.array(audio_raw)
            audio_raw = librosa.resample(audio_raw, orig_sr=audio_raw_sr, target_sr=16000).astype(np.float32)
        elif (self.manifest_format == "parquet" and (isinstance(audio_path, str) or isinstance(audio_path, list))) or (self.manifest_format == "jsonl" and isinstance(audio_path, list)):
            if self.code_type == "SNAC":
                audio_res, audio_length = get_snac_answer_token(audio_path)
            elif self.code_type == "CosyVoice":
                audio_tokens = audio_path
                if self.code_layer == 1:
                    audio_res, audio_length = get_single_layer_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa)
                else:
                    audio_res, audio_length = get_group_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa, self.code_layer)
            return audio_res, audio_length
        else:
            audio_raw = whisper.load_audio(audio_path)
            
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320
            audio_length = audio_length // 5
            audio_res = audio_raw
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2
            audio_length = audio_length // 5
            audio_res = audio_mel

        return audio_res, audio_length

    def upsample_tokens(self, tokens, upsampling_factor, method="repeat"):
        """
        Upsample the input tokens based on the given method and factor.
        
        Args:
            tokens: List[int], a list of token IDs to be upsampled.
            upsampling_factor: int, the factor by which the tokens will be upsampled.
            method: str, the upsampling method, either "repeat" or "blank".

        Returns:
            List[int], the upsampled token IDs.
        """
        if upsampling_factor <= 1:
            return tokens

        upsampled_tokens = []
        blank_token = self.tokenizer.pad_token_id  # Use pad token as the blank token

        if method == "repeat":
            # Repeat each token 'upsampling_factor' times
            for token in tokens:
                upsampled_tokens.extend([token] * upsampling_factor)

        elif method == "blank":
            # Add (upsampling_factor - 1) blank tokens after each token
            for token in tokens:
                upsampled_tokens.append(token)
                upsampled_tokens.extend([blank_token] * (upsampling_factor - 1))

        else:
            raise ValueError("Unsupported upsampling method. Choose 'repeat' or 'blank'.")

        return upsampled_tokens
    
    def __getitem__(self, index):
        data_dict = self.data_list[index] 
        data_struct = self.data_struct_list[index]
        task_type = self.task_type  
        audio_mel = None
        example_ids = None
        key = None
        audio_length = 0
        target_audio_length = 0

        if task_type != "s2s":
            raise ValueError(f"task_type {task_type} is not supported")

        ### get data
        if self.manifest_format == "jsonl":
            source_audio = data_dict.get("question_audio", None)
            target_audio = data_dict.get("answer_cosyvoice_speech_token", None)
            if data_struct[1] is None or data_struct[2] == 1.0:
                source_text = data_dict.get("question", None) 
                target_text = data_dict.get("answer", None) 
                think_text = data_dict.get("solution", None)
            elif data_struct[1] == "question":
                source_text = data_dict.get(f"question_compress_{data_struct[2]}", None)
                think_text = data_dict.get("solution", None)
                target_text = data_dict.get("answer", None)
            elif data_struct[1] == "question_solution":
                source_text = data_dict.get(f"question_compress_{data_struct[2]}", None)
                think_text = data_dict.get(f"solution_compress_{data_struct[2]}", None)
                target_text = data_dict.get("answer", None)   
            elif data_struct[1] == "solution":
                source_text = data_dict.get("question", None) 
                think_text = data_dict.get(f"solution_compress_{data_struct[2]}", None)
                target_text = data_dict.get("answer", None)   
            elif data_struct[1] == "answer":
                source_text = data_dict.get("question", None) 
                think_text = data_dict.get("solution", None)
                target_text = data_dict.get(f"answer_compress_{data_struct[2]}", None)
            else:
                raise NotImplementedError("compress_stage must be one of [question, answer]")
            key = data_dict.get("key", None)
        else:
            raise ValueError("manifest_format must be one of [jsonl]")

        audio_mel, audio_length = self.extract_audio_feature(source_audio) # audio_length = len(audio) 
        target_audio, target_audio_length = self.extract_audio_feature(target_audio) # target_audio_length = len(audio) + [eoa]
            
        prompt = self.prompt_template.format(data_struct[3])
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = [self._input_t] + prompt_ids + [self._eot]
        prompt_ids = torch.tensor(prompt_ids).unsqueeze(0)   # [ bacth_size, prompt_length ]

        example_ids = [self.layershift(self._input_a, 0)] + [self.layershift(self._pad_a, 0)] * audio_length + [(self.layershift(self._eoa, 0)), self.special_token_t]
        example_ids = torch.tensor(example_ids).unsqueeze(0)  # [ bacth_size, prompt_length ]

        example_ids = torch.cat((prompt_ids, example_ids), dim = 1)  

        prompt_length = prompt_ids.size(1)
        input_length  = audio_length 
        input_mask_length = input_length + prompt_length + 3 # NOTE: here 3 include <boa> <eoa> <ans_t>

        if self.inference_mode:
            example_mask = example_ids[0].ge(-1)  # [True,True]
            example_ids = example_ids.squeeze()

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel,
                "input_length": input_length,
                "audio_length": audio_length,
                "target_audio": target_audio,
                "target_audio_length": target_audio_length,
                "key": key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_length": prompt_length,
                "task_type": task_type,
            }

        # build MIX answer
        answer_text = self.answer_template.format(target_text)
        answer_text_ids = self.tokenizer.encode(answer_text)  # [answer]
        
        question_text = self.answer_template.format(source_text)
        question_text_ids = self.tokenizer.encode(question_text)  # [question]

        think_text = self.answer_template.format(think_text)
        think_text_ids = self.tokenizer.encode(think_text)  # [think]

        if self.upsample_text_tokens:
            answer_text_ids = self.upsample_tokens(answer_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)
            question_text_ids = self.upsample_tokens(question_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)
            think_text_ids = self.upsample_tokens(think_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)


        question_text_ids = [self._com_qt] + question_text_ids + [self._eot]
        question_text_ids = torch.tensor(question_text_ids, dtype=torch.int64)

        think_text_ids = [self._com_think] + think_text_ids + [self._eot] 
        think_text_ids = torch.tensor(think_text_ids, dtype=torch.int64)
        
        answer_text_ids = [self._com_at] + answer_text_ids + [self._eot, self._com_aat] 
        answer_text_ids = torch.tensor(answer_text_ids, dtype=torch.int64)


        if data_struct[0] == "com-reason" :
            answer_com_ids = torch.cat((question_text_ids.unsqueeze(0), think_text_ids.unsqueeze(0), answer_text_ids.unsqueeze(0)), dim=1)  
            com_text_length = len(question_text_ids) + len(think_text_ids) + len(answer_text_ids)
        elif data_struct[0] == "sats-reason":
            answer_com_ids = torch.cat((think_text_ids.unsqueeze(0), answer_text_ids.unsqueeze(0)), dim=1)  
            com_text_length = len(think_text_ids) + len(answer_text_ids)
        elif data_struct[0] == "sats-reason-is":
            max_index = len(self.data_list)
            # randomly perturb index: with prob 0.3, apply a ±10% offset
            perturbed_index = index
            if random.random() < 0.3:
                noise = random.uniform(-0.1, 0.1) * max_index
                perturbed_index = max(0, min(int(index + noise), max_index - 1))
            # truncate the front of think_text_ids by ratio perturbed_index / max_index
            truncate_ratio = perturbed_index / max_index
            think_text_ids = self.tokenizer.encode(think_text) 
            truncate_len = int(len(think_text_ids) * truncate_ratio)
            think_text_ids = think_text_ids[truncate_len:]
            think_text_ids = [self._com_think] + think_text_ids + [self._eot] 
            think_text_ids = torch.tensor(think_text_ids, dtype=torch.int64)
            answer_com_ids = torch.cat((think_text_ids.unsqueeze(0), answer_text_ids.unsqueeze(0)), dim=1)  
            com_text_length = len(think_text_ids) + len(answer_text_ids)
        elif data_struct[0] == "sats-reason-ht":
            think_text_ids = [self._com_qt , self._com_think]
            think_text_ids = torch.tensor(think_text_ids, dtype=torch.int64)
            answer_com_ids = torch.cat((think_text_ids.unsqueeze(0), answer_text_ids.unsqueeze(0)), dim=1)  
            com_text_length = len(think_text_ids) + len(answer_text_ids)
        elif data_struct[0] == "com":
            answer_com_ids = torch.cat((question_text_ids.unsqueeze(0), answer_text_ids.unsqueeze(0)), dim=1)  
            com_text_length = len(question_text_ids)  + len(answer_text_ids)
        elif data_struct[0] == "sats" or data_struct[0] == "s2s":
            answer_com_ids = answer_text_ids.unsqueeze(0) 
            com_text_length = len(answer_text_ids)
        elif data_struct[0] == "sds":
            answer_com_ids = torch.tensor([self._com_aat], dtype=torch.int64).unsqueeze(0)  
            com_text_length = len(answer_text_ids)
        else:
            raise ValueError("manifest_format must be one of them")

        com_length = com_text_length + target_audio_length 
        
        # build labels
        labels_ids = copy.deepcopy(answer_com_ids)
        ori_example_ids = copy.deepcopy(example_ids) # this is [ prompt, qusetion_audio ]
        
        # build audio 
        if target_audio is None:    
            raise ValueError("target_audio is None")

        assert input_mask_length == ori_example_ids.size(1)

        audio_labels_ids = []
        audio_input_ids = []
        for i in range(self.code_layer):
            audio_labels_ids.append(torch.cat((torch.tensor([ -1 ] * ( input_mask_length + com_text_length)).unsqueeze(0), target_audio[i].unsqueeze(0)), dim=1)) 
            audio_input_ids.append(torch.cat((torch.tensor([ self.layershift(self._pad_a, i) ] * ( input_mask_length + com_text_length)).unsqueeze(0), self.layershift(target_audio[i], i).unsqueeze(0)), dim=1))

        text_labels_ids = torch.cat((torch.tensor([ -1 ] * input_mask_length).unsqueeze(0), labels_ids, torch.tensor([ -1 ] * target_audio_length).unsqueeze(0)), dim=1)

        shifted_audios = []
        for i in range(self.code_layer):
            # shift the i-th audio layer
            # shifted_audios.append(self.layershift(target_audio[i], i))

            ### fix aduio information input
            shifted_audios.append(self.layershift(torch.tensor([ self._pad_a ] * target_audio_length), i))

        # averaged_audio_ids = torch.mean(shifted_audios.float(), dim=0).unsqueeze(0)  # averaging here may be problematic
        example_ids = torch.cat((ori_example_ids, answer_com_ids, shifted_audios[0].unsqueeze(0)), dim=1)  # [prompt, question_audio, answer_com, eos] 
        labels_ids  = copy.deepcopy(example_ids)

        # batch size
        if not isinstance(example_ids, list):
            example_ids = [example_ids]
        if not isinstance(labels_ids, list):
            labels_ids = [labels_ids]
        example_ids = torch.stack(example_ids).squeeze()
        labels_ids = torch.stack(labels_ids).squeeze()

        audio_input_ids = torch.stack(audio_input_ids).squeeze()

        text_labels_ids = text_labels_ids.squeeze()
        audio_labels_ids= torch.stack(audio_labels_ids).squeeze()

        # mask prompt, question_audio
        labels_ids[:input_mask_length] = -1  

        # mask proccessing
        example_mask = example_ids.ge(-1)  # mask to bool

        label_mask = labels_ids.ge(0)  # init_interm to bool
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer_com,eos,-100]

        text_label_mask = text_labels_ids.ge(0)
        text_labels_ids[~text_label_mask] = self.IGNORE_INDEX

        audio_label_mask = audio_labels_ids.ge(0)
        audio_labels_ids[~audio_label_mask] = self.IGNORE_INDEX

        assert example_ids.shape == text_labels_ids.shape

        data_dict =  {
            "input_ids": example_ids, # with layershif
            "labels": labels_ids, # without layershif
            "attention_mask": example_mask, 
            "audio_mel": audio_mel,   # question_audio
            "input_length": input_length, # question_audio_length
            "audio_length": audio_length, # question_audio_length
            "target_audio": target_audio, # answer_audio
            "target_audio_length": target_audio_length, # answer_audio_length
            "key": key,
            "source_text": source_text, 
            "target_text": target_text,
            "prompt_length": prompt_length,
            "task_type": task_type,
            "text_labels": text_labels_ids,
            "audio_labels": audio_labels_ids,
            "audio_input_ids": audio_input_ids,
        }

        # print_shapes_or_lengths(data_dict)
        return data_dict

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    # sequence = torch.nn.functional.pad(sequence, (0, padding_length)) FIXME: this is wrong before in SLAM-LLM
                    padding_tensor = torch.full((sequence.size(0), padding_length), padding_idx, dtype=sequence.dtype)
                    if padding_side == "left":
                        sequence = torch.cat((padding_tensor, sequence), dim=1)
                    else:
                        sequence = torch.cat((sequence, padding_tensor), dim=1)
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["input_length"] + s['prompt_length'] + 3 for s in samples] #[321, 321, 321, 321]
        input_answer_lengths = [len(s["input_ids"]) - s["input_length"] - s['prompt_length'] - 3 for s in samples]  #[264, 99, 206, 141]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        # NOTE: left padding for prompt and right padding for answer 
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        input_length = torch.tensor([s["input_length"] for s in samples])
        audio_length = torch.tensor([s["audio_length"] for s in samples])
        audio_raw = None
        audio_mask = None
        audio_mel = None
        audio_mel_post_mask = None

        if self.input_type == "raw" and self.task_type in ["s2s", "asr"]:
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel" and self.task_type in ["s2s", "asr"]:
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index] + 1 + samples[index]['prompt_length'] # +1 for <bos>
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        task_type = [s['task_type'] for s in samples]

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            target_text = [s['target_text'] for s in samples]
            source_text = [s['source_text'] for s in samples]
            target_audio = [s['target_audio'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "target_texts": target_text,
                "source_texts": source_text,
                "target_audio": target_audio,
                "task_types": task_type
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX
                )
            for index in range(len(samples))
        ])


        # ====== fix text_labels (in case it is also 1D) ======
        text_labels = torch.stack([
            self.padding(
                self.padding(samples[index]['text_labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX
                )
            for index in range(len(samples))
        ])

        # ====== fix audio_labels: use its own length ======
        audio_labels = torch.stack([
            self.padding(
                self.padding(samples[index]['audio_labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side= "left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX
            )
            for index in range(len(samples))
        ])

        audio_input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["audio_input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])
                
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask,
            "task_types": task_type,
            "text_labels": text_labels,
            "audio_labels": audio_labels,
            "audio_input_ids": audio_input_ids
        }


class SpeechDatasetJsonl_Inter(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "<SYSTEM>: {}\n "
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        self.manifest_format = dataset_config.get("manifest_format", "parquet")
        self.seed = dataset_config.get("seed", 42)
        self.rng = random.Random(self.seed)
        self.split_size = dataset_config.get("split_size", 0.1)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        assert self.manifest_format in ["parquet", "jsonl"], "manifest_format must be one of [parquet, jsonl]"

        # vocab config
        self.vocab_config = dataset_config.get("vocab_config", None)
        self.text_vocabsize = self.vocab_config.text_vocabsize
        self.text_specialtokens = self.vocab_config.text_specialtokens
        self.audio_vocabsize = self.vocab_config.audio_vocabsize
        self.audio_specialtokens = self.vocab_config.audio_specialtokens
        self.padded_text_vocabsize = self.vocab_config.padded_text_vocabsize
        self.padded_audio_vocabsize = self.vocab_config.padded_audio_vocabsize
        self.total_vocabsize = self.vocab_config.total_vocabsize
        self._eot = self.vocab_config.eot
        self._pad_t = self.vocab_config.pad_t
        self._input_t = self.vocab_config.input_t
        self._answer_t = self.vocab_config.answer_t
        self._asr = self.vocab_config.asr
        self._eoa = self.vocab_config.eoa
        self._pad_a = self.vocab_config.pad_a
        self._input_a = self.vocab_config.input_a
        self._answer_a = self.vocab_config.answer_a
        self._split = self.vocab_config.split
        self.code_layer = self.vocab_config.code_layer
        
        self._com_qt = self.vocab_config.com_qt
        self._com_at = self.vocab_config.com_at
        self._com_think = self.vocab_config.com_think
        self._com_aat= self.vocab_config.com_aat

        self.special_token_a = self._answer_a
        self.special_token_t = self._answer_t

        # task config 
        self.task_type = dataset_config.get("task_type", "s2s")

        # upsample config 
        self.upsample_text_tokens = dataset_config.get("upsample_text_tokens", False)
        self.upsampling_factor = dataset_config.get("upsampling_factor", 1)
        self.upsample_method = dataset_config.get("upsample_method", "repeat")

        # code type config
        self.code_type = dataset_config.get("code_type", "SNAC")
        if self.code_type != "SNAC" and self.code_type != "CosyVoice":
            raise ValueError("code_type must be one of [SNAC, CosyVoice]")
        
        # number of tokens for latency
        self.num_latency_tokens = dataset_config.get("num_latency_tokens", 1)

        # layershift config
        self.do_layershift = dataset_config.get("do_layershift", True)
        if self.do_layershift:
            self.layershift = layershift
        else:
            self.layershift = simple_shift

        # read dataset
        if split == "train":
            list_json_path = dataset_config.train_data_path
        elif split in ("val", "valid", "validation"):
            list_json_path = getattr(dataset_config, "val_data_path", None)
            if list_json_path is None:
                raise ValueError("dataset_config.val_data_path not found")
        elif split == "test":
            list_json_path = getattr(dataset_config, "test_data_path", None)
            if list_json_path is None:
                raise ValueError("dataset_config.test_data_path not found")
        else:
            raise ValueError(f"Unknown split: {split}")   
               

        self.cfg = load_config_from_yaml(list_json_path)
        self.data_list=[]
        self.data_struct_list=[]
        for ds in self.cfg.datasets:
            rate = float(ds.sampling_rate)
            with open(ds.dataset_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    # sample
                    if random.random() >= rate:
                        continue

                    data_dict = json.loads(line)
                    self.data_list.append(data_dict)
                    self.data_struct_list.append((ds.data_struct, ds.compress.stage, ds.compress.rate, ds.system_prompt))
        
        mix_flag = True
        if mix_flag: # sequential or mixed training
            rng = random.Random(self.seed)
            idx = list(range(len(self.data_list)))
            rng.shuffle(idx)
            self.data_list = [self.data_list[i] for i in idx]
            self.data_struct_list = [self.data_struct_list[i] for i in idx]

    def interleave_text_speech(self, text_ids, speech_ids, text_chunk=13, speech_chunk=26,
                           pad_text=-1, pad_speech=None,  # pad_speech: the layer-shifted pad_a
                           use_real_speech=True):
        """
        text_ids: 1D LongTensor or list[int]
        speech_ids: 1D LongTensor or list[int]  (one layer's speech sequence; may already include eoa, etc.)
        Returns: interleaved_ids(list[int]), text_pos_mask(list[bool]), speech_pos_mask(list[bool])
        """
        if torch.is_tensor(text_ids):
            text_ids = text_ids.tolist()
        if torch.is_tensor(speech_ids):
            speech_ids = speech_ids.tolist()

        out = []
        is_text = []
        is_speech = []

        ti = 0
        si = 0
        while ti < len(text_ids) or si < len(speech_ids):
            # text chunk
            if ti < len(text_ids):
                tseg = text_ids[ti:ti+text_chunk]
                out.extend(tseg)
                is_text.extend([True]*len(tseg))
                is_speech.extend([False]*len(tseg))
                ti += len(tseg)

            # speech chunk
            if si < len(speech_ids):
                if use_real_speech:
                    aseg = speech_ids[si:si+speech_chunk]
                    si += len(aseg)
                else:
                    # insert only placeholder speech (fixed length speech_chunk)
                    aseg = [pad_speech] * speech_chunk
                out.extend(aseg)
                is_text.extend([False]*len(aseg))
                is_speech.extend([True]*len(aseg))

            # if real tokens are fewer than speech_chunk it will be shorter; TODO: padding strategy
        return out, is_text, is_speech

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)

    def extract_audio_feature(self, audio_path):
        # audio path is a dictionary, resample the audio to 16kHz
        if self.manifest_format == "parquet" and isinstance(audio_path, dict):
            audio_raw = audio_path['array']
            audio_raw_sr = audio_path['sampling_rate']
            if not isinstance(audio_raw, np.ndarray):
                audio_raw = np.array(audio_raw)
            audio_raw = librosa.resample(audio_raw, orig_sr=audio_raw_sr, target_sr=16000).astype(np.float32)
        elif (self.manifest_format == "parquet" and (isinstance(audio_path, str) or isinstance(audio_path, list))) or (self.manifest_format == "jsonl" and isinstance(audio_path, list)):
            if self.code_type == "SNAC":
                audio_res, audio_length = get_snac_answer_token(audio_path)
            elif self.code_type == "CosyVoice":
                audio_tokens = audio_path
                if self.code_layer == 1:
                    audio_res, audio_length = get_single_layer_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa)
                else:
                    audio_res, audio_length = get_group_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa, self.code_layer)
            return audio_res, audio_length
        else:
            audio_raw = whisper.load_audio(audio_path)
            
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320
            audio_length = audio_length // 5
            audio_res = audio_raw
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2
            audio_length = audio_length // 5
            audio_res = audio_mel

        return audio_res, audio_length

    def upsample_tokens(self, tokens, upsampling_factor, method="repeat"):
        """
        Upsample the input tokens based on the given method and factor.
        
        Args:
            tokens: List[int], a list of token IDs to be upsampled.
            upsampling_factor: int, the factor by which the tokens will be upsampled.
            method: str, the upsampling method, either "repeat" or "blank".

        Returns:
            List[int], the upsampled token IDs.
        """
        if upsampling_factor <= 1:
            return tokens

        upsampled_tokens = []
        blank_token = self.tokenizer.pad_token_id  # Use pad token as the blank token

        if method == "repeat":
            # Repeat each token 'upsampling_factor' times
            for token in tokens:
                upsampled_tokens.extend([token] * upsampling_factor)

        elif method == "blank":
            # Add (upsampling_factor - 1) blank tokens after each token
            for token in tokens:
                upsampled_tokens.append(token)
                upsampled_tokens.extend([blank_token] * (upsampling_factor - 1))

        else:
            raise ValueError("Unsupported upsampling method. Choose 'repeat' or 'blank'.")

        return upsampled_tokens
    
    def __getitem__(self, index): 
        data_dict = self.data_list[index] 
        data_struct = self.data_struct_list[index]
        task_type = self.task_type  
        audio_mel = None
        example_ids = None
        key = None
        audio_length = 0
        target_audio_length = 0

        if task_type != "s2s":
            raise ValueError(f"task_type {task_type} is not supported")

        ### get data
        if self.manifest_format == "jsonl":
            source_audio = data_dict.get("question_audio", None)
            target_audio = data_dict.get("answer_cosyvoice_speech_token", None)
            if data_struct[1] is None or data_struct[2] == 1.0:
                source_text = data_dict.get("question", None) 
                target_text = data_dict.get("answer", None) 
                think_text = data_dict.get("solution", None)
            elif data_struct[1] == "question":
                source_text = data_dict.get(f"question_compress_{data_struct[2]}", None)
                think_text = data_dict.get("solution", None)
                target_text = data_dict.get("answer", None)
            elif data_struct[1] == "solution":
                source_text = ""
                think_text = data_dict.get(f"solution_compress_{data_struct[2]}", None)
                target_text = data_dict.get("answer", None)   
            elif data_struct[1] == "answer":
                source_text = ""
                think_text = ""
                target_text = data_dict.get(f"answer_compress_{data_struct[2]}", None)
            else:
                raise NotImplementedError("compress_stage must be one of [question, answer]")
            key = data_dict.get("key", None)
        else:
            raise ValueError("manifest_format must be one of [jsonl]")

        audio_mel, audio_length = self.extract_audio_feature(source_audio) # audio_length = len(audio) 
        target_audio, target_audio_length = self.extract_audio_feature(target_audio) # target_audio_length = len(audio) + [eoa]

        if len(target_audio) != self.code_layer:
            raise ValueError(f"{index} target_audio layer mismatch: expect {self.code_layer}, got {len(target_audio)}")

            
        prompt = self.prompt_template.format(data_struct[3])
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = [self._input_t] + prompt_ids + [self._eot]
        prompt_ids = torch.tensor(prompt_ids).unsqueeze(0)   # [ bacth_size, prompt_length ]

        example_ids = [self.layershift(self._input_a, 0)] + [self.layershift(self._pad_a, 0)] * audio_length + [(self.layershift(self._eoa, 0)), self.special_token_t]
        example_ids = torch.tensor(example_ids).unsqueeze(0)  # [ bacth_size, prompt_length ]

        example_ids = torch.cat((prompt_ids, example_ids), dim = 1)  

        prompt_length = prompt_ids.size(1)
        input_length  = audio_length 
        input_mask_length = input_length + prompt_length + 3 # NOTE: here 3 include <boa> <eoa> <ans_t>

        if self.inference_mode:
            example_mask = example_ids[0].ge(-1)  # [True,True]
            example_ids = example_ids.squeeze()

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel,
                "input_length": input_length,
                "audio_length": audio_length,
                "target_audio": target_audio,
                "target_audio_length": target_audio_length,
                "key": key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_length": prompt_length,
                "task_type": task_type,
            }

        # build 
        answer_text = self.answer_template.format(target_text)
        answer_text_ids = self.tokenizer.encode(answer_text)  # [answer]
        
        question_text = self.answer_template.format(source_text)
        question_text_ids = self.tokenizer.encode(question_text)  # [question]

        think_text = self.answer_template.format(think_text)
        think_text_ids = self.tokenizer.encode(think_text)  # [think]

        if self.upsample_text_tokens:
            answer_text_ids = self.upsample_tokens(answer_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)
            question_text_ids = self.upsample_tokens(question_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)
            think_text_ids = self.upsample_tokens(think_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)


        question_text_ids = [self._com_qt] + question_text_ids + [self._eot]
        question_text_ids = torch.tensor(question_text_ids, dtype=torch.int64)

        think_text_ids = [self._com_think] + think_text_ids + [self._eot] 
        think_text_ids = torch.tensor(think_text_ids, dtype=torch.int64)
        
        answer_text_ids = [self._com_at] + answer_text_ids + [self._eot, self._com_aat] 
        answer_text_ids = torch.tensor(answer_text_ids, dtype=torch.int64)


        ### bulid interleave

        if data_struct[0] == "sats-inter": 
            TEXT_CHUNK = 13
            SPEECH_CHUNK = 26
            USE_REAL_SPEECH = True  # if False, must explicitly control how many speech chunks to insert (not via si)
            
            # input-side speech pad (used when USE_REAL_SPEECH=False)
            speech0_pad_shifted = -1
            
            # take layer-0 real speech, just as a placeholder
            speech0 = torch.tensor([speech0_pad_shifted] * target_audio_length)  # 1D tensor, length = target_audio_length

            # interleave
            interleaved_ids, is_text_pos, is_speech_pos = self.interleave_text_speech(
                answer_text_ids,
                speech0,
                text_chunk=TEXT_CHUNK,
                speech_chunk=SPEECH_CHUNK,
                pad_speech=speech0_pad_shifted,
                use_real_speech=USE_REAL_SPEECH
            )

        else:
            raise ValueError("manifest_format must be one of them")
        
        
        # build labels
        interleaved_ids = torch.tensor(interleaved_ids, dtype=torch.long).unsqueeze(0)
        labels_ids = copy.deepcopy(interleaved_ids)
        ori_example_ids = copy.deepcopy(example_ids) # this is [ prompt, qusetion_audio ]
        
        # build audio: audio is left unchanged to save pad-embedding compute; text format is the final format
        if target_audio is None:    
            raise ValueError("target_audio is None")

        assert input_mask_length == ori_example_ids.size(1)

        audio_labels_ids = []
        audio_input_ids = []
        for i in range(self.code_layer):
            audio_labels_ids.append(target_audio[i].unsqueeze(0))
            audio_input_ids.append(self.layershift(target_audio[i], i).unsqueeze(0))

        text_labels_ids = torch.cat((torch.tensor([ -1 ] * input_mask_length).unsqueeze(0), interleaved_ids), dim=1) 

        example_ids = torch.cat((ori_example_ids, interleaved_ids), dim=1)  # [prompt, question_audio, answer_interleave, eos] 
        labels_ids  = copy.deepcopy(example_ids)
        
        is_text_pos = torch.tensor(is_text_pos, dtype=torch.bool)
        is_speech_pos = torch.tensor(is_speech_pos, dtype=torch.bool)
        pad = torch.zeros(input_mask_length, dtype=torch.bool)
        is_text_pos = torch.cat((pad, is_text_pos), dim=0)
        is_speech_pos = torch.cat((pad, is_speech_pos), dim=0)

        # batch size
        if not isinstance(example_ids, list):
            example_ids = [example_ids]
        if not isinstance(labels_ids, list):
            labels_ids = [labels_ids]
        example_ids = torch.stack(example_ids).squeeze()
        labels_ids = torch.stack(labels_ids).squeeze()
        
        text_labels_ids = text_labels_ids.squeeze()

        audio_input_ids = torch.stack(audio_input_ids, dim=0).squeeze(1)
        audio_labels_ids = torch.stack(audio_labels_ids, dim=0).squeeze(1)

        # mask prompt, question_audio
        labels_ids[:input_mask_length] = -1  

        # mask proccessing
        example_mask = example_ids.ge(-1)  # mask to bool

        label_mask = labels_ids.ge(0)  # init_interm to bool
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer_com,eos,-100]

        text_label_mask = text_labels_ids.ge(0)
        text_labels_ids[~text_label_mask] = self.IGNORE_INDEX

        audio_label_mask = audio_labels_ids.ge(0)
        audio_labels_ids[~audio_label_mask] = self.IGNORE_INDEX



        data_dict =  {
            "input_ids": example_ids, # with layershif
            "labels": labels_ids, # without layershif
            "attention_mask": example_mask, 
            "audio_mel": audio_mel,   # question_audio
            "input_length": input_length, # question_audio_length
            "audio_length": audio_length, # question_audio_length
            "target_audio": target_audio, # answer_audio
            "target_audio_length": target_audio_length, # answer_audio_length
            "key": key,
            "source_text": source_text, 
            "target_text": target_text,
            "prompt_length": prompt_length,
            "task_type": task_type,
            "text_labels": text_labels_ids,
            "audio_labels": audio_labels_ids,
            "audio_input_ids": audio_input_ids,
            "is_text_pos": is_text_pos,
            "is_speech_pos": is_speech_pos,
        }

        # print_shapes_or_lengths(data_dict)
        return data_dict

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    # sequence = torch.nn.functional.pad(sequence, (0, padding_length)) FIXME: this is wrong before in SLAM-LLM
                    padding_tensor = torch.full((sequence.size(0), padding_length), padding_idx, dtype=sequence.dtype)
                    if padding_side == "left":
                        sequence = torch.cat((padding_tensor, sequence), dim=1)
                    else:
                        sequence = torch.cat((sequence, padding_tensor), dim=1)
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["input_length"] + s['prompt_length'] + 3 for s in samples] #[321, 321, 321, 321]
        input_answer_lengths = [len(s["input_ids"]) - s["input_length"] - s['prompt_length'] - 3 for s in samples]  #[264, 99, 206, 141]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        audio_label_max_length = max(s["audio_labels"].shape[-1] for s in samples)
        
        # NOTE: left padding for prompt and right padding for answer 
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        input_length = torch.tensor([s["input_length"] for s in samples])
        audio_length = torch.tensor([s["audio_length"] for s in samples])
        audio_raw = None
        audio_mask = None
        audio_mel = None
        audio_mel_post_mask = None

        if self.input_type == "raw" and self.task_type in ["s2s", "asr"]:
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel" and self.task_type in ["s2s", "asr"]:
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index] + 1 + samples[index]['prompt_length'] # +1 for <bos>
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        task_type = [s['task_type'] for s in samples]

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            target_text = [s['target_text'] for s in samples]
            source_text = [s['source_text'] for s in samples]
            target_audio = [s['target_audio'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "target_texts": target_text,
                "source_texts": source_text,
                "target_audio": target_audio,
                "task_types": task_type
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX
                )
            for index in range(len(samples))
        ])


        # ====== fix text_labels (in case it is also 1D) ======
        text_labels = torch.stack([
            self.padding(
                self.padding(samples[index]['text_labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX
                )
            for index in range(len(samples))
        ])

        # ====== fix audio_labels: use its own length ======
        audio_labels = torch.stack([
            self.padding(
                samples[index]["audio_labels"],
                audio_label_max_length - samples[index]["audio_labels"].shape[-1],
                self.IGNORE_INDEX
            )
            for index in range(len(samples))
        ])

        audio_input_ids = torch.stack([
            self.padding(
                samples[index]["audio_input_ids"],
                audio_label_max_length - samples[index]["audio_input_ids"].shape[-1],
                self.tokenizer.pad_token_id
            )
            for index in range(len(samples))
        ])

        # ======= is_text_pos, is_speech_pos ======
        is_text_pos = torch.stack([
            self.padding(
                self.padding(
                    torch.as_tensor(samples[index]["is_text_pos"], dtype=torch.bool),
                    input_prompt_max_length - input_prompt_lengths[index],
                    False,
                    padding_side="left"
                ),
                input_answer_max_length - input_answer_lengths[index],
                False
            )
            for index in range(len(samples))
        ])

        is_speech_pos = torch.stack([
            self.padding(
                self.padding(
                    torch.as_tensor(samples[index]["is_speech_pos"], dtype=torch.bool),
                    input_prompt_max_length - input_prompt_lengths[index],
                    False,
                    padding_side="left"
                ),
                input_answer_max_length - input_answer_lengths[index],
                False
            )
            for index in range(len(samples))
        ])

                
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask,
            "task_types": task_type,
            "text_labels": text_labels,
            "audio_labels": audio_labels,
            "audio_input_ids": audio_input_ids,
            "is_text_pos": is_text_pos,
            "is_speech_pos": is_speech_pos,
        }



def get_speech_dataset(dataset_config, tokenizer, split):
    if dataset_config.get("dataset_version", None) == "mix-com":
        dataset = SpeechDatasetJsonl_CoM(dataset_config, tokenizer, split)
    elif dataset_config.get("dataset_version", None) == "mix-inter":
        dataset = SpeechDatasetJsonl_Inter(dataset_config, tokenizer, split)
    elif dataset_config.get("dataset_version", None) == "mix-paral":
        raise ValueError("refer to SLAM-Omni code.")
    else:
        raise ValueError("Unsupported dataset version!")
    return dataset
