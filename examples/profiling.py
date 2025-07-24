#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention.flex_attention import create_block_mask

import logging
import math
import os
import sys
import warnings
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset

import transformers
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import Qwen3ForCausalLM
import json
import numpy as np
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl
from cut_cross_entropy import linear_cross_entropy

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
create_block_mask = torch.compile(create_block_mask, dynamic = False)

logger = logging.getLogger(__name__)

DEVICES = {
    'NVIDIA GeForce RTX 3090': 142,
    'NVIDIA GeForce RTX 3090 Ti': 160,
    'NVIDIA H100 80GB HBM3': 990,
    'NVIDIA GH200 480GB': 990,
}

class WandbMFUCallback(TrainerCallback):
    def __init__(self, device_name = None):
        self._time_of_second_step: Optional[float] = None
        self._flops_at_second_step: Optional[float] = None
        self._time_for_train_steps = 0.0
        self._first_step_finished = False
        self._device_name = device_name
        if self._device_name is None:
            self._device_name = torch.cuda.get_device_name()
        if self._device_name not in DEVICES:
            raise Exception('device name is not recognized.')

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start_time = time.time()
        if not self._first_step_finished:
            return

        if self._time_of_second_step is None:
            self._time_of_second_step = self._step_start_time
            if state is not None:
                self._flops_at_second_step = state.total_flos
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        delta_time_seconds = time.time() - self._step_start_time
        if not self._first_step_finished:
            self._first_step_finished = True
            return
        self._time_for_train_steps += delta_time_seconds

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if self._time_of_second_step is None:
            return
        
        delta_time_seconds_train = time.time() - self._time_of_second_step
        delta_time_seconds_step = self._time_for_train_steps
        
        if self._flops_at_second_step is not None and (
            state is not None and state.total_flos > 0.0
        ):
            flops_since_second_step_on_all_devices = (
                state.total_flos - self._flops_at_second_step
            )
            flops_step = flops_since_second_step_on_all_devices / delta_time_seconds_step
            flops_train = flops_since_second_step_on_all_devices / delta_time_seconds_train
            device_flops_per_second = DEVICES[self._device_name] * 1e12
            train_step_mfu = flops_step / device_flops_per_second
            train_mfu = flops_train / device_flops_per_second

            wandb.log({
                "train_step_mfu": train_step_mfu,
                "train_mfu": train_mfu,
            }, step=state.global_step)

class ProfilerCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.prof.step()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "flash_attention_2",
                "flex_attention",
                "sdpa"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    block_size: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    offsets = offsets[offsets != -1]
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )

def length_to_offsets(lengths, device):
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def generate_doc_mask_mod(offsets):
    
    offsets = pad_sequence(offsets, batch_first = True, padding_value = -1)
    docs = [_offsets_to_doc_ids_tensor(offsets[i]) for i in range(offsets.shape[0])]
    docs = torch.stack(docs, 0)
    
    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[b, q_idx] == docs[b, kv_idx]
        return causal_mask & document_mask
    
    return document_causal_mask

def generate_list_sum_n(n, length=5, min_val=5):

    numbers = [min_val] * length
    remaining = n - min_val * length

    for _ in range(remaining):
        numbers[random.randint(0, length - 1)] += 1

    random.shuffle(numbers)
    return numbers

def pad_attention_mask_3d(attention_mask, max_size = 4096, value = 0.0):
    maxlen = attention_mask.shape[-1]
    return F.pad(
        attention_mask,
        (0, max_size - maxlen, 0, max_size - maxlen),
        value = value,
    )

def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self, 
        input_ids, 
        attention_mask = None, 
        position_ids = None, 
        labels = None, 
        **kwargs,
    ):
        if self.config._attn_implementation == 'flex_attention':
            attention_mask = kwargs.pop('cu_seq_lens_q')
            kwargs.pop('cu_seq_lens_k')
            kwargs.pop('max_length_q')
            kwargs.pop('max_length_k')
            seq_len = position_ids.shape[-1]
            device = position_ids.device
            document_causal_mask = generate_doc_mask_mod(attention_mask[None])
            attention_mask = create_block_mask(
                document_causal_mask, None, None, seq_len, seq_len, device, _compile = True)
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            auto_shift_loss = linear_cross_entropy(
                embeddings.to(torch.bfloat16), 
                self.lm_head.weight.to(torch.bfloat16), 
                labels, 
                shift=True,
                impl="cce_kahan_full_c"
            )
            return {'loss': auto_shift_loss}
        return super_out

def flash_attention_2_collator(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [b['input_ids'] for b in batch]
    position_ids = [b['position_ids'] for b in batch]
    labels = [b['input_ids'].copy() for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    input_ids = np.concatenate(input_ids)
    position_ids = np.concatenate(position_ids)
    labels = np.concatenate(labels)
    query_lens = np.concatenate(attention_mask)
    cumsum = [0] + np.cumsum(query_lens).tolist()
    max_cumsum = int(np.max(cumsum))
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
    max_seqlen_q = np.max(query_lens)
    return {
        'input_ids': torch.tensor(input_ids)[None],
        'position_ids': torch.tensor(position_ids)[None],
        'labels': torch.tensor(labels)[None],
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_k,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level
        # at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, attn_implementation):
            self.attn_implementation = attn_implementation

        def __getitem__(self, idx):

            data = {
                'input_ids': np.random.randint(0, high=100000, size=(sequence_length,), dtype=np.int64),
                'attention_mask': np.array(generate_list_sum_n(sequence_length, length=20, min_val=10), dtype=np.int64),
                'position_ids': np.arange(sequence_length, dtype=np.int64)
            }

            if self.attn_implementation in {'flash_attention_2', 'flex_attention'}:
                return data

            data['labels'] = data["input_ids"].copy()

            masking = data.pop('attention_mask')
            masks = []
            for m in masking:
                masks.append(torch.tril(torch.ones(m, m)))
            attention_mask = block_diagonal_concat_inverted(*masks)
            data['attention_mask'] = pad_attention_mask_3d(
                attention_mask, sequence_length, min_dtype)
        
            return data

        def __len__(self):
            return 10000
    
    if model_args.attn_implementation == 'sdpa':
        data_collator = default_data_collator
    else:
        data_collator = flash_attention_2_collator
    
    dataset = DatasetFixed(model_args.attn_implementation)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype
    )
    
    with torch.profiler.profile(with_flops = True, record_shapes = True) as prof:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
            callbacks=[ProfilerCallback(prof=prof)]
        )
        trainer.train()
        
    prof.export_chrome_trace(f"trace-{model_args.attn_implementation}.json")

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
