# LLaMA-3B Edge Acceleration

## Abstract
This repository contains a complete reproducible pipeline for:
* Loading and quantizing LLAMA3-3B (Instruct) using the HQQ quantization framework
* Applying DoRA adapters for efficient fine-tuning
* Training the adapted model on the WikiText-2 dataset
* Merging the fine-tuned adapter with the base model, saving the merged FP16 weights
* Uploading  the merged model  to the Hugging Face Hub
* Downloading the merged model, applying quantization locally, and benchmarking inference performance


## The training and fine-tuning pipeline
This project is structured into two main code parts and one utility module.

**Custom Utilities**

A custom module, `hqq_utils.py`, provides the necessary utilities to enable model quantization, patching, and device mapping using the HQQ framework.

**Part 1: Training, DoRA integration, and uploading**

In this part, we load the LLAMA3-3B (Instruct) model, apply 4-bit HQQ quantization, integrate DoRA adapters using PEFT, fine-tune the model on the WikiText-2 dataset, and finally merge the adapter weights back into the base model. We then upload both the merged model and the tokenizer to the Hugging Face Hub. You can implement it from `Training_Dora.py`

**Part 2: Downloading, re-quantizing, and inference benchmarking**

This section covers downloading the merged model from the Hugging Face Hub, applying local HQQ quantization with specified parameters, and running inference benchmarks, including throughput measurement and perplexity evaluation.You can implement it from `Inference.py`

The following are the step-by-step processes of how we build and deploy the project.

## Requirements Set up
Set up environment using the following:
***requirements.txt***
```
torch
torchvision
torchaudio
transformers
huggingface-hub[cli]
datasets==3.5.0
timm==1.0.15
accelerate==1.6.0
bitsandbytes>=0.43.0
peft
tqdm
gemlite==0.4.4
hqq==0.2.5
triton==3.2.0
```

Install all requirements:
`pip install -r requirements.txt`

## Custom Utilities

This file defines:
* Model size calculation (get_size_of_model)
* Layer tagging utilities (get_linear_tags_from_model)
* Custom patching classes (CustomPatch)
* Quantization management classes (CustomHQQHFModel, AutoHQQHFModel)
* Device-aware linear layer replacement with HQQLinear

These components ensure that Linear layers inside Hugging Face transformer models can be efficiently replaced with quantized counterparts. 
Below is the full source code used in this project.
***hqq_utils.py***
```
import torch
import torch.nn as nn
from torch import float16
from tqdm.auto import tqdm

from typing import Union, Callable
from functools import partial
import json
import timm
import os

from hqq.core.quantize import HQQLinear
from hqq.core.utils import cleanup

from hqq.models.base import (
    forward_device_hooked,
    get_all_children_from_model,
    find_parent,
    is_leaf_module,
    BaseHQQModel,
    BasePatch
)

from hqq.models.hf.base import BaseHQQHFModel

_QUANT_LAYERS = [nn.Linear, HQQLinear]
_IGNORE_LINEAR = ['lm_head']

def get_size_of_model(model):
    size_in_bytes = 0
    for _, module in model.named_modules():
        if isinstance(module, HQQLinear):
            # W_q / Scale / Zero / Bias
            size_in_bytes += module.W_q.numel() * module.W_q.element_size()
            size_in_bytes += module.meta['scale'].numel() * module.meta['scale'].element_size()
            size_in_bytes += module.meta['zero'].numel() * module.meta['zero'].element_size()

            if isinstance(getattr(module, 'bias'), torch.Tensor):
                size_in_bytes += module.bias.numel() * module.bias.element_size()

        elif is_leaf_module(module):
            for param in module.parameters():
                size_in_bytes += param.numel() * param.element_size()
            for buffer in module.buffers():
                size_in_bytes += buffer.numel() * buffer.element_size()

    return size_in_bytes

# Get all linear tags available
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name)
    return list(linear_tags)

class CustomPatch(BasePatch):
    # This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name], patch_param),
            )

        cleanup()

        # These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod
    def set_auto_linear_tags(cls, model, ignore: list = _IGNORE_LINEAR) -> None:
        if hasattr(model, "linear_tags") is False:
            linear_tags = cls.get_linear_tags()
            model.linear_tags = (
                linear_tags
                if len(linear_tags) > 0
                else get_linear_tags_from_model(model, ignore=ignore)
            )
            model.base_class = cls

class CustomHQQTimmModel(BaseHQQModel):
    # Create empty model
    @classmethod
    def create_model(cls, save_dir, kwargs):
        with open(cls.get_config_file(save_dir), "r") as file:
            config = json.load(file)
        model = timm.create_model(
            config["architecture"] + "." + config["tag"], pretrained=False
        )
        return model

    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as error:
            print(error)

        with open(cls.get_config_file(save_dir), "w") as file:
            json.dump(model.default_cfg, file)

    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
    ):
        # Check if the model was already quantized
        if getattr(model, "hqq_quantized", False):
            print("Model was already quantized")
            return

        # Set linear tags automatically
        cls.setup_model(model)

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        elif quant_config == {}:
            patch_params = {key: None for key in model.linear_tags}
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.blocks)   # TODO: Modify layers to blocks
                if hasattr(model, "model")
                else len(model.blocks)
            )
            all_blocks = ["blocks." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".blocks" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".blocks" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # print(device_map)

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is HQQLinear:
                return linear_layer

            current_device = device_map[linear_layer.name]
            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "blocks") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.blocks)):
                core_model.blocks[i].device = device_map[core_model.blocks[i].name]
                core_model.blocks[i].forward_orig = core_model.blocks[i].forward
                core_model.blocks[i].forward = partial(
                    forward_device_hooked, core_model.blocks[i]
                )

        # Set base class
        model.base_class = cls

        model.hqq_quantized = True

        return model

class CustomHQQHFModel(BaseHQQHFModel):
    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
    ):
        # Check if the model was already quantized
        if getattr(model, "hqq_quantized", False):
            print("Model was already quantized")
            return

        # Set linear tags automatically
        cls.setup_model(model)

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        elif quant_config == {}:
            patch_params = {key: None for key in model.linear_tags}
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.layers)
                if hasattr(model, "model")
                else len(model.layers)
            )
            all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # print(device_map)

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is HQQLinear:
                return linear_layer

            current_device = device_map[linear_layer.name]
            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        model.hqq_quantized = True

        return model

# Auto class used for HF models if no architecture was manually setup
class AutoHQQHFModel(CustomHQQHFModel, CustomPatch):
    pass

class AutoHQQTimmModel(CustomHQQTimmModel, CustomPatch):
    pass
```


## Part 1: Training, DoRA integration, and uploading
We will begin training our model. First, we need to import the following tools.
```
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import login
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from hqq_utils import AutoHQQHFModel, get_size_of_model
```
### Supporting Functions
In addition to the patching classes, we define several essential utility functions.

These include:
* A quantization configuration builder tailored to LLAMA3
* A tokenization function to process dataset examples
* A function to group tokenized texts into fixed-length blocks for training

```
def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config

    return quant_config

def tokenize_fn(examples):
        return tokenizer(examples["text"])

def group_texts(examples):
    block_size = 512
    concat = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concat["input_ids"]) // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_len, block_size)] for k, t in concat.items()}
    result["labels"] = result["input_ids"].copy()
    return result

```
### Step 1: Load and Quantize Base Model


We load LLAMA3-3B (Instruct) and apply 4-bit HQQ quantization.
```
login(token="hf_YOUR_ACCESS_TOKEN")  # Replace with your token

torch.manual_seed(0)
random.seed(0)
device = 'cuda:0'

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

HQQLinear.set_backend(HQQBackend.PYTORCH)
print(f"Model size before quantization: {get_size_of_model(model) / (1024**2):.2f} MiB")

quant_config = get_quant_config_slm(model)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.bfloat16, device=device)
print(f"Model size after quantization: {get_size_of_model(model) / (1024**2):.2f} MiB")

```


### Step 2: Integrate  DoRA Adapters
```
lora_cfg = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    task_type=TaskType.CAUSAL_LM,
    use_dora=True
)
for p in model.parameters():
    p.requires_grad = False
model = get_peft_model(model, lora_cfg)
model.to(torch.bfloat16)
model.print_trainable_parameters()
```

### Step 3: Fine-Tune on WikiText-2

```
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])


lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)

training_args = TrainingArguments(
    output_dir="./adapter_ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./my_adapter")
tokenizer.save_pretrained("./my_adapter")
```

### Step 4: Merging adapter with the base model and upload to Hugging Face Hub
```
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, attn_implementation="sdpa").eval()
lora_model = PeftModel.from_pretrained(base, "./my_adapter").eval()
merged = lora_model.merge_and_unload().eval()

hf_repo = "will200112/quantized-llama3-3b"
merged.push_to_hub(hf_repo, safe_serialization=True)
tokenizer.push_to_hub(hf_repo)

print(f"Model and tokenizer uploaded to https://huggingface.co/{hf_repo}")

```

## Part 2: Downloading, re-quantizing, and inference benchmarking
We will begin downloading, re-quantizing our model. First, we need to import the following tools.
```
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cuda as bk
from tqdm import tqdm, auto
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from peft import PeftModel
from hqq.utils.patching import prepare_for_inference
from hqq.core.quantize import BaseQuantizeConfig
from hqq_utils import get_linear_tags_from_model, get_size_of_model, AutoHQQHFModel
```
### Supporting Functions
We define several essential functions.
These include:
* A quantization configuration builder tailored to LLAMA3
* A token generation loop using past key values
* A perplexity evaluation function against the WikiText-2 dataset
```
def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=1024)

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config

    return quant_config
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)

    nsamples = test_enc.numel() // model.seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()


```
### Step 1: Download the model we uploaded before from Hugging Face

```
bk.enable_flash_sdp(False)
bk.enable_mem_efficient_sdp(True)
bk.enable_math_sdp(False)
torch.manual_seed(0)
random.seed(0)
device = "cuda:0"

hf_repo = "will200112/quantized-llama3-3b"  

merged = AutoModelForCausalLM.from_pretrained(
    hf_repo,
    torch_dtype=torch.float16,
    device_map=device,
    attn_implementation="sdpa"
).eval()
```
### Step 2: Quantize the merged model with 4-bit & group size= 1024
``` 
quant_cfg = get_quant_config_slm(merged)
AutoHQQHFModel.quantize_model(
    merged,
    quant_config=quant_cfg,
    compute_dtype=torch.float16,
    device=device,
)

print(f"Model size after quantization: {get_size_of_model(merged)/(1024**2):.2f} MiB")
```
### Step 3: Run the inference benchmarking and perplexity evaluation
```
tokenizer = AutoTokenizer.from_pretrained(hf_repo)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    merged.config.pad_token_id = tokenizer.eos_token_id

model = merged
prepare_for_inference(model.model, backend="gemlite")

model.forward = torch.compile(
    model.forward,
    mode="max-autotune",
    fullgraph=False,
    dynamic=True,
)
model.prefill_forward = model.forward

cache_device = next(model.parameters()).device
max_new_tokens = 256
past_key_values = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=max_new_tokens + 16,
    device=cache_device,
    dtype=torch.float16,
)

warmup_prompt = "Explain what AI is."
inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
for _ in tqdm(range(5), desc="Warm Up..."):
    _ = generate(model, input_ids, past_key_values, max_new_tokens)
    past_key_values.reset()

prompt = "How to learn a new language?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
tputs = []
time_record = []

for _ in tqdm(range(10), desc="Test Inference"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    generated = generate(model, input_ids, past_key_values, max_new_tokens)
    past_key_values.reset()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    tput = generated[0][input_ids.shape[1]:].shape[0] / (elapsed_ms / 1000)
    time_record.append(elapsed_ms / 1000)
    tputs.append(tput)

response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
sorted_tputs = np.sort(tputs)[2:-2]
org_tput = np.mean(sorted_tputs)

print(f'Prompt: {prompt}\nResponse: {response}\n')
print(f'Time Record: {time_record}')
print(f'Throughput Record: {tputs} toks/s\n')
print(f'Throughput: {org_tput} toks/s')

ppl = evaluate_ppl(model, tokenizer, device)
print(f"Perplexity (PPL): {ppl}")

with open("result.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "value"])
    writer.writerow([0, round(ppl, 2)])
    writer.writerow([1, round(org_tput, 1)])
```
 At the end of the full pipeline, it will generate a result file:
***result.csv***
This file contains two key performance metrics: PPL, Throughput 
