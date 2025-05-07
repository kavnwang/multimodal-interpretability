import torch
import nnsight
from PIL import Image
import os, json, argparse, itertools, math, torch, numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
from vllm.inputs import TextPrompt
from nnsight.modeling.vllm import VLLM
import einops
import json

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model_name = "llava-hf/llava-1.5-7b-hf"
model = VLLM(model_name, device='cuda', dispatch=True)
processor = AutoProcessor.from_pretrained(model_name)

# First, load the image and define it properly
img = Image.open("/root/multimodal-interpretability/scripts/dataset/clean-1.jpg")

# LLaVA requires a special token format for image inputs
# Use the correct placeholder format: "<image>"
with model.trace() as tr, tr.invoke(TextPrompt(prompt="<image>\nHello", multi_modal_data={'image': img})):
    # LLaVA 1.5-7B has 32 layers
    num_layers = 32
    qkv = [model.language_model.model.layers[L].self_attn.qkv_proj.output.save() for L in range(num_layers)]

print(qkv) 