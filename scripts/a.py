#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


model_name = "llava-hf/llava-1.5-7b-hf"
model = VLLM(model_name, device='cuda', dispatch=True)
processor = AutoProcessor.from_pretrained(model_name)

print(model.language_model.model.layers[0].self_attn.qkv_proj)


# In[3]:


print(model.language_model.model.layers[0].self_attn.o_proj)


# In[6]:


with model.trace() as tr, tr.invoke(TextPrompt(prompt="Hello", multi_modal_data={'image': Image.open("test.png")})):
    qkv = model.language_model.model.layers[0].self_attn.qkv_proj.output.save()


# In[7]:


print(qkv)


# In[9]:


img = Image.open("/root/multimodal-interpretability/scripts/dataset/clean-1.jpg")
with model.trace() as tr, tr.invoke(TextPrompt(prompt="Hello", multi_modal_data={'image': img})):
    qkv = model.language_model.model.layers[0].self_attn.qkv_proj.output.save()

