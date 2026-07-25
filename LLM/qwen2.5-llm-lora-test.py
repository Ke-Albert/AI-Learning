#!/usr/bin/env python3
"""Qwen2.5 微调推理脚本 - 加载 LoRA 权重进行推理"""
import os, sys, time
import torch

# CACHE_DIR  = os.path.join(os.path.dirname(__file__), ".cache", "modelscope", "models", "qwen--Qwen2.5-0.5B-Instruct", "snapshots", "master")
CACHE_DIR=r'D:\Model\Qwen2.5-0.5B-Instruct'
BASE_MODEL = CACHE_DIR
LORA_DIR   = "fine_tune_output/20260724_202249/checkpoint-10"

print("Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"     OK: Vocabulary size {len(tokenizer)}")

# 加载基座模型 + LoRA 权重
print("\nLoading base model...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

# 用 PEFT 加载 LoRA adapter 并合并到基座模型
print("Loading LoRA weights...")
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()   # 将 LoRA 权重合并进基座，释放 LoRA 内存
print(f"     OK: Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

# 用 chat template 做正式对话测试
print("\nChat test:")
messages = [
    {"role": "system",   "content": "You are a helpful assistant."},
    {"role": "user",     "content": "你是什么模型"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100)
resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(tokenizer.eos_token):].strip()
print(f"     {resp}")

print("\n" + "="*60)
print("Fine-tuning inference test passed!")
print("="*60)