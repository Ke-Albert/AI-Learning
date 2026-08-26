#!/usr/bin/env python3
"""Qwen2.5 微调训练脚本 - 支持魔搭社区下载"""
import os, sys, time

print("="*60)
print("Qwen2.5 LoRA 微调测试")
print("="*60)

# 系统信息先显示
print("\nSystem Info:")
print(f"     Python: {sys.version.split()[0]}")

import torch
print(f"     PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"     CUDA: {torch.version.cuda}")
    try:
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"     VRAM: {mem:.1f} GB")
    except Exception as e:
        print(f"     GPU info error: {e}")

# ModelScope 缓存目录
MODEL_CACHE = os.path.join(os.path.dirname(__file__), '.cache', 'modelscope')
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE

print("\nFrom ModelScope loading model...")
from modelscope import snapshot_download
try:
    model_path = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', revision='master')
except Exception as e:
    print(f"[Warning] ModelScope failed, fallback to HF: {e}")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Tokenizer
print("\nLoading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"     OK: Vocabulary size {len(tokenizer)}")

# Model
print("\nLoading model to GPU...")
start = time.time()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
print(f"     OK: Loaded in {time.time()-start:.1f}s")
print(f"        Params: {sum(p.numel() for p in model.parameters()):,}")

# LoRA
from peft import LoraConfig, get_peft_model, TaskType
print("\nConfiguring LoRA...")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"     OK: Trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# 训练数据
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

print("\nPreparing training data...")
samples = [
    {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "你是谁?"},
                  {"role": "assistant", "content": "你是LitchiCheng微调的Qwen2.5模型"}]},
    {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "你是谁微调的?"},
                  {"role": "assistant", "content": "LitchiCheng"}]},
    {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "你是不是标准的Qwen2.5模型?"},
                  {"role": "assistant", "content": "不是，我是LitchiCheng微调的模型"}]},
]

def format_sample(s):
    return {"text": tokenizer.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=False)}

dataset = Dataset.from_list([format_sample(s) for s in samples])
print(f"     OK: {len(dataset)} samples")

# 训练
timestamp = time.strftime('%Y%m%d_%H%M%S')
output_dir = f'fine_tune_output/{timestamp}'
os.makedirs(output_dir, exist_ok=True)

print("\nTraining (10 steps)...")
args = TrainingArguments(
    output_dir=output_dir, per_device_train_batch_size=1,
    gradient_accumulation_steps=4, learning_rate=2e-4, max_steps=100,
    logging_steps=1, save_steps=10, fp16=True, report_to="none",
)

trainer = SFTTrainer(model=model, train_dataset=dataset, args=args)
start = time.time()
result = trainer.train()
elapsed = time.time() - start

print(f"\n     OK: Done in {elapsed:.1f}s")
print(f"        Loss: {result.training_loss:.4f}")
print(f"        Speed: {result.global_step/elapsed:.2f} steps/sec")

# 保存
model.save_pretrained(f'{output_dir}/lora')
tokenizer.save_pretrained(f'{output_dir}/lora')
with open(f'{output_dir}/config.json', 'w') as f:
    json.dump({'r': 16, 'alpha': 32, 'lr': 2e-4, 'steps': 10}, f, indent=2)
print(f"\nSaved to {output_dir}/")

print("\n" + "="*60)
print("Fine-tuning end!")
print("="*60)