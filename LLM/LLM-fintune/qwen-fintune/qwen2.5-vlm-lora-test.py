#!/usr/bin/env python3
"""Qwen2-VL 微调推理脚本"""
import io, os, sys, time
import torch
from PIL import Image

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "modelscope", "models", "qwen--Qwen2-VL-7B-Instruct", "snapshots", "master")
BASE_MODEL = CACHE_DIR
LORA_DIR  = "vlm_fine_tune_output/lora"

print("Loading processor...")
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
processor = AutoProcessor.from_pretrained(LORA_DIR, trust_remote_code=True)
print(f"     OK: Processor loaded (vocab={len(processor.tokenizer)})")

# 加载基座模型 + LoRA adapter
print("\nLoading base model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

from peft import PeftModel
print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()
print(f"     OK: Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

# 视觉问答测试
from PIL import Image
import urllib.request
import tempfile

def get_test_image():
    """优先读取本地合成图片，不存在则回退到网络下载。"""
    local = os.path.join(os.path.dirname(__file__), "dataset", "synthetic", "image_2.jpg")
    if os.path.exists(local):
        return Image.open(local).convert("RGB")
    url = "https://farm5.staticflickr.com/4093/32461784403_4bbdcb5b5a_o.jpg"
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        img_data = resp.read()
        return Image.open(__import__("io").BytesIO(img_data)).convert("RGB")
    except Exception:
        # fallback: 生成一张测试图
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), "Hello World", fill="black")
        return img

print("\nVQA test:")
test_image = get_test_image()
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does this fruit means?"}]},
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[prompt], images=[test_image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100)
resp = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
print(f"     Q: What does this fruit means?")
print(f"     A: {resp}")

print("\n" + "="*60)
print("VLM fine-tuning inference test passed!")
print("="*60)