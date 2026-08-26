#!/usr/bin/env python3
"""Qwen2-VL 微调训练脚本 - LoRA + TextVQA 数据集"""
import os, sys, time, glob
from pathlib import Path

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "modelscope", "models", "qwen--Qwen2-VL-7B-Instruct", "snapshots", "master")
BASE_MODEL = CACHE_DIR
OUTPUT_DIR = "vlm_fine_tune_output"

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# 下载模型
def ensure_model():
    if os.path.isdir(os.path.join(CACHE_DIR, "model.safetensors")):
        print(f"Model already cached at {CACHE_DIR}")
        return True
    print("Downloading Qwen2-VL-7B-Instruct from ModelScope...")
    try:
        from modelscope import snapshot_download
        path = snapshot_download(
            "qwen/Qwen2-VL-7B-Instruct",
            cache_dir=os.path.join(os.path.dirname(__file__), ".cache", "modelscope", "models"),
        )
        # 链接到标准路径
        link_target = os.path.join(CACHE_DIR)
        if not os.path.exists(link_target):
            os.makedirs(os.path.dirname(link_target), exist_ok=True)
            os.symlink(path, link_target)
        print(f"     OK: Model cached at {CACHE_DIR}")
        return True
    except Exception as e:
        print(f"     FAIL: {e}")
        try:
            from modelscope import snapshot_download
            path = snapshot_download(
                "qwen/Qwen2-VL-7B-Instruct",
                cache_dir=os.path.join(os.path.dirname(__file__), ".cache", "modelscope", "models"),
            )
            print(f"     OK: Downloaded to {path}")
            return True
        except Exception as e2:
            print(f"     FAIL2: {e2}")
            return False

if not ensure_model():
    sys.exit(1)

# 创建合成数据集
def create_synthetic_dataset():
    synth_dir = os.path.join(os.path.dirname(__file__), "dataset", "synthetic")
    os.makedirs(synth_dir, exist_ok=True)

    test_data = [
        {"text": "LitchiCheng", "question": "What is written in the image?", "answer": "The image says: LitchiCheng"},
        {"text": "荔枝澄=LitchiCheng", "question": "What does the image show?", "answer": "LitchiCheng is 荔枝澄"},
    ]

    from PIL import Image, ImageDraw, ImageFont

    def _find_cjk_font(size: int = 36) -> ImageFont.FreeTypeFont:
        """在系统中搜索支持中文的字体，按优先级尝试。"""
        cjk_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/simhei/SimHei.ttf",
        ]
        for path in cjk_candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        print("     WARN: No CJK font found, falling back to DejaVu Sans")
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)

    for i, item in enumerate(test_data):
        img_path = os.path.join(synth_dir, f"image_{i}.jpg")
        if os.path.exists(img_path):
            continue
        img = Image.new('RGB', (400, 150), color='white')
        draw = ImageDraw.Draw(img)
        font = _find_cjk_font(36)
        bbox = draw.textbbox((0, 0), item["text"], font=font)
        text_width = bbox[2] - bbox[0]
        x = (400 - text_width) // 2
        y = (150 - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), item["text"], fill='black', font=font)
        img.save(img_path)

    # 构建数据集
    from datasets import Dataset as HFDataset
    samples = []
    for i in range(len(test_data)):
        samples.append({
            "id": f"image_{i}",
            "question": test_data[i]["question"],
            "answers": [test_data[i]["answer"]],
        })
    ds = HFDataset.from_list(samples)
    print(f"     OK: {len(ds)} synthetic VQA samples created")
    return synth_dir

data_dir = create_synthetic_dataset()

# 数据处理
def build_dataset():
    synth_dir = os.path.join(os.path.dirname(__file__), "dataset", "synthetic")

    from datasets import Dataset as HFDataset
    raw_data = [
        {"id": f"image_{i}", "question": q, "answer": a}
        for i, (q, a) in enumerate([
            ("What is written in the image?", "The image says: LitchiCheng"),
            ("What does the image show?", "LitchiCheng is 荔枝澄"),
            ("What does this fruit means?", "This is the profile photo of LitchiCheng"),
        ])
    ]

    def format_sample(example):
        from PIL import Image
        image_path = os.path.join(synth_dir, f"{example['id']}.jpg")
        images = [Image.open(image_path)]
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]},
             {"role": "assistant", "content": example["answer"]}],
            tokenize=False, add_generation_prompt=True
        )
        return {"image": images, "text": text}

    ds = HFDataset.from_list(raw_data)
    print(f"     OK: {len(ds)} samples")
    formatted = [format_sample(s) for s in ds]
    return HFDataset.from_list(formatted)

# 加载模型 + processor
print("\nLoading Qwen2-VL model...")
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
print(f"     OK: Processor loaded (vocab={len(processor.tokenizer)})")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
print(f"     OK: Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

# LoRA 配置
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

# 构建训练数据
def format_sample(example, img_dir):
    image_path = os.path.join(img_dir, f"{example['id']}.jpg")
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        images = [Image.open(image_path)]
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "text": example["question"]}]},
             {"role": "assistant", "content": example["answers"][0]}],
            tokenize=False, add_generation_prompt=True
        )
        return {"image": images, "text": text}
    except Exception:
        return None

import random
random.seed(42)
train_dataset = build_dataset()

# SFT 训练
from trl import SFTTrainer, SFTConfig

def collate_fn(examples):
    images = [item["image"][0] for item in examples]
    texts = [item["text"] for item in examples]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch

args = SFTConfig(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=2e-4, max_steps=100,
    logging_steps=1, save_steps=10, fp16=True, report_to="none",
    remove_unused_columns=False, dataset_text_field="",
)

trainer = SFTTrainer(
    model=model, train_dataset=train_dataset, args=args, data_collator=collate_fn,
)

start = time.time()
result = trainer.train()
elapsed = time.time() - start

print(f"\n     OK: Training done in {elapsed:.1f}s")
print(f"        Loss: {result.training_loss:.4f}")
print(f"        Speed: {result.global_step/elapsed:.2f} steps/sec")

# 保存模型和 LoRA adapter
lora_dir = os.path.join(OUTPUT_DIR, "lora")
model.save_pretrained(lora_dir)
processor.save_pretrained(lora_dir)
print(f"\n     OK: Saved to {lora_dir}")
print("="*60)