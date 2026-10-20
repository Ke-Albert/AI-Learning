# AI-Learning 学习索引

按主题维护深度学习学习记录。`main` 保存主题介绍与学习状态，代码在对应的 `topic/*` 分支中。

## 主题

| 主题分支 | 学习内容 | 状态 |
| --- | --- | --- |
| [深度学习基础与实战](../../tree/topic/dl-fundamentals) | 房价预测、图像分类、归一化与损失函数 | 已有代码，待逐项验证 |
| [视觉 Transformer](../../tree/topic/vision-transformers) | ViT 与 Swin Transformer | 已有代码，待逐项验证 |
| [扩散模型](../../tree/topic/diffusion) | DDPM 与 DDIM | 已有代码，待逐项验证 |
| [大模型注意力机制](../../tree/topic/llm-attention) | DeepSeek MLA 实现 | 已有代码，待逐项验证 |
| [大语言模型微调](../../tree/topic/llm-finetuning) | 继续预训练、指令微调、LoRA/QLoRA、诗词微调 | 已有代码，待逐项验证 |
| [多模态模型微调](../../tree/topic/multimodal-finetuning) | Qwen VLM、视觉模型微调与图像数据构建 | 已有代码，待逐项验证 |
| [强化学习与偏好对齐](../../tree/topic/llm-alignment) | RL 基础、PPO/RLHF、DPO、ORPO、GRPO | 已有代码，待逐项验证 |
| [检索增强与问答系统](../../tree/topic/rag-qa) | 手写 RAG、LangChain、LlamaIndex、Embedding 微调与文档问答 | 已有代码，待逐项验证 |
| [分布式训练](../../tree/topic/distributed-training) | Accelerate 训练示例（探索中） | 探索中 |

状态描述依据现有文件和提交记录，尚未重新运行实验，不将代码提交等同于完成学习。

## 使用与维护

```bash
git switch topic/rag-qa
git switch main
```

在对应主题分支提交代码；回到 main 更新本索引中的进展、结论和分支链接。各主题分支为长期学习分支，不整分支合并回 main。共享代码如需修改，应明确同步到需要它的主题。

## 历史与备份

- 拆分前的 main：`d170a5da755329bbc90ae57381b905c04ef405b1`，完整提交历史仍然保留。
- 工作区学习记录快照：`backup/pre-topic-split-20261020-174653`，包含拆分前非忽略文件的本地修改及新增文件。
- 原有 `q/a-system-llamaindex` 分支保持原样，其问答文件已纳入 RAG 主题。
- `play.ipynb` 尚未确定主题，保留在备份分支；IDE 配置也仅保留在历史和备份中。
- 本地 `.env`、缓存等被忽略文件不纳入主题提交。

本次调整只整理内容，不声称完成模型训练、效果评估或问答系统在线验证。
