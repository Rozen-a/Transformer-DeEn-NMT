# Transformer-DeEn-NMT

本项目是一个基于 **Transformer** 架构的德语-英语（De-En）神经机器翻译（NMT）系统。项目完整实现了论文《Attention Is All You Need》中的核心机制，并在 **WMT16** 德英平行语料库上进行了验证。

## 🌟 项目亮点

- **先进的架构设计**：完全摒弃 RNN 和 CNN，利用**自注意力（Self-Attention）**和**位置编码（Positional Encoding）**处理序列数据，支持全序列并行计算。
- **端到端数据流水线**：
  - 集成了 **Moses Tokenization**（通过 `sacremoses`）进行语言相关的初步清洗。
  - 应用了 **BPE（Byte Pair Encoding）** 子词分词技术（通过 `subword-nmt`），有效解决 OOV（Out-of-Vocabulary）问题。
  - 自定义 `LangPairDataset` 支持缓存加速，并结合 `TokenBatchCreator` 按标记数量动态分组，最大化 GPU 利用率。
- **深度模型优化**：
  - **多头注意力（Multi-Head Attention）**：并行捕获不同子空间的语义关联。
  - **训练策略**：集成了标签平滑（Label Smoothing）、Noam 学习率衰减调度器。
  - **正则化**：在子层应用残差连接（Residual Connection）与层归一化（Layer Normalization）。
- **完善的评估与可视化**：支持 **BLEU-4** 指标自动化评估，并提供注意力热力图（Attention Heatmap）分析模型对齐逻辑。

## 🏗️ 模型架构

模型遵循经典的 Transformer 编码器-解码器结构：

### 1. 编码器 (Encoder)
- 堆叠 6 层相同的 Encoder Layer。
- 核心组件：多头自注意力、位置前馈网络（Point-wise FFN）。
- 负责提取源语言（德语）的深度语义特征。

### 2. 解码器 (Decoder)
- 堆叠 6 层相同的 Decoder Layer。
- **掩码自注意力 (Masked Self-Attention)**：通过因果掩码（Causal Mask）防止“偷看”未来信息。
- **交互注意力 (Encoder-Decoder Attention)**：解码器通过此层关注编码器的最终输出，实现源语与目标语的语义对齐。

### 3. 共有组件
- **位置编码 (Positional Encoding)**：利用正弦/余弦函数注入绝对位置信息。
- **Add & Norm**：每个子层后均跟随残差连接与层归一化。

## 📊 数据处理流程

1. **初步分词**：运行 `data_multi30k.py` 使用 Moses 分词器对德语和英语进行标准化处理。
2. **子词建模**：应用 BPE 算法学习并应用子词切分，生成 `.bpe` 文件。
3. **数据加载**：`LangPairDataset` 加载 BPE 处理后的文本，过滤超长句子。
4. **批处理**：按句长动态分组，确保每批包含约 4096 个 Tokens。

## 🚀 训练与推理

- **损失函数**：带掩码的交叉熵损失（CrossEntropyWithPadding）+ 标签平滑。
- **优化器**：Adam (β1=0.9, β2=0.98)。
- **学习率调度**：实现 Noam 学习率调度器，随训练步数动态调整。
- **训练监控**：集成 **TensorBoard**，实时记录训练/验证损失及学习率曲线。
- **模型保存**：自动保存验证集表现最佳的检查点（`best.ckpt`）。
- **核心脚本**：详细实现与训练结果见 `transformer_带bleu-autodl训练结果.ipynb`。

## 📈 评估指标

- **BLEU Score**：使用 NLTK 库计算 BLEU-4 分数，定量衡量翻译质量。
- **Attention Map**：生成 2D 矩阵热力图，可视化 Query 与 Key 之间的权重分布，验证模型对齐效果。

## 📂 项目结构

```text
Project2_Transformer/
├── wmt16/                  # WMT16 原始语料与 BPE 处理后的数据
├── wmt16_cut/              # Moses 分词后的中间结果
├── .cache/                 # 预处理数据缓存 (.npy)
├── checkpoints/            # 模型检查点存储 (best.ckpt)
├── runs/                   # TensorBoard 训练日志
├── transformer_带bleu-autodl训练结果.ipynb  # 核心实现、训练与评估
├── data_multi30k.py        # 基于 Moses 的数据清洗脚本
└── README.md               # 项目说明文档
```

---
*注：本项目开发基于 PyTorch 2.5+，建议在支持 CUDA 的环境下运行。*
