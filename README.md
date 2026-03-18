# Transformer-DeEn-NMT

本项目是一个基于 **Transformer** 架构的德语-英语神经机器翻译（NMT）系统。项目使用 **WMT16** 德英平行语料库进行训练，完整实现了论文《Attention Is All You Need》中的核心机制。

## 🌟 项目亮点

- **完全基于注意力机制**：摒弃了传统的 RNN 和 CNN，利用自注意力（Self-Attention）实现全序列并行计算。
- **高效的数据流水线**：自定义 `LangPairDataset` 支持缓存加速，并实现按句长分组的采样器（TokenBatchCreator）以优化批处理效率。
- **先进的训练策略**：集成了标签平滑（Label Smoothing）、Noam 学习率衰减调度器以及早停机制。
- **可视化与评估**：支持 **BLEU-4** 指标自动化评估，并提供注意力热力图（Attention Heatmap）可视化功能，直观展现模型翻译时的对齐逻辑。

## 🏗️ 模型架构

模型严格遵循 Transformer 的编码器-解码器（Encoder-Decoder）结构：

### 1. 基础组件
- **词嵌入与位置编码 (Embedding & Positional Encoding)**：将词汇映射为稠密向量，并注入正弦/余弦位置信息以捕获序列顺序。
- **多头注意力 (Multi-Head Attention)**：通过多个注意力头并行捕获不同子空间的语义关联。
- **前馈神经网络 (Position-wise Feed-Forward)**：双层线性变换结合 ReLU 激活函数，增强模型的非线性表达能力。
- **残差连接与层归一化 (Add & Norm)**：在每个子层后应用，有效缓解深层网络的梯度消失问题。

### 2. 编码器 (Encoder)
- 由 6 层相同的 Encoder Layer 堆叠而成。
- 负责提取源语言（德语）的深度语义特征。

### 3. 解码器 (Decoder)
- 由 6 层相同的 Decoder Layer 堆叠而成。
- **掩码自注意力 (Masked Self-Attention)**：防止模型在预测当前词时“偷看”未来的信息。
- **交叉注意力 (Encoder-Decoder Attention)**：解码器通过此层关注编码器的输出，实现源语言与目标语言的对齐。

## 📊 数据处理流程

1. **语料加载**：使用 WMT16 德英平行语料，通过 `LangPairDataset` 进行加载。
2. **文本预处理**：过滤超长句子，执行清洗与标准化。
3. **分词 (Tokenizer)**：自研分词器，支持 `[BOS]`、`[EOS]`、`[PAD]` 等特殊标记，实现文本的高效编码与解码。
4. **批处理优化**：
   - `TransformerBatchSampler`：按句长动态分组。
   - `TokenBatchCreator`：保证每批大约包含 4096 个标记，最大化 GPU 利用率。

## 🚀 训练与优化

- **损失函数**：使用带掩码的交叉熵损失（CrossEntropyWithPadding），并引入 **标签平滑 (Label Smoothing)** 提升泛化能力。
- **优化器**：Adam 优化器。
- **学习率调度**：实现 **NoamDecayScheduler**，根据模型维度和步数动态调整学习率。
- **回调机制**：完善的模型保存与早停逻辑，记录训练过程中的各项指标。

## 📈 评估指标

- **BLEU-4 Score**：定量评估翻译质量。
- **Attention Map**：生成注意力分布矩阵，可视化源语言词汇与目标语言词汇的关联权重。

## 📂 项目结构

```text
Project2_Transformer/
├── wmt16/                  # 训练、验证与测试数据集
├── .cache/                 # 预处理数据缓存
├── transformer_训练.ipynb   # 核心训练与模型实现脚本
├── README.md               # 项目说明文档
└── data_multi30k.py        # 辅助数据处理工具
```

---
*注：本项目目前处于开发完善阶段。*
