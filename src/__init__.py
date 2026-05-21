"""query-document 二分类实验工具包。

包内模块按职责拆分：`data` 负责统一 JSONL dataloader 和 prompt，`crc`
负责 CRC/defer/选样，`model` 和 `trainer` 负责 LoRA 训练与本地打分，
`embeddings` 负责 embedding 工件读取与覆盖校验。
"""
