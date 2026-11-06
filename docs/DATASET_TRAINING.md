# 本地音源准备与 VoxCPM2 LoRA 训练

本工具直接使用本地文件，输出官方训练器可读的清单；不会下载音源或大模型，也不会在数据准备阶段自动开始训练。基础模型由单独的下载脚本准备，真正训练由官方脚本执行。

## 1. 输入组织

支持两种自动识别布局：

| 布局 | 例子 | 说话人 |
|---|---|---|
| 单人目录 | `raw/voice_a/001.wav` 与 `001.txt` | 当 `source=raw/voice_a` 时，默认为 `voice_a` |
| 多人目录 | `raw/voice_a/001.wav`、`raw/voice_b/001.wav`，各有同名文字 | 当 `source=raw` 时，第一层子目录是说话人 |

纯文本 `.lab` 也可以使用。文字按 UTF-8 读取，可带 BOM。音频格式包括 WAV、FLAC、MP3、M4A、AAC、OGG、Opus、AIFF、WMA；SoundFile 无法解码的格式需要可用的 FFmpeg/ffprobe。

如果同一个说话人还按 session 分了子目录，请显式传 `speaker`，避免把 session 当成不同说话人。根目录与子目录同时有音频且未指定说话人时会报出布局歧义，不静默混人。显式指定 `speaker` 表示这次输入全部属于该说话人。

同名 `.txt` 与 `.lab` 同时存在但内容不同会进入报告。带有时间戳的 HTK `.lab` 不会被误当成普通文本；请先转换成正确的逐片段纯转写。原始音频、文字始终不改写。

## 2. 准备数据

CLI 的完整选项可以通过 `rvcre dataset prepare --help` 查看：

```bash
rvcre dataset prepare raw/voice_a workspace/datasets/voice_a_v1 --speaker voice_a
```

底层 Python API 可以直接运行：

```python
from pathlib import Path
from rvcre.dataset import prepare_dataset

summary = prepare_dataset(
    source=Path("raw/voice_a"),
    output=Path("workspace/datasets/voice_a_v1"),
    speaker="voice_a",
    language="zh",
    validation_ratio=0.1,
    seed=42,
    min_seconds=3,
    max_seconds=30,
    transcribe=False,
    asr_model="small",
)
print(summary)
```

输出目录必须新建或为空，避免覆盖上一版数据。允许输出位于源目录下，扫描时会排除输出目录；但不允许输出等于或包含源目录。

输出文件：

| 文件 | 内容 |
|---|---|
| `train.jsonl` | 训练样本 |
| `validation.jsonl` | 验证样本；数据不足时可以为空 |
| `audio/<speaker>/<source_id>_<segment>.wav` | 单声道、16 kHz、float WAV；没有降噪、改调或自动响度归一化 |
| `metadata.jsonl` | 源文件、说话人、时间区间、划分、转写来源及待复核标记 |
| `preparation_report.json` | 计数、路径、警告和被跳过材料的具体原因 |

JSONL 每行包含 `audio`、`text`、`ref_audio`、`duration`、`ref_duration`、`dataset_id`。音频路径相对于 JSONL 所在目录，整个数据目录可搬移。示意：

```json
{"audio":"audio/voice_a/clip_001.wav","text":"今天的风很轻。","ref_audio":"audio/voice_a/clip_002.wav","duration":4.2,"ref_duration":5.1,"dataset_id":0}
```

16 kHz 是 VoxCPM2 AudioVAE encoder 的训练输入采样率；官方 v2 模板同时设置 48 kHz decoder 输出用于生成及音频日志。不要为了输出 48 kHz 就把训练 `sample_rate` 改成 48000。来源：[官方 v2 LoRA 配置](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/conf/voxcpm_v2/voxcpm_finetune_lora.yaml)。

## 3. 长音频与缺文字

默认只接收 3–30 秒的有效音频。缺文字会列出 `missing_transcript`；过长而没有分段对齐的录音会列出 `too_long_unaligned`。即使有一整篇文字，也不能简单按秒数平均分给切片。

启用可选 ASR 时：

1. 在主环境安装 `.[asr]`，并事先准备本地 CTranslate2 格式的 faster-whisper 模型。
2. 传入 `transcribe=True`，`asr_model` 可以是本地模型目录；默认 `small` 只查本地 Hugging Face 缓存。
3. 工具使用 CPU/int8、`local_files_only=True`，不会暗中联网下载。
4. 已有正确短音频转写仍直接使用，不必加载 ASR。缺文字或长音频才走转写。
5. ASR 片段按其时间戳合并；单个超长或重叠/无效片段会报告，不伪造内部对齐。
6. 长录音原有整篇文字不会被假装强制对齐：采用的是 ASR 的分段文字，并保留原文件供人工复核。

ASR 输出是初稿。请重点复核角色名、专业词、数字、方言、笑声、呼吸、句首句尾；工具会在 `needs_text_review` 与 `asr_review_samples` 中记录，但不设置新的人工授权关卡。缺少 ASR 依赖或本地模型时会报告 `asr_unavailable`，已有合格转写的材料仍能被准备。

当前解码/ASR 会在内存中处理一个源文件。非常长的录音应先按真实录音段整理；首版不是面向数小时混合节目的完整说话人分离和强制对齐系统。

## 4. 防止数据泄漏

- 划分单位是**原输入文件**。同一长录音产生的所有切片绑定相同 `source_id`，只进入一个划分。
- 每个说话人独立划分，固定文件集合和 seed 得到同样划分；增加或重命名文件后应视为新数据版本。
- 同一说话人至少有两个有效源文件时，非零验证比例尽量保留至少一个验证源文件和一个训练源文件。
- 参考音频只取同说话人、同划分的其他样本；优先不同源文件。不从验证集给训练集取参考。
- 一个划分只有一个片段时写 `ref_audio=null`，避免把预测目标自身作为参考。上游允许无参考样本。
- 如果输入已经是外部切好的小片段，工具不能凭文件名可靠知道它们来自同一长录音。应先保留原录音组或人工安排独立验证材料；重复拷贝同一内容也需要用户清理。

## 5. 生成可检查的 LoRA 配置

先用 `setup_backends.py` 准备固定版本上游代码。脚本默认读取 `config/backends.json`，与应用使用同一套相对路径解析规则，也支持 `RVCRE_BACKEND_CONFIG`。

```bash
python scripts/train_voxcpm.py \
  --data workspace/datasets/voice_a_v1 \
  --output workspace/training/voice_a_lora_v1 \
  --config-only
```

生成 `finetune_lora.yaml` 和 `training_plan.json`。这一步不载入模型、不启动训练；即使大权重尚未下载，也可以先检查配置。上游 checkout 和实际训练清单仍须存在。

默认设置来自真实的 v2 LoRA YAML：rank/alpha 32、LM 和 DiT LoRA 开启、projection LoRA 关闭、学习率 `1e-4`、batch 2、梯度累积 8、1000 步。为方便本地 Windows 运行，数据 worker 默认为 0，预处理 worker 为 1。这是可执行起点，不是针对用户音源调优后的最佳值。

包装器还会验证 JSONL 与音频路径、读取本地模型架构，并静态核对实际 `train()` 的配置字段。它不提供“检查通过等于训练质量好”的判断。

## 6. 真正启动训练

确保 `models/VoxCPM2` 和 `.venvs/voxcpm` 已准备后，去掉 `--config-only`：

```bash
python scripts/train_voxcpm.py \
  --data workspace/datasets/voice_a_v1 \
  --output workspace/training/voice_a_lora_v1 \
  --steps 1000 --batch-size 2 --grad-accum-steps 8
```

也可以显式指定 `--repo`、`--model`、`--python`。执行的是：

```text
<VoxCPM环境Python> -u <上游仓库>/scripts/train_voxcpm_finetune.py --config_path <本次输出>/finetune_lora.yaml
```

调用时设置 Hugging Face/Transformers/Datasets 离线环境变量；不会把模型 ID 当成悄悄下载的捷径。训练的显存、速度和声音效果需要实机验证，当前 CPU 测试没有执行神经训练。

上游会在 `checkpoints` 下保存步数目录、LoRA 权重、`lora_config.json`、优化器和训练状态。已有 `latest` 时会尝试续训；新实验使用新输出目录。评估时用独立验证文本与音频，并保留微调前的零样本基线。

## 7. 不会自动发生的事

训练包装器不自动把 LoRA 发布到远端，不自动覆盖应用的默认模型，不把第三方角色原音打入 Git，也不决定素材许可。用户提供的素材授权与项目 GPL 是不同的记录。完成训练后，应先试听和记录对应底模，再显式配置该 LoRA。

## 8. 在应用中加载已训练的 LoRA

当前 worker 已支持 `lora_weights_path`，会将它传给官方 `VoxCPM.from_pretrained()`。在现有 `config/backends.json` 的 `voxcpm` 条目中加入该字段，保留已有的 `seedvc` 条目、模型 revision 等其他设置。下面是 Windows 下的 `voxcpm` 条目示意：

```json
{
  "voxcpm": {
    "python": "../.venvs/voxcpm/Scripts/python.exe",
    "upstream_dir": "../third_party/VoxCPM",
    "model_dir": "../models/VoxCPM2",
    "device": "cuda",
    "lora_weights_path": "../workspace/training/voice_a_lora_v1/checkpoints/step_0001000"
  }
}
```

路径相对于 `config/`，与 [`config/backends.example.json`](../config/backends.example.json) 的其余字段一致。示例对应本页的 `--output workspace/training/voice_a_lora_v1`；如果使用 Windows 指南的 `data/training/...`，应改为 `../data/training/...`。把 `step_0001000` 换成实际生成的步数目录。

`lora_weights_path` 应指向同时包含 **`lora_config.json` 与 `lora_weights.safetensors`（或 `lora_weights.ckpt`）的完整检查点目录**。官方加载器会从该目录读取训练时保存的 rank、alpha 等配置。`model_dir` 仍指向训练时对应的 VoxCPM2 基础模型，不能改成 LoRA 目录。使用固定步数目录比滚动变化的 `latest` 更便于复现。实现来源：[本地 worker](../src/rvcre/workers/voxcpm.py)、[官方加载入口](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/src/voxcpm/core.py)。

首版**没有 UI 内的 LoRA 切换控件**，也不会根据声音档案自动选择 LoRA。这是整个 `voxcpm` 后端的配置，影响其新任务中的设计和 TTS。修改配置后再提交任务；删除 `lora_weights_path` 或设为 `null` 可恢复基础模型。接口路径已核实，实际 LoRA 载入和声音效果仍需在训练完成后实机验证。

官方参考：[训练入口](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/scripts/train_voxcpm_finetune.py)、[manifest 读取代码](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/src/voxcpm/training/data.py)、[官方微调指南](https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html)。
