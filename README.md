# RVCre · 声构

**设计一个声音，把它保存下来，再用它表达。**

本地声音设计、文字合成（TTS）与音色转换（VC）工作台。你可以导入参考录音，也可以用声音描述和参数生成候选；选中的声音保存为 `VoiceProfile`，供 TTS 和 VC 共用。

## v0.1 已实现什么

| 能力 | 当前实现 |
| --- | --- |
| 声音库 | 保留原录音、选取固定参考、保存来源与准确转写、测量有声部分音高等统计 |
| 声音设计 | 参数编译成声音描述，VoxCPM2 生成候选，试听后保存；可从既有档案的设计初值创建新版本 |
| 文字合成 | 固定参考 + 表达指令；独立 Hi-Fi 模式要求准确转写，并拒绝不支持的额外表达控制 |
| 音色转换 | Seed-VC v2 纯音色模式，输入源表演和目标参考；保留原长度比例的设置 |
| 使用入口 | 中文 Gradio 界面、命令行、本地 FastAPI、任务排队/取消/音频下载 |
| 模型准备 | 分后端独立环境，固定上游源码和权重版本，显式下载；推理使用本地资源 |
| 训练准备 | 按说话人整理音频与文本、按原文件划分训练/验证、生成官方 VoxCPM2 LoRA 配置并调用其训练器 |

**验证范围：**已执行 CPU 工作流、音频分析、数据整理、接口与 Gradio 构建/HTTP 测试。神经模型适配依据固定上游源码编写；本轮未下载大模型权重、未执行真实 GPU 推理或训练，也没有完成 TTS/VC 音色盲听。测试里的信号和后端替身仅用于测试，产品没有假音频兜底。详见 [验证记录](docs/VALIDATION.md)。

参数面板中的明亮、气声、沙哑等是**设计指令强度**；目前按描述档位映射，并非已经独立解耦的连续物理参数。导入录音只给出粗略音高设计提示，其余旋钮为设计初值。改变身份会生成新候选，不保证只改变单一特征。声音身份依靠保存的参考音频，随机种子用于记录生成条件。

## Windows 快速开始

准备 **Python 3.11、Git**。安装脚本使用 **PowerShell 7（`pwsh`）**，也可使用下面的手动安装命令。FFmpeg 用于部分音频格式解码；WAV 可以直接使用。下面命令在仓库根目录运行。

### 1. 安装界面与分析工具

```powershell
pwsh -File .\scripts\Setup-RVCre.ps1
.\scripts\Start-RVCre.cmd
```

这一步可使用声音导入和分析，尚未安装语音生成模型。也可手动安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m rvcre ui
```

默认地址为 `http://127.0.0.1:7860`。完整操作、代理和常见问题见 [Windows 安装指南](docs/WINDOWS.md)。

### 2. 准备声音设计与 TTS

```powershell
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend voxcpm
.\.venv\Scripts\python.exe scripts/download_models.py --backend voxcpm
```

安装脚本建立独立的 `.venvs/voxcpm` 环境，取得固定版本的上游源码，并写入 `config/backends.json`。下载脚本只下载所选后端列明的模型资源；生成过程不会再静默联网找模型。

固定版本的模型文件约需 **4.96 GB（VoxCPM2）+ 2.61 GB（Seed-VC 及辅助模型）= 7.57 GB**；这不包含 PyTorch、两个独立环境、下载缓存及生成音频的空间。可以先只准备 VoxCPM2，完成声音设计和 TTS 后再添加 VC。

### 3. 准备音色转换

```powershell
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend seedvc
.\.venv\Scripts\python.exe scripts/download_models.py --backend seedvc
```

安装默认使用 CUDA 版 PyTorch；具体版本与较新显卡的选择见安装指南。`--device cpu` 是显式的 CPU 安装选项，速度与可用性仍需按实际模型验证。先检查下载/安装计划可用 `--dry-run`，不会下载或改写配置。

### 4. 检查并启动

```powershell
.\.venv\Scripts\python.exe -m rvcre doctor
.\scripts\Start-RVCre.cmd
```

`doctor` 检查路径和依赖，不会加载大模型。它显示就绪，只代表资源与包检查通过；真实推理仍以生成任务的结果为准。初版每次任务单独启动模型进程，一组设计候选共用一次加载，完成后释放进程资源。

## 使用流程

**导入声音：**在「声音库」上传单人干声并命名。建议 5–12 秒。超过 12 秒时会按能量选取参考窗口，原录音完整保留；整段转写不会错误地套用到截取片段上。Hi-Fi 请导入已经剪好、文字准确对应的短参考。

**创造声音：**在「声音设计」填写描述、选择声音倾向并生成候选。试听后保存，再到「语音生成」选择这个身份。以旧档案为起点时，保存结果会建立新版本和父档案关联，不覆盖原声音。

**TTS：**填入台词，选择可控表达或 Hi-Fi。当前单次最多 600 字符，长文请按段提交。语速是模型指令倾向，不是精确的播放倍率。

**VC：**选择目标声音，再上传源表演。默认仅转换音色；不把语音识别后重新朗读当作 VC。此版本不提供精确 F0 曲线锁定，也不承诺情绪与节奏逐项无损。

声音档案保存基准参考与生成记录，各模型自行编码自己的条件。v0.1 尚未把神经条件缓存持久化到档案，也没有通用可跨模型互换的 speaker embedding。

## 命令行与 API

```powershell
# 导入一个声音（控制台返回声音 id）
.\.venv\Scripts\python.exe -m rvcre voices import .\reference.wav --name "夜航" --transcript "参考音频中实际说的话"

# 查看声音与任务
.\.venv\Scripts\python.exe -m rvcre voices list
.\.venv\Scripts\python.exe -m rvcre jobs list

# 创建设计候选；命令会等待完成，然后返回任务 id 和输出路径
.\.venv\Scripts\python.exe -m rvcre design --spec examples/voice_spec.json --candidates 3
.\.venv\Scripts\python.exe -m rvcre voices save-candidate JOB_ID --index 0 --name "夜航"

# 两种生成方式读取同一个声音 id
.\.venv\Scripts\python.exe -m rvcre tts --voice VOICE_ID --text "现在，我们可以开始新的故事了。" --emotion calm
.\.venv\Scripts\python.exe -m rvcre vc --voice VOICE_ID --source .\performance.wav

# 独立 HTTP API，交互文档 http://127.0.0.1:8000/docs
.\.venv\Scripts\python.exe -m rvcre serve
```

`JOB_ID`、`VOICE_ID` 换成实际返回的编号。声音库、任务与上传数据默认在 `data/`。`ui`、`serve` 及其他操作支持 `--workspace`、`--config`。同一工作目录同时只由一个生成队列执行；已经通过界面运行时，在该界面或 API 中提交/取消任务。CLI 生成命令会等待完成，Ctrl+C 会停止其任务。

Linux/macOS 的应用层可使用 `python3.11 -m venv .venv` 与 `.venv/bin/python`；两个模型后端在其他硬件上的推理能力以各自实现和实际验证为准。

## 以后使用音源训练

先使用预训练模型完成设计与克隆。确实需要加强某个角色的一致性时，再整理数据并使用 LoRA：

```powershell
# 先整理一个说话人的目录；录音旁放同名 .txt 或 .lab 转写
.\.venv\Scripts\python.exe -m rvcre dataset prepare .\datasets\raw\speaker-a .\datasets\prepared --speaker speaker-a

# 先写配置和调用计划，不执行训练
.\.venv\Scripts\python.exe scripts/train_voxcpm.py --data datasets/prepared --output training_runs/speaker-a --config-only

# 本地模型和数据准备好后启动官方训练器
.\.venv\Scripts\python.exe scripts/train_voxcpm.py --data datasets/prepared --output training_runs/speaker-a
```

文件组织、多人数据、可选本地 ASR、训练/验证隔离和检查点使用见 [数据与训练](docs/DATASET_TRAINING.md)。已有游戏音源可通过同一个整理入口导入；仓库初始化时没有实际音源，尚未检查其目录或质量。来源、使用范围和公开发布权限单独记录，软件许可证不替代素材授权。

## 文档

| 文档 | 内容 |
| --- | --- |
| [项目方案](docs/PROJECT_PLAN.md) | 目标、模型分工、原创声音与参考声音流程 |
| [架构](docs/ARCHITECTURE.md) | VoiceProfile、身份/表达分离、后端契约、本地运行 |
| [参数说明](docs/PARAMETERS.md) | 哪些是测量、哪些是指令、哪些尚需研究 |
| [Windows 安装](docs/WINDOWS.md) | 环境、模型下载、启动、代理与排错 |
| [数据与训练](docs/DATASET_TRAINING.md) | 音源整理、JSONL、LoRA 和本地训练 |
| [路线图](docs/ROADMAP.md) | 已实现与后续校准、缓存、适配器、长文本 |
| [验证记录](docs/VALIDATION.md) | 测试范围、当前限制、真实模型验收方法 |
| [许可与来源](docs/LICENSING.md) | 项目许可、声音来源及素材边界 |
| [第三方清单](THIRD_PARTY_NOTICES.md) | 上游源码、模型权重及依赖声明 |

## 开发

```bash
python -m pip install -e '.[dev,ui]'
python -m ruff check src tests scripts
python -m pytest -q
```

模型源码在 `third_party/`，模型权重在 `models/`，隔离环境在 `.venvs/`；这些与 `data/`、`datasets/`、`training_runs/` 默认不提交 Git。备份声音时保存整个 `data/voices/`；每个档案内部使用相对参考路径，可迁移到另一个工作目录。

## 许可

RVCre 自有代码采用 [GPL-3.0-only](LICENSE)。VoxCPM2、Seed-VC 及各辅助模型保留各自的许可证与版权声明，详见第三方清单。声音参数化不构成素材或声音权益的自动授权，也不能用相似度阈值证明声音的全球唯一性。
