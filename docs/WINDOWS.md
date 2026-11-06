# Windows 本地安装与运行

主应用、VoxCPM2、Seed-VC 使用三个独立环境。可以先打开界面和分析录音，再按需要准备一个或两个生成后端。首次安装依赖和下载模型需要联网，完成后神经推理工作进程使用本地文件并开启离线模式。

下面的命令在项目根目录运行。脚本安装入口使用 **PowerShell 7（`pwsh`）**，避免 Windows PowerShell 5.1 对无 BOM 的 UTF-8 中文脚本产生编码差异；没有 PowerShell 7 时可按下方命令手动创建环境。准备好 Python 3.11、Git；部分音频格式还需要 FFmpeg/ffprobe。主应用支持 Python 3.11 及以上，模型安装脚本固定创建 Python 3.11 环境。

## 1. 先安装界面与 CPU 工具

```powershell
pwsh -NoProfile -File .\scripts\Setup-RVCre.ps1
```

如果 Python Launcher 找不到 Python，可给 setup 传真实解释器路径：

```powershell
pwsh -NoProfile -File .\scripts\Setup-RVCre.ps1 -Python "C:\Python311\python.exe"
```

没有 PowerShell 7 时，也可以在现有 PowerShell 中手动执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
```

两种方式都创建 `.venv` 并安装应用和界面依赖，不下载基础模型。启动：

```powershell
.\scripts\Start-RVCre.cmd
```

也可直接运行：

```powershell
.\.venv\Scripts\python.exe -m rvcre ui --workspace data --host 127.0.0.1 --port 7860
```

默认工作目录是 `data/`，服务只监听本机 `127.0.0.1`。声音档案、任务记录和输出音频都保存在本地。没有配置模型时仍能进入界面、导入和分析参考；生成任务会给出缺失环境或权重的具体原因，不显示伪造结果。

## 2. 准备模型环境

先查看将运行的安装命令，不做安装：

```powershell
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend voxcpm --backend seedvc --dry-run
```

真正安装可以分开进行：

```powershell
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend voxcpm
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend seedvc
```

脚本克隆固定 commit 的上游源码，分别创建 `.venvs/voxcpm` 与 `.venvs/seedvc`，安装依赖并执行基础导入检查，最后写入 `config/backends.json`。不会在此步骤下载模型权重。

当前安装配置是明确的可复核起点：VoxCPM 使用 PyTorch 2.8.0/CUDA 12.8 wheel，Seed-VC 默认使用官方依赖对应的 PyTorch 2.4.0/CUDA 12.4 wheel。脚本允许显式切换 `--torch-version` / `--torch-index`，不能把成功装包等同于每一代显卡都已经验证。

如果选择 CPU 安装，必须显式写 `--device cpu`；这表示采用 CPU wheel，不代表大模型已达到适合交互的速度。VoxCPM2 的 v2 环境不允许选择旧的 PyTorch 2.4 配置。

已有上游 checkout 包含未提交更改时，脚本会停止更新该 checkout，避免覆盖用户修改。先保留本地改动再重新设置，不需要重新删除整个应用。

## 3. 按需下载固定模型

只列出下载清单：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --backend voxcpm --backend seedvc --dry-run
```

下载对应后端：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --backend voxcpm
.\.venv\Scripts\python.exe scripts/download_models.py --backend seedvc
```

下载在各自的后端环境中执行，固定 Hugging Face revision 和文件白名单。VoxCPM2 放到 `models/VoxCPM2/`，Seed-VC 及其依赖放到 `models/SeedVC/`；完成后写 `rvcre-models.json` 收据。下载脚本是显式联网步骤，实际生成工作进程不在运行中补下载。

按已固定文件清单统计，VoxCPM2 模型资产约 **4.96 GB**，Seed-VC 所需资产约 **2.61 GB**，合计约 **7.57 GB**。这是模型文件的十进制大小，不含 PyTorch、依赖环境和可能的下载缓存。实际文件与 revision 的唯一代码清单是 [`backends/specs.py`](../src/rvcre/backends/specs.py)；统计来自对应官方 HF revision API，不是显存需求。

Seed-VC v2 使用 HuBERT/ASTRAL 内容表示；下载的 Whisper 部分是 tokenizer 等配置，不是一次隐含的 Whisper ASR 服务。

## 4. 检查并完成第一条任务

```powershell
.\.venv\Scripts\python.exe -m rvcre doctor
.\.venv\Scripts\python.exe -m rvcre voices import "D:\Audio\reference.wav" --name "声音A" --source-note "本地素材，保留原始来源记录"
.\.venv\Scripts\python.exe -m rvcre voices list
```

`doctor` 检查路径、依赖、必需模型文件；不会载入完整模型，也不能证明某张显卡上的真实推理已经成功。拿到实际声音 ID 后：

```powershell
.\.venv\Scripts\python.exe -m rvcre tts --voice <VOICE_ID> --text "今天的风很轻，我们慢慢走。"
.\.venv\Scripts\python.exe -m rvcre vc --voice <VOICE_ID> --source "D:\Audio\performance.wav"
```

`<VOICE_ID>` 是示意占位符，请替换后再运行。也可直接在 UI 里选择档案操作。神经任务在独立进程里执行，首次启动有加载时间；失败详情保存在该任务目录的 `backend.log`。

## 5. 训练数据与 LoRA

```powershell
.\.venv\Scripts\python.exe -m rvcre dataset prepare "D:\Audio\voice_a" "data\datasets\voice_a_v1" --speaker voice_a
.\.venv\Scripts\python.exe scripts/train_voxcpm.py --data "data\datasets\voice_a_v1" --output "data\training\voice_a_lora_v1" --config-only
```

检查准备报告和生成配置后，去掉 `--config-only` 才真正调用训练。训练包装器使用 `.venvs/voxcpm` 的 Python，并遵守与应用相同的配置路径规则。完整说明见 [DATASET_TRAINING.md](DATASET_TRAINING.md)。

## 6. 常见问题

| 现象 | 下一步 |
|---|---|
| 找不到 Python 3.11 | 用 `--python` 或 setup 的 `-Python` 指定解释器完整路径 |
| 界面可开，生成后端未配置 | 运行对应 `setup_backends.py --backend ...`，随后显式下载模型 |
| `.wav` 可读，M4A/其他文件失败 | 安装可执行的 FFmpeg/ffprobe，或先转成 WAV；不要仅改扩展名 |
| `doctor` 通过，GPU 推理仍失败 | 看 `backend.log` 中实际 CUDA、显存或依赖错误；doctor 不是 GPU 推理测试 |
| Hi-Fi 报参考转写缺失 | 导入与文字逐字对应且无需截取的短参考；可控模式不需要该转写 |
| ASR 找不到模型 | 提供已有本地 faster-whisper 模型目录；准备工具不会联网自动下载 |
| 相对模型路径不对 | `config/backends.json` 内相对路径以 `config/` 为基准，例如 `../models/VoxCPM2` |

当前已完成的验证以 CPU 和应用流程为主。没有把“Windows 文档已写好”表述为“所有 Windows GPU 已实测”；具体记录见 [VALIDATION.md](VALIDATION.md)。
