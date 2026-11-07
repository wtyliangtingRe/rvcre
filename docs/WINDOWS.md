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

### 使用已有 Conda 的另一种入口

如果 Windows 的 `py -3.11` 找不到解释器，可以直接用 Conda 创建 Python 3.11 主环境：

```powershell
conda create -n rvcre python=3.11 pip -y
conda env config vars set -n rvcre PYTHONNOUSERSITE=1 PIP_USER=0
conda activate rvcre
python -m pip install -e ".[ui]"
python -m pip check
python -m rvcre ui
```

`PYTHONNOUSERSITE=1` 关闭默认用户包目录，避免读取 `%APPDATA%\Python\Python311\site-packages` 中其他工具的依赖；`PIP_USER=0` 关闭 pip 用户目录安装模式。这两个设置只保存在 `rvcre` 环境中。已有环境可直接设置变量，然后 `conda deactivate`、`conda activate rvcre` 使其生效。它们不会还原此前已经被其他安装操作替换的共享包。

日常先 `conda activate rvcre`，再在仓库根目录执行 `python -m rvcre ui`。当前 `Start-RVCre.cmd` / `Start-RVCre.ps1` 固定使用 `.venv`；Conda 路线使用上述 Python 入口。下文命令开头的 `.\.venv\Scripts\python.exe` 相应换成 `python`，模型专用的 `.venvs/voxcpm` 和 `.venvs/seedvc` 仍由脚本自动管理。

检查隔离状态可运行：

```powershell
python -c "import sys, site; print(sys.executable); print('ENABLE_USER_SITE:', site.ENABLE_USER_SITE); print('Roaming package paths:', [p for p in sys.path if 'roaming' in p.lower() and 'site-packages' in p.lower()])"
```

解释器应位于 Conda 的 `envs/rvcre/`，后两项应为 `False`、`[]`。来源：[Python 用户目录开关](https://docs.python.org/3.11/using/cmdline.html#envvar-PYTHONNOUSERSITE)、[Conda 环境变量](https://docs.conda.io/projects/conda/en/stable/commands/env/config/vars/set.html)。

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

### 软件源与构建依赖

PyTorch 三件套使用 `--torch-index` 选择的官方 wheel 源。其余 Python 包默认沿用本机 pip 配置；需要明确指定通用包源时，可传 `--package-index`：

```powershell
.\.venv\Scripts\python.exe scripts/setup_backends.py --backend voxcpm --package-index https://pypi.org/simple
```

这个选项用于普通依赖和它们的隔离构建环境，保持 PyTorch 的 CUDA wheel 源独立。使用 Conda 时，命令开头换成 `python`。也可以明确指定自己使用的通用 Python 镜像地址。

若 PyTorch 已装好，接着在 `Installing build dependencies` 阶段报 `No matching distribution found for setuptools_scm>=8`，表示 pip 没能从当前可用候选中找到所需构建工具。该包有支持 Python 3.11 的版本，仅凭这一行不能确定是索引配置、镜像还是网络原因。先用上面的显式 PyPI 命令重试；同一环境中满足版本要求的 PyTorch 会被复用。

VoxCPM 的构建依赖在上游 `pyproject.toml` 中声明。pip 会在临时隔离环境安装这些工具，因此只在主程序 Conda 环境里手动装 `setuptools-scm` 不会解决隔离构建的取包问题。安装脚本保留构建隔离；此问题不需要清空下载缓存、删掉模型环境或修改上游文件。来源：[固定版本 VoxCPM 构建声明](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/pyproject.toml)、[pip 构建隔离说明](https://pip.pypa.io/en/stable/reference/build-system/#build-isolation)。

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
| Conda 安装时引用 Roaming 用户目录旧包 | 按上面的 Conda 路线关闭用户包目录，重新激活后正常补装并执行 `pip check` |
| 构建阶段找不到 `setuptools_scm>=8` | 用 `--package-index https://pypi.org/simple` 明确普通依赖源；保留已装好的 PyTorch |
| 界面可开，生成后端未配置 | 运行对应 `setup_backends.py --backend ...`，随后显式下载模型 |
| `.wav` 可读，M4A/其他文件失败 | 安装可执行的 FFmpeg/ffprobe，或先转成 WAV；不要仅改扩展名 |
| `doctor` 通过，GPU 推理仍失败 | 看 `backend.log` 中实际 CUDA、显存或依赖错误；doctor 不是 GPU 推理测试 |
| Hi-Fi 报参考转写缺失 | 导入与文字逐字对应且无需截取的短参考；可控模式不需要该转写 |
| ASR 找不到模型 | 提供已有本地 faster-whisper 模型目录；准备工具不会联网自动下载 |
| 相对模型路径不对 | `config/backends.json` 内相对路径以 `config/` 为基准，例如 `../models/VoxCPM2` |

当前已完成的验证以 CPU 和应用流程为主。没有把“Windows 文档已写好”表述为“所有 Windows GPU 已实测”；具体记录见 [VALIDATION.md](VALIDATION.md)。
