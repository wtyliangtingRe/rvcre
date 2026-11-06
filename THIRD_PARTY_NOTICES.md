# Third-party notices

RVCre's integration code is licensed under **GPL-3.0-only**. Third-party source,
packages, and model artifacts retain their own licenses and notices. This file
records the selected upstreams; it does not replace their original LICENSE or
NOTICE files. Model weights and user audio are not bundled in this repository.

## Pinned model backends

| Component | Upstream / author | Selected revision | License | Use |
|---|---|---|---|---|
| VoxCPM source | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | `4b8dd4440d33396b3f429201f8af0529b4d2c601` | [Apache-2.0](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/LICENSE) | Voice design, TTS, official LoRA trainer/configuration |
| VoxCPM2 model | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `32279effe8c19989596f05d353d1447f51d9e915` | Apache-2.0, model card | Local base model and AudioVAE |
| Seed-VC source | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | `51383efd921027683c89e5348211d93ff12ac2a8` | [GPL-3.0](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/LICENSE) | Timbre conversion; upstream archived 2025-11-21 |
| Seed-VC v2 weights | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC) | `257283f9f41585055e8f858fba4fd044e5caed6e` | GPL-3.0, model card | CFM and AR artifacts selected by upstream wrapper |
| ASTRAL quantizer weights | [Plachta/ASTRAL-quantization](https://huggingface.co/Plachta/ASTRAL-quantization) | `4a2e9679f76eb03753adc8c503e3c23bb9c22f26` | GPL-3.0, model card | Seed-VC content representation |
| CAMPPlus weights | [funasr/campplus](https://huggingface.co/funasr/campplus) | `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` | Apache-2.0, model card | Speaker conditioning |
| Whisper tokenizer/configuration | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | `973afd24965f72e36ca33b3055d56a652f456b4d` | Apache-2.0, this HF repository's model card | Tokenizer/configuration only; Whisper acoustic weights are not selected for Seed-VC |
| HuBERT weights | [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) | `ff022d095678a2995f3c49bab18a96a9e553f782` | Apache-2.0, model card | Speech features |
| BigVGAN weights | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) | `633ff708ed5b74903e86ff1298cf4a98e921c513` | MIT, repository LICENSE | Waveform synthesis |

The original [OpenAI Whisper source](https://github.com/openai/whisper/blob/main/LICENSE)
uses MIT; this is distinct from the license metadata of the selected Hugging
Face tokenizer repository above. Do not substitute one record for the other.

The exact download file list is maintained in
[`src/rvcre/backends/specs.py`](src/rvcre/backends/specs.py). Download receipts
record the revision and local destination. When redistributing an installed
bundle, include the applicable upstream licenses/notices and corresponding
source obligations, including transitive components; a download receipt alone
does not replace them.

## Application and optional data tooling

The application uses NumPy, SciPy, SoundFile/libsndfile, librosa, Pydantic,
FastAPI, Uvicorn, Gradio, PyYAML and other packages declared in
`pyproject.toml`. Optional local transcription uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and CTranslate2.
Package distributions include their respective licenses. Installed versions
and platform-specific binaries should be retained in an environment receipt
when creating a redistributable build.

FFmpeg may be used from the user's system or provided by an upstream dependency.
Its applicable license depends on the actual build and enabled components;
RVCre does not relicense that executable or assume every FFmpeg build has the
same license. No FFmpeg executable is committed here.

## Model configuration attribution

`scripts/train_voxcpm.py` reads OpenBMB's actual
[`conf/voxcpm_v2/voxcpm_finetune_lora.yaml`](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/conf/voxcpm_v2/voxcpm_finetune_lora.yaml)
and invokes their trainer. RVCre's changes are the local manifest/path
validation, run-plan generation, argument overrides and offline invocation.
The upstream training algorithm remains OpenBMB's implementation.

## User-provided audio

User recordings, game dialogue, role identities, transcripts and user-trained
checkpoints are not automatically licensed under GPL by being placed in an
RVCre workspace. Keep a short source/provenance note with local assets. The
current source repository contains no Genshin Impact recordings or other
user-provided voice corpus; examples contain newly written text and placeholder
paths only. See [docs/LICENSING.md](docs/LICENSING.md).
