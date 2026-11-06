"""Audited upstream versions and the assets needed by the two adapters.

Source interfaces were checked on 2026-11-06. Model revisions are Hugging Face
commit hashes, not moving branch names. Importing this module loads no ML code.
"""

UPSTREAMS = {
    "voxcpm": {
        "url": "https://github.com/OpenBMB/VoxCPM.git",
        "revision": "4b8dd4440d33396b3f429201f8af0529b4d2c601",
        "directory": "VoxCPM",
    },
    "seedvc": {
        "url": "https://github.com/Plachtaa/seed-vc.git",
        "revision": "51383efd921027683c89e5348211d93ff12ac2a8",
        "directory": "seed-vc",
    },
}

MODEL_ASSETS = {
    "voxcpm": [
        {
            "repo_id": "openbmb/VoxCPM2",
            "revision": "32279effe8c19989596f05d353d1447f51d9e915",
            "size_bytes": 4960730347,
            "directory": "",
            "files": [
                "README.md", "config.json", "model.safetensors", "audiovae.pth",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
                "tokenization_voxcpm2.py",
            ],
        },
    ],
    "seedvc": [
        {
            "repo_id": "Plachta/Seed-VC",
            "revision": "257283f9f41585055e8f858fba4fd044e5caed6e",
            "size_bytes": 712196640,
            "directory": "main",
            "files": ["README.md", "v2/ar_base.pth", "v2/cfm_small.pth"],
        },
        {
            "repo_id": "Plachta/ASTRAL-quantization",
            "revision": "4a2e9679f76eb03753adc8c503e3c23bb9c22f26",
            "size_bytes": 156343567,
            "directory": "astral",
            "files": ["README.md", "bsq32/bsq32_light.pth", "bsq2048/bsq2048_light.pth"],
        },
        {
            "repo_id": "funasr/campplus",
            "revision": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
            "size_bytes": 28038355,
            "directory": "campplus",
            "files": ["README.md", "campplus_cn_common.bin"],
        },
        {
            # Seed-VC uses this tokenizer; the Whisper acoustic model is not needed.
            "repo_id": "openai/whisper-small",
            "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
            "size_bytes": 4389142,
            "directory": "whisper-tokenizer",
            "files": [
                "README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                "normalizer.json", "preprocessor_config.json",
            ],
        },
        {
            "repo_id": "facebook/hubert-large-ll60k",
            "revision": "ff022d095678a2995f3c49bab18a96a9e553f782",
            "size_bytes": 1261924568,
            "directory": "hubert",
            "files": ["README.md", "config.json", "preprocessor_config.json", "pytorch_model.bin"],
        },
        {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "revision": "633ff708ed5b74903e86ff1298cf4a98e921c513",
            "size_bytes": 449238690,
            "directory": "bigvgan",
            "files": ["README.md", "LICENSE", "config.json", "bigvgan_generator.pt"],
        },
    ],
}

REQUIRED_MODEL_FILES = {
    "voxcpm": ["config.json", "tokenizer.json", "model.safetensors", "audiovae.pth"],
    "seedvc": [
        "main/v2/ar_base.pth", "main/v2/cfm_small.pth",
        "astral/bsq32/bsq32_light.pth", "astral/bsq2048/bsq2048_light.pth",
        "campplus/campplus_cn_common.bin", "whisper-tokenizer/tokenizer.json",
        "hubert/config.json", "hubert/preprocessor_config.json", "hubert/pytorch_model.bin",
        "bigvgan/config.json", "bigvgan/bigvgan_generator.pt",
    ],
}

REQUIRED_MODULES = {
    "voxcpm": ["torch", "torchaudio", "numpy", "soundfile", "voxcpm", "transformers", "librosa"],
    "seedvc": [
        "torch", "torchaudio", "numpy", "soundfile", "librosa", "transformers",
        "hydra", "omegaconf", "yaml", "einops", "munch", "pydub", "imageio_ffmpeg",
    ],
}
