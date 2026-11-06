from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvcre.service import VoiceService


@pytest.fixture
def tone(tmp_path):
    """Analytic signal for plumbing/pitch tests; never presented as model-generated speech."""
    path = tmp_path / "measurement_signal.wav"
    sr = 16000
    t = np.arange(sr * 5) / sr
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), sr)
    return path


class RecordingRunner:
    """Test double only. Production workers have no synthetic-audio fallback."""
    def __init__(self, source):
        self.source = source
        self.requests = []

    def status(self):
        return [{"name": "test_only", "ready": True, "details": []}]

    def run(self, backend, request, job_dir, cancel_event=None):
        import shutil
        self.requests.append((backend, request))
        outputs = []
        for i in range(request.get("candidates", 1)):
            dest = Path(job_dir) / f"test_output_{i}.wav"
            shutil.copyfile(self.source, dest)
            outputs.append({"path": str(dest), "seed": request.get("seed", 42) + i})
        return {"outputs": outputs, "metadata": {"test_double": True}}


@pytest.fixture
def service(tmp_path, tone):
    svc = VoiceService(tmp_path / "workspace", runner=RecordingRunner(tone))
    yield svc
    svc.close()
