"""One local generation queue, persisted as readable files."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from filelock import FileLock, Timeout

from .storage import item_path, read_json, timestamp, write_json


class JobManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rvcre")
        self.events: dict[str, threading.Event] = {}
        self.futures = {}
        self.closed = False
        self.process_lock = FileLock(self.root / ".queue.lock", thread_local=False)
        self.owns_queue = False

    def _take_queue(self):
        if self.owns_queue:
            return
        try:
            self.process_lock.acquire(timeout=0)
        except Timeout as exc:
            raise ValueError("另一个 RVCre 进程正在使用此任务队列，请通过该界面提交，或使用其他工作目录。") from exc
        self.owns_queue = True
        for path in self.root.glob("*/job.json"):
            job = read_json(path)
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="上次运行中断，请重新提交任务。", finished_at=timestamp())
                write_json(path, job)

    def get(self, identifier: str) -> dict:
        with self.lock:
            return read_json(item_path(self.root, identifier, "job.json"))

    def list(self) -> list[dict]:
        with self.lock:
            return sorted(
                (read_json(p) for p in self.root.glob("*/job.json")),
                key=lambda j: j["created_at"], reverse=True,
            )

    def _update(self, identifier: str, **changes) -> dict:
        with self.lock:
            path = item_path(self.root, identifier, "job.json")
            job = read_json(path)
            job.update(changes)
            write_json(path, job)
            return job

    def submit(self, task: str, request: dict, handler: Callable) -> dict:
        identifier = uuid.uuid4().hex[:12]
        directory = self.root / identifier
        job = {
            "id": identifier, "task": task, "status": "queued", "created_at": timestamp(),
            "started_at": None, "finished_at": None, "outputs": [], "metadata": {}, "error": None,
        }
        event = threading.Event()
        with self.lock:
            if self.closed:
                raise RuntimeError("任务队列已关闭，请重新启动 RVCre。")
            self._take_queue()
            write_json(directory / "request.json", request)
            write_json(directory / "job.json", job)
            self.events[identifier] = event
            self.futures[identifier] = self.executor.submit(self._run, identifier, handler, event)
        return job

    def _run(self, identifier: str, handler: Callable, event: threading.Event) -> None:
        if event.is_set():
            self._update(identifier, status="cancelled", finished_at=timestamp())
            return
        self._update(identifier, status="running", started_at=timestamp())
        try:
            result = handler(self.root / identifier, event)
            if event.is_set():
                self._update(identifier, status="cancelled", finished_at=timestamp())
            else:
                self._update(identifier, status="succeeded", outputs=result["outputs"],
                             metadata=result.get("metadata", {}), finished_at=timestamp())
        except Exception as exc:
            self._update(identifier, status="cancelled" if event.is_set() else "failed",
                         error=None if event.is_set() else str(exc), finished_at=timestamp())

    def cancel(self, identifier: str) -> dict:
        with self.lock:
            job = self.get(identifier)
            if job["status"] not in ("queued", "running"):
                return job
            if identifier not in self.events:
                raise ValueError("请在启动此任务的 RVCre 界面或 API 中取消任务。")
            self.events[identifier].set()
            if self.futures[identifier].cancel():
                return self._update(identifier, status="cancelled", finished_at=timestamp())
            return self._update(identifier, cancel_requested=True)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for identifier, event in self.events.items():
                event.set()
                if self.futures[identifier].cancel():
                    self._update(identifier, status="cancelled", finished_at=timestamp())
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.owns_queue:
            self.process_lock.release()
            self.owns_queue = False
