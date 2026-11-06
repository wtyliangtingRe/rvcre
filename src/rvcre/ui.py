"""Local Chinese Gradio interface. All audio work is delegated to VoiceService.

The UI stores job IDs in per-session State components. A Timer reads real job
records; it never substitutes example audio when a backend is unavailable.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gradio as gr

if TYPE_CHECKING:
    from .service import VoiceService


_ACTIVE = {"queued", "running"}
_SPEC_KEYS = (
    "pitch", "resonance", "brightness", "breathiness", "roughness",
    "maturity", "presentation", "description",
)
_DEFAULT_SPEC = ("medium", "balanced", 0.0, 0.0, 0.0, "adult", "neutral", "")
_SAMPLE_TEXT = "你好，很高兴认识你。窗外的风轻轻吹过树梢，我们慢慢说，不着急。"
_CSS = """
.gradio-container { background: var(--body-background-fill); }
#rv-shell { max-width: 1260px; margin: 0 auto; padding: 22px 8px 36px; }
.rv-hero { display: flex; align-items: center; justify-content: space-between;
  gap: 28px; padding: 22px 2px 25px; }
.rv-eyebrow { color: #148577; font-size: 12px; font-weight: 700;
  letter-spacing: .13em; }
.rv-hero h1 { font-size: clamp(24px, 3vw, 34px); font-weight: 680;
  letter-spacing: -.025em; margin: 7px 0 10px; line-height: 1.35; }
.rv-hero p { color: var(--body-text-color-subdued); margin: 0; line-height: 1.7; }
.rv-wave { display: flex; align-items: center; gap: 7px; padding: 18px 30px;
  border: 1px solid var(--border-color-primary); border-radius: 22px;
  background: var(--background-fill-secondary); }
.rv-wave span { display: block; width: 6px; background: #20a28e; border-radius: 8px; }
.rv-wave span:nth-child(1), .rv-wave span:nth-child(7) { height: 13px; opacity: .45; }
.rv-wave span:nth-child(2), .rv-wave span:nth-child(6) { height: 26px; opacity: .7; }
.rv-wave span:nth-child(3), .rv-wave span:nth-child(5) { height: 39px; }
.rv-wave span:nth-child(4) { height: 53px; }
.rv-card { padding: 20px !important; border: 1px solid var(--border-color-primary);
  border-radius: 16px; background: var(--block-background-fill); }
.rv-status { min-height: 68px; padding: 14px 16px !important;
  border-left: 3px solid #20a28e; border-radius: 7px;
  background: var(--background-fill-secondary); }
.rv-muted { color: var(--body-text-color-subdued); font-size: 13px; }
#main-tabs > .tab-nav { margin-bottom: 18px; }
#main-tabs > .tab-nav button { font-size: 15px; padding: 12px 19px; }
#rv-shell .prose h3 { margin-top: 0; }
#rv-shell .prose table { font-size: 13px; }
@media (max-width: 700px) {
  #rv-shell { padding-top: 8px; }
  .rv-wave { display: none; }
  .rv-hero { padding: 16px 6px 22px; }
  .rv-card { padding: 14px !important; }
}
"""


def _escape(value: Any) -> str:
    """Keep voice names, notes and backend errors as text in Markdown outputs."""
    text = html.escape(str(value), quote=False)
    for char in ("\\", "`", "*", "_", "[", "]", "#", "|"):
        text = text.replace(char, "\\" + char)
    return text


def _job_message(job: dict) -> str:
    status = job.get("status", "unknown")
    labels = {
        "queued": "正在排队", "running": "正在生成", "succeeded": "生成完成",
        "failed": "任务未完成", "cancelled": "任务已取消",
    }
    title = labels.get(status, "正在读取任务")
    if job.get("cancel_requested") and status in _ACTIVE:
        title = "已请求取消，正在停止任务"
    detail = {
        "queued": "任务已提交，等待本地引擎处理。",
        "running": "本地引擎正在工作，结果会自动出现在这里。",
        "succeeded": f"已生成 {len(job.get('outputs', []))} 段音频，可以试听或下载。",
        "cancelled": "你可以调整设置后重新提交。",
    }.get(status, "")
    if status == "failed":
        detail = _escape(job.get("error") or "引擎没有返回有效结果，请检查下方的本地引擎状态。")
    identifier = _escape(job.get("id", ""))
    return f"**{title}**\n\n{detail}\n\n任务编号：`{identifier}`"


def _analysis_message(profile: dict) -> str:
    analysis = profile.get("analysis") or {}
    if not analysis:
        return "暂时没有参考片段的测量结果。"

    def number(key: str, unit: str = "", digits: int = 1) -> str:
        value = analysis.get(key)
        return "—" if value is None else f"{float(value):,.{digits}f}{unit}"

    low, high = analysis.get("pitch_p10_hz"), analysis.get("pitch_p90_hz")
    pitch_range = "—" if low is None or high is None else f"{low:.1f}–{high:.1f} Hz"
    rows = [
        ("参考时长", number("duration_seconds", " 秒")),
        ("采样率", number("sample_rate", " Hz", 0)),
        ("基频中位数", number("pitch_median_hz", " Hz")),
        ("基频范围（10%–90%）", pitch_range),
        ("均方根电平", number("rms_dbfs", " dBFS")),
    ]
    result = "| 参考片段测量 | 结果 |\n| :--- | :--- |\n"
    result += "\n".join(f"| {label} | {value} |" for label, value in rows)
    result += "\n\n这些统计描述这段录音；声音身份同时保存在参考音频中。"
    selection = profile.get("selection") or {}
    if selection.get("selection_method") == "energy_window":
        start = selection.get("reference_start_seconds", 0)
        end = selection.get("reference_end_seconds", 0)
        result += f"\n\n已选取原录音的 **{start:g}–{end:g} 秒**，请试听上方参考。"
        if profile.get("source_transcript") and not profile.get("transcript"):
            result += "整段转写未用于此片段；HiFi 请导入已剪好且转写对应的片段。"
    return result


def _design_view(job: dict, *, clear: bool = False) -> tuple:
    """Outputs: ID, message, timer, cancel, submit, 3 players, choice, save."""
    busy = job.get("status") in _ACTIVE
    outputs = job.get("outputs") or []
    completed = job.get("status") == "succeeded"
    players: list[dict] = []
    for index in range(3):
        if completed and index < len(outputs):
            players.append(gr.update(value=outputs[index]["path"], visible=True))
        elif clear or not busy:
            players.append(gr.update(value=None, visible=False))
        else:
            players.append(gr.skip())
    if completed:
        choices = [
            (f"候选 {index + 1}" + (f" · seed {item['seed']}" if item.get("seed") is not None else ""), index)
            for index, item in enumerate(outputs[:3])
        ]
        choice = gr.update(choices=choices, value=0 if choices else None, interactive=bool(choices))
    elif clear or not busy:
        choice = gr.update(choices=[], value=None, interactive=False)
    else:
        choice = gr.skip()
    return (
        job["id"], _job_message(job), gr.update(active=busy),
        gr.update(interactive=busy and not job.get("cancel_requested", False)),
        gr.update(interactive=not busy), *players, choice,
        gr.update(interactive=completed and bool(outputs)),
    )


def _generation_view(job: dict, *, clear: bool = False) -> tuple:
    """Outputs: ID, message, timer, cancel, TTS submit, VC submit, player."""
    busy = job.get("status") in _ACTIVE
    outputs = job.get("outputs") or []
    if job.get("status") == "succeeded" and outputs:
        player = gr.update(value=outputs[0]["path"], visible=True)
    elif clear or not busy:
        player = gr.update(value=None, visible=False)
    else:
        player = gr.skip()
    return (
        job["id"], _job_message(job), gr.update(active=busy),
        gr.update(interactive=busy and not job.get("cancel_requested", False)),
        gr.update(interactive=not busy), gr.update(interactive=not busy), player,
    )


class _UiActions:
    """Thin, testable translation between components and the service contract."""

    def __init__(self, service: VoiceService):
        self.service = service

    @staticmethod
    def _call(function: Callable, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(str(exc) or "操作未完成，请检查本地引擎状态。", title="无法完成操作") from exc

    def _voice_updates(self, preferred: str | None = None) -> tuple:
        voices = self._call(self.service.list_voices)
        choices = [(f"{v['name']} · v{v.get('revision', 1)}", v["id"]) for v in voices]
        valid = {value for _, value in choices}
        selected = preferred if preferred in valid else (choices[0][1] if choices else None)
        variants = [("创建全新声音", "")] + [
            (v["name"], v["id"]) for v in voices
        ]
        return (
            gr.update(choices=choices, value=selected),
            gr.update(choices=choices, value=selected),
            gr.update(choices=variants),
            f"**{len(voices)} 个声音身份** · 导入和设计的声音都保存在这里。",
        )

    def refresh_voices(self, selected: str | None = None) -> tuple:
        return self._voice_updates(selected)

    def import_voice(self, name: str, audio_path: str | None, transcript: str,
                     notes: str, source_note: str) -> tuple:
        if not audio_path:
            raise gr.Error("请先上传或录制一段参考声音。")
        if not name.strip():
            raise gr.Error("请为声音填写名称。")
        profile = self._call(
            self.service.import_voice, name=name.strip(), audio_path=audio_path,
            transcript=transcript.strip(), notes=notes.strip(), source_note=source_note.strip(),
            source_kind="reference",
        )
        return (f"已保存 **{_escape(profile['name'])}**，现在可以在「语音生成」中使用。",
                *self._voice_updates(profile["id"]))

    def voice_details(self, voice_id: str | None) -> tuple:
        if not voice_id:
            return "### 还没有选中声音\n\n导入一段参考录音，或去「声音设计」创建新声音。", None, "", {}
        profile = self._call(self.service.get_voice, voice_id)
        origin = "声音设计" if (profile.get("source") or {}).get("kind") == "designed" else "录音导入"
        description = f"### {_escape(profile['name'])}\n\n{origin} · 版本 {profile.get('revision', 1)}"
        if profile.get("notes"):
            description += "\n\n" + _escape(profile["notes"])
        if profile.get("transcript"):
            description += "\n\n**参考转写**\n\n" + _escape(profile["transcript"])
        else:
            description += "\n\n参考转写未设置，可使用可控表达模式和声音转换。"
        return description, profile.get("reference_path"), _analysis_message(profile), profile

    def selected_voice(self, voice_id: str | None) -> str:
        if not voice_id:
            return "先在声音库导入录音，或保存一个设计候选。"
        profile = self._call(self.service.get_voice, voice_id)
        modes = "可控表达 · HiFi · 声音转换" if profile.get("transcript") else "可控表达 · 声音转换"
        return f"当前身份：**{_escape(profile['name'])}** · v{profile.get('revision', 1)}\n\n可用方式：{modes}"

    def load_variant(self, voice_id: str | None) -> tuple:
        if not voice_id:
            return (*_DEFAULT_SPEC, None, "描述你希望听到的声音，再试听候选。")
        profile = self._call(self.service.get_voice, voice_id)
        spec = profile.get("design_spec") or {}
        values = tuple(spec.get(key, default) for key, default in zip(_SPEC_KEYS, _DEFAULT_SPEC))
        name = _escape(profile["name"])
        origin = profile.get("design_spec_origin")
        if origin == "pitch_hint_from_measurement":
            notice = (f"以 **{name}** 为设计起点。音高是粗略提示，其余是设计初值；"
                      "参考音色不会原样带入新设计。重新设计的候选无法保证只改变单一属性。")
        elif origin == "defaults_not_measured":
            notice = (f"以 **{name}** 为设计起点。以下是设计初值，并非从录音还原的音色参数；"
                      "参考音色不会原样带入新设计。重新设计的候选无法保证只改变单一属性。")
        else:
            notice = (f"已载入 **{name}** 的设计参数。重新设计候选，无法保证只改变单一属性；"
                      "试听后另存为新声音。")
        return (*values, voice_id, notice)

    def _ensure_idle(self, job_id: str | None) -> None:
        if job_id and self._call(self.service.get_job, job_id).get("status") in _ACTIVE:
            raise gr.Error("当前任务还在运行，请等待完成或先取消。")

    def start_design(self, job_id: str | None, text: str, pitch: str, resonance: str,
                     brightness: float, breathiness: float, roughness: float, maturity: str,
                     presentation: str, description: str, candidates: int, seed: int,
                     parent_voice_id: str | None) -> tuple:
        self._ensure_idle(job_id)
        spec = dict(zip(_SPEC_KEYS, (
            pitch, resonance, brightness, breathiness, roughness, maturity, presentation, description,
        )))
        if seed is None:
            raise gr.Error("请填写随机种子，例如 42。")
        job = self._call(self.service.submit_design, text=text, spec=spec,
                         candidates=int(candidates), seed=int(seed), parent_voice_id=parent_voice_id)
        return _design_view(job, clear=True)

    def save_candidate(self, job_id: str | None, index: int | None, name: str) -> tuple:
        if not job_id or index is None:
            raise gr.Error("请等待候选生成完成，再选择要保存的声音。")
        if not name.strip():
            raise gr.Error("请为选中的候选填写名称。")
        profile = self._call(self.service.save_candidate, job_id, int(index), name.strip())
        return (f"已将候选 {int(index) + 1} 保存为 **{_escape(profile['name'])}**，可用于 TTS 和 VC。",
                *self._voice_updates(profile["id"]))

    def start_tts(self, job_id: str | None, voice_id: str | None, text: str, mode: str,
                  speed: float, emotion: str, instruction: str, seed: int) -> tuple:
        self._ensure_idle(job_id)
        if not voice_id:
            raise gr.Error("请先选择一个声音身份。")
        if seed is None:
            raise gr.Error("请填写随机种子，例如 42。")
        # Hidden expression controls must not leak into HiFi requests.
        performance = None if mode == "hifi" else {
            "speed": float(speed), "emotion": emotion, "instruction": instruction,
        }
        job = self._call(self.service.submit_tts, voice_id=voice_id, text=text,
                         performance=performance, mode=mode, seed=int(seed))
        return _generation_view(job, clear=True)

    def start_vc(self, job_id: str | None, voice_id: str | None, source_path: str | None,
                 steps: int, similarity: float, intelligibility: float) -> tuple:
        self._ensure_idle(job_id)
        if not voice_id:
            raise gr.Error("请先选择一个声音身份。")
        if not source_path:
            raise gr.Error("请先上传或录制需要转换的源音频。")
        settings = {"diffusion_steps": int(steps), "similarity_cfg_rate": float(similarity),
                    "intelligibility_cfg_rate": float(intelligibility)}
        job = self._call(self.service.submit_vc, voice_id=voice_id, source_path=source_path, settings=settings)
        return _generation_view(job, clear=True)

    def poll_design(self, job_id: str | None) -> tuple:
        if not job_id:
            return (gr.skip(), gr.skip(), gr.update(active=False), *[gr.skip() for _ in range(7)])
        return _design_view(self._call(self.service.get_job, job_id))

    def poll_generation(self, job_id: str | None) -> tuple:
        if not job_id:
            return (gr.skip(), gr.skip(), gr.update(active=False), *[gr.skip() for _ in range(4)])
        return _generation_view(self._call(self.service.get_job, job_id))

    def cancel_design(self, job_id: str | None) -> tuple:
        if not job_id:
            return self.poll_design(None)
        return _design_view(self._call(self.service.cancel_job, job_id))

    def cancel_generation(self, job_id: str | None) -> tuple:
        if not job_id:
            return self.poll_generation(None)
        return _generation_view(self._call(self.service.cancel_job, job_id))

    def backend_status(self) -> str:
        try:
            backends = self.service.backend_status()
        except Exception as exc:
            return "引擎状态暂不可用：" + _escape(exc)
        if not backends:
            return "尚未配置本地引擎。请先按项目 README 准备模型并设置 `config/backends.json`。"
        rows = []
        for backend in backends:
            name = backend.get("name") or backend.get("id") or backend.get("backend") or "本地引擎"
            name = {"voxcpm": "VoxCPM2 · 设计 / TTS", "seedvc": "Seed-VC2 · VC"}.get(name, name)
            ready = backend.get("ready", backend.get("available"))
            if backend.get("enabled") is False:
                status = "未启用"
            elif ready is True:
                status = "依赖就绪"
            elif ready is False:
                status = "待准备"
            else:
                status = backend.get("status") or "待检查"
            detail = (backend.get("message") or backend.get("reason") or backend.get("error")
                      or backend.get("details") or "")
            if isinstance(detail, list):
                detail = "；".join(str(item) for item in detail)
            if ready is True:
                detail = "路径与依赖检查通过；模型会在首次生成时加载。"
            elif backend.get("configured") is False:
                detail = "尚未配置。按 README 准备此后端；" + str(detail)
            rows.append(f"| {_escape(name)} | {_escape(status)} | {_escape(detail).replace(chr(10), ' ')} |")
        return "| 引擎 | 状态 | 说明 |\n| :--- | :--- | :--- |\n" + "\n".join(rows)


def _mode_controls(mode: str) -> tuple:
    hifi = mode == "hifi"
    message = ("HiFi 使用参考片段及其准确转写；表达跟随参考。" if hifi else
               "按表达指令合成。语速数值表示倾向，不是精确的播放倍速。")
    return gr.update(visible=not hifi), message


def build_ui(service: VoiceService) -> gr.Blocks:
    """Build the three workflows without starting servers or loading speech models."""
    actions = _UiActions(service)
    private = {"api_visibility": "private"}
    with gr.Blocks(title="RVCre · 本地声音工作室", analytics_enabled=False) as app:
        design_job = gr.State(None)
        generation_job = gr.State(None)
        parent_voice = gr.State(None)
        design_timer = gr.Timer(1, active=False)
        generation_timer = gr.Timer(1, active=False)
        with gr.Column(elem_id="rv-shell"):
            gr.HTML(
                '<header class="rv-hero"><div><div class="rv-eyebrow">RVCre / 本地声音工作室</div>'
                '<h1>一种声音，多种表达。</h1><p>从参考录音或一个想法开始，保存声音，再让文字和录音用它表达。</p>'
                '</div><div class="rv-wave" aria-hidden="true">'
                '<span></span><span></span><span></span><span></span><span></span><span></span><span></span>'
                '</div></header>'
            )
            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("01  声音库", id="library"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, min_width=300, elem_classes="rv-card"):
                            gr.Markdown("### 导入一个声音\n\n建议使用 5–12 秒清晰、单人的干声。原录音也会保留。")
                            import_name = gr.Textbox(label="声音名称", placeholder="例如：青柚 · 自然声线")
                            import_audio = gr.Audio(label="参考录音", type="filepath", sources=["upload", "microphone"])
                            import_transcript = gr.Textbox(label="参考转写（可选）", lines=3,
                                placeholder="准确填写录音说了什么，便于使用 HiFi。长音频会自动选取参考片段。")
                            with gr.Accordion("备注与来源", open=False):
                                import_notes = gr.Textbox(label="声音备注", lines=2, placeholder="记录声音特点或用途")
                                import_source = gr.Textbox(label="来源说明", placeholder="例如：自己的录音，2026-11-06")
                            import_button = gr.Button("保存到声音库", variant="primary")
                            import_status = gr.Markdown("保存时会分析参考片段，首次分析可能需要稍等。", elem_classes="rv-muted")
                        with gr.Column(scale=6, min_width=320, elem_classes="rv-card"):
                            with gr.Row():
                                library_voice = gr.Dropdown(choices=[], value=None, label="我的声音", scale=5, interactive=True)
                                refresh_voices = gr.Button("刷新", scale=1, min_width=70)
                            voice_count = gr.Markdown("正在读取声音库…", elem_classes="rv-muted")
                            voice_details = gr.Markdown("### 选中声音后，在这里试听参考。")
                            reference_audio = gr.Audio(label="保存的参考片段", interactive=False, type="filepath", buttons=["download"])
                            analysis = gr.Markdown()
                            with gr.Accordion("完整声音档案", open=False):
                                profile_json = gr.JSON(label="声音档案")

                with gr.Tab("02  声音设计", id="design"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, min_width=320, elem_classes="rv-card"):
                            gr.Markdown("### 描述你想听到的声音")
                            variant = gr.Dropdown(choices=[("创建全新声音", "")], value="", label="设计起点", interactive=True)
                            variant_notice = gr.Markdown("描述你希望听到的声音，再试听候选。", elem_classes="rv-muted")
                            description = gr.Textbox(label="声音描述", lines=3,
                                placeholder="例如：清晰温暖的成年女声，轻盈自然，适合安静地讲故事。")
                            with gr.Row():
                                pitch = gr.Dropdown([("偏低", "low"), ("适中", "medium"), ("偏高", "high")], value="medium", label="音高倾向")
                                resonance = gr.Dropdown([("轻盈", "light"), ("均衡", "balanced"), ("饱满", "full")], value="balanced", label="共鸣倾向")
                            with gr.Row():
                                maturity = gr.Dropdown([("年轻成人", "young_adult"), ("成人", "adult"), ("成熟", "mature")], value="adult", label="成熟感")
                                presentation = gr.Dropdown([("中性", "neutral"), ("偏女性", "feminine"), ("偏男性", "masculine")], value="neutral", label="声音表现")
                            with gr.Accordion("音色细调", open=False):
                                brightness = gr.Slider(-1, 1, value=0, step=0.1, label="明亮倾向 · 柔暗 ↔ 明亮")
                                breathiness = gr.Slider(0, 1, value=0, step=0.05, label="气声倾向")
                                roughness = gr.Slider(0, 1, value=0, step=0.05, label="沙哑倾向")
                                gr.Markdown("这些是设计指令的感知强度，不是录音测得的物理参数；以试听结果为准。", elem_classes="rv-muted")
                            design_text = gr.Textbox(value=_SAMPLE_TEXT, label="试听文本", lines=3, max_lines=5,
                                info="单次最多 240 个字符。短句更便于比较候选。")
                            with gr.Row():
                                candidates = gr.Slider(1, 3, value=3, step=1, precision=0, label="候选数量")
                                design_seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32 - 1, label="随机种子")
                            design_submit = gr.Button("生成声音候选", variant="primary")
                        with gr.Column(scale=6, min_width=320, elem_classes="rv-card"):
                            gr.Markdown("### 试听，再留下喜欢的声音")
                            design_status = gr.Markdown("候选生成后会出现在这里。保存后，同一声音可用于 TTS 和 VC。", elem_classes="rv-status")
                            with gr.Row():
                                design_cancel = gr.Button("取消任务", interactive=False, variant="stop")
                                design_refresh = gr.Button("刷新任务状态")
                            candidate_players = [
                                gr.Audio(label=f"候选 {i + 1}", type="filepath", interactive=False,
                                         visible=False, buttons=["download"])
                                for i in range(3)
                            ]
                            selected_candidate = gr.Dropdown(choices=[], value=None, label="要保存的候选", interactive=False)
                            candidate_name = gr.Textbox(label="为声音命名", placeholder="例如：青柚 · 温暖叙述")
                            candidate_save = gr.Button("保存选中声音", interactive=False, variant="primary")
                            candidate_save_status = gr.Markdown()

                with gr.Tab("03  语音生成", id="generation"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=6, min_width=320, elem_classes="rv-card"):
                            generation_voice = gr.Dropdown(choices=[], value=None, label="使用哪个声音", interactive=True)
                            selected_voice = gr.Markdown("先在声音库导入录音，或保存一个设计候选。", elem_classes="rv-muted")
                            with gr.Tabs():
                                with gr.Tab("文字转语音 · TTS"):
                                    tts_text = gr.Textbox(label="要说的话", lines=6, placeholder="在这里写下想让这个声音说的话。", info="单次最多 600 个字符；长篇文本请按段提交。")
                                    tts_mode = gr.Radio([("可控表达", "controllable"), ("HiFi 参考复现", "hifi")], value="controllable", label="合成方式")
                                    mode_note = gr.Markdown(_mode_controls("controllable")[1], elem_classes="rv-muted")
                                    with gr.Column() as expression_controls:
                                        with gr.Row():
                                            speed = gr.Slider(0.7, 1.4, value=1, step=0.05, label="语速倾向（指令控制）")
                                            emotion = gr.Dropdown([
                                                ("自然", "neutral"), ("愉快", "happy"), ("悲伤", "sad"),
                                                ("生气", "angry"), ("平静", "calm"), ("兴奋", "excited"),
                                            ], value="neutral", label="情绪")
                                        instruction = gr.Textbox(label="表达补充（可选）", placeholder="例如：像朋友聊天，句末稍停顿。")
                                    with gr.Accordion("生成选项", open=False):
                                        tts_seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32 - 1, label="随机种子")
                                    tts_submit = gr.Button("生成语音", variant="primary")
                                with gr.Tab("录音换声音 · VC"):
                                    gr.Markdown("上传原始表演，转换为所选声音身份。")
                                    source_audio = gr.Audio(label="要转换的源录音", sources=["upload", "microphone"], type="filepath")
                                    with gr.Accordion("转换选项", open=False):
                                        vc_steps = gr.Slider(1, 100, value=30, step=1, precision=0, label="扩散步数")
                                        vc_similarity = gr.Slider(0, 1, value=0.7, step=0.05, label="音色相似度引导")
                                        vc_intelligibility = gr.Slider(0, 1, value=0.7, step=0.05, label="可懂度引导")
                                        gr.Markdown("引导强度不是百分比评分。首次使用建议保持默认值。", elem_classes="rv-muted")
                                    vc_submit = gr.Button("转换声音", variant="primary")
                        with gr.Column(scale=5, min_width=320, elem_classes="rv-card"):
                            gr.Markdown("### 生成结果")
                            generation_status = gr.Markdown("选择声音，填写文字或上传录音后开始。", elem_classes="rv-status")
                            with gr.Row():
                                generation_cancel = gr.Button("取消任务", interactive=False, variant="stop")
                                generation_refresh = gr.Button("刷新任务状态")
                            generated_audio = gr.Audio(label="结果音频", type="filepath", interactive=False,
                                                       visible=False, buttons=["download"])

            with gr.Accordion("本地引擎与模型准备", open=False):
                gr.Markdown("模型权重与依赖准备好后可离线运行。界面不会自动下载模型。")
                backend_status = gr.Markdown("正在检查本地引擎…")
                backend_refresh = gr.Button("检查引擎状态", size="sm")
                gr.Markdown("安装、后端配置与可选训练路线见项目 README。声音导入和参考分析无需加载语音生成模型。", elem_classes="rv-muted")

        voice_outputs = [library_voice, generation_voice, variant, voice_count]
        design_outputs = [design_job, design_status, design_timer, design_cancel, design_submit,
                          *candidate_players, selected_candidate, candidate_save]
        generation_outputs = [generation_job, generation_status, generation_timer, generation_cancel,
                              tts_submit, vc_submit, generated_audio]
        spec_components = [pitch, resonance, brightness, breathiness, roughness, maturity, presentation, description]

        app.load(actions.refresh_voices, inputs=[library_voice], outputs=voice_outputs, **private)
        app.load(actions.backend_status, outputs=[backend_status], **private)
        refresh_voices.click(actions.refresh_voices, inputs=[library_voice], outputs=voice_outputs, **private)
        library_voice.change(actions.voice_details, inputs=[library_voice],
                             outputs=[voice_details, reference_audio, analysis, profile_json], **private)
        library_voice.input(lambda value: gr.update(value=value), inputs=[library_voice], outputs=[generation_voice], **private)
        generation_voice.change(actions.selected_voice, inputs=[generation_voice], outputs=[selected_voice], **private)
        import_button.click(actions.import_voice,
                            inputs=[import_name, import_audio, import_transcript, import_notes, import_source],
                            outputs=[import_status, *voice_outputs], concurrency_limit=1, **private)
        variant.change(actions.load_variant, inputs=[variant], outputs=[*spec_components, parent_voice, variant_notice], **private)
        design_submit.click(actions.start_design,
                            inputs=[design_job, design_text, *spec_components, candidates, design_seed, parent_voice],
                            outputs=design_outputs, concurrency_id="design-submit", concurrency_limit=1, **private)
        design_timer.tick(actions.poll_design, inputs=[design_job], outputs=design_outputs,
                          queue=False, show_progress="hidden", **private)
        design_refresh.click(actions.poll_design, inputs=[design_job], outputs=design_outputs,
                             queue=False, show_progress="hidden", **private)
        design_cancel.click(actions.cancel_design, inputs=[design_job], outputs=design_outputs,
                            queue=False, **private)
        candidate_save.click(actions.save_candidate, inputs=[design_job, selected_candidate, candidate_name],
                             outputs=[candidate_save_status, *voice_outputs], concurrency_limit=1, **private)
        tts_mode.change(_mode_controls, inputs=[tts_mode], outputs=[expression_controls, mode_note], **private)
        tts_submit.click(actions.start_tts,
                         inputs=[generation_job, generation_voice, tts_text, tts_mode, speed, emotion, instruction, tts_seed],
                         outputs=generation_outputs, concurrency_id="generation-submit", concurrency_limit=1, **private)
        vc_submit.click(actions.start_vc,
                        inputs=[generation_job, generation_voice, source_audio, vc_steps, vc_similarity, vc_intelligibility],
                        outputs=generation_outputs, concurrency_id="generation-submit", concurrency_limit=1, **private)
        generation_timer.tick(actions.poll_generation, inputs=[generation_job], outputs=generation_outputs,
                              queue=False, show_progress="hidden", **private)
        generation_refresh.click(actions.poll_generation, inputs=[generation_job], outputs=generation_outputs,
                                 queue=False, show_progress="hidden", **private)
        generation_cancel.click(actions.cancel_generation, inputs=[generation_job], outputs=generation_outputs,
                                queue=False, **private)
        backend_refresh.click(actions.backend_status, outputs=[backend_status], **private)
    return app


def launch_ui(workspace: Path, config_path: Path | None = None, host: str = "127.0.0.1",
              port: int = 7860, open_browser: bool = True) -> None:
    """Launch a local-only-by-default UI, exposing only reference and result audio."""
    from .service import VoiceService

    service = VoiceService(Path(workspace), config_path)
    theme = gr.themes.Soft(
        primary_hue="teal", secondary_hue="slate", neutral_hue="slate", radius_size="lg",
        font=["system-ui", "Segoe UI", "Microsoft YaHei", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(button_primary_background_fill="#158674", button_primary_background_fill_hover="#0e7162")
    app = build_ui(service)
    try:
        app.queue(default_concurrency_limit=4).launch(
            server_name=host, server_port=port, inbrowser=open_browser, share=False,
            theme=theme, css=_CSS, show_error=True, footer_links=[], run_history=False,
            allowed_paths=[str(service.workspace / "voices"), str(service.workspace / "jobs")],
        )
    finally:
        service.close()
