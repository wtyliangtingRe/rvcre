"""Public user controls. Semantic strengths are design requests, not physical measurements."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitch: Literal["low", "medium", "high"] = "medium"
    resonance: Literal["light", "balanced", "full"] = "balanced"
    brightness: float = Field(default=0, ge=-1, le=1)
    breathiness: float = Field(default=0, ge=0, le=1)
    roughness: float = Field(default=0, ge=0, le=1)
    maturity: Literal["young_adult", "adult", "mature"] = "adult"
    presentation: Literal["neutral", "feminine", "masculine"] = "neutral"
    description: str = Field(default="", max_length=600)


class Performance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speed: float = Field(default=1.0, ge=0.7, le=1.4)
    emotion: Literal["neutral", "happy", "sad", "angry", "calm", "excited"] = "neutral"
    instruction: str = Field(default="", max_length=400)


def design_instruction(spec: VoiceSpec) -> str:
    """Compile a small vocabulary of requested attributes; never imply exact independent axes."""
    words = [
        {"young_adult": "年轻成年人的声音", "adult": "成年人的声音", "mature": "成熟的成年声线"}[spec.maturity],
        {"low": "音调偏低", "medium": "中等音调", "high": "音调偏高"}[spec.pitch],
        {"light": "声音轻盈", "balanced": "共鸣均衡", "full": "声音饱满厚实"}[spec.resonance],
    ]
    if spec.presentation != "neutral":
        words.append({"feminine": "偏女性的声音表现", "masculine": "偏男性的声音表现"}[spec.presentation])
    if spec.brightness > 0.25:
        words.append("非常明亮清透" if spec.brightness > 0.7 else "略微明亮")
    elif spec.brightness < -0.25:
        words.append("深沉柔暗" if spec.brightness < -0.7 else "略微柔暗")
    if spec.breathiness > 0.15:
        words.append("明显的气声" if spec.breathiness > 0.65 else "轻微气声")
    if spec.roughness > 0.15:
        words.append("明显的沙哑质感" if spec.roughness > 0.65 else "轻微沙哑质感")
    if spec.description.strip():
        words.append(spec.description.strip())
    return "，".join(words)


def performance_instruction(performance: Performance) -> str:
    words = []
    if performance.speed < 0.9:
        words.append("语速较慢")
    elif performance.speed > 1.1:
        words.append("语速较快")
    if performance.emotion != "neutral":
        words.append({
            "happy": "愉快的语气", "sad": "悲伤的语气", "angry": "愤怒的语气",
            "calm": "平静的语气", "excited": "兴奋的语气",
        }[performance.emotion])
    if performance.instruction.strip():
        words.append(performance.instruction.strip())
    return "，".join(words)
