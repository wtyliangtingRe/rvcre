"""Split narration at natural boundaries while retaining every spoken character."""

import re


TTS_SEGMENT_CHARS = 600
NARRATION_SEGMENT_CHARS = 240
NARRATION_ANCHOR_CHARS = 80
_PARAGRAPH = re.compile(r"\n[ \t]*\n+")
_LINE = re.compile(r"\n")
_SENTENCE = re.compile(r'''(?:[。！？!?]+|\.(?=\s|$|[”’"']))[。！？!?….”’」』）》】"']*''')
_CLAUSE = re.compile(r"[，,；;：:、]+[”’」』）》】\"']*")
_SPACE = re.compile(r"\s+")


def split_tts_text(text: str, max_chars: int = TTS_SEGMENT_CHARS) -> list[str]:
    """Prefer paragraphs, then sentences, clauses and spaces before a hard cut.

    The limit counts Python Unicode characters, including punctuation and spaces,
    just like the existing short-text validator. Only boundary whitespace is
    removed; no punctuation, numbers or words are rewritten.
    """
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("每段字符上限必须为正整数。")
    remaining = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks = []
    while len(remaining) > max_chars:
        cut = 0
        # Include one lookahead character so a decimal point at the boundary
        # is not mistaken for the end of a sentence.
        window = remaining[:max_chars + 1]
        for pattern in (_PARAGRAPH, _LINE, _SENTENCE, _CLAUSE, _SPACE):
            boundaries = [match.end() for match in pattern.finditer(window)
                          if match.end() <= max_chars and remaining[:match.end()].strip()]
            if boundaries:
                cut = boundaries[-1]
                break
        cut = cut or max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def plan_tts_segments(text: str, narration: bool = False) -> list[str]:
    """Plan an optional short, fixed opening prompt followed by bounded clips.

    The opening is part of the requested narration, not extra calibration text.
    Its entire audio and matching text are reused, avoiding guessed alignment
    between an arbitrary audio tail and a different text suffix.
    """
    if not isinstance(narration, bool):
        raise ValueError("首段定调选项必须为布尔值。")
    if not narration:
        return split_tts_text(text)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) <= NARRATION_SEGMENT_CHARS:
        return split_tts_text(normalized, NARRATION_SEGMENT_CHARS)
    # Avoid anchoring a whole article on a few-character title or greeting.
    # Prefer a natural boundary in the latter half of the opening window;
    # earlier newlines remain inside the prompt instead of ending it early.
    window = normalized[:NARRATION_ANCHOR_CHARS + 1]
    cut = NARRATION_ANCHOR_CHARS
    for pattern in (_PARAGRAPH, _LINE, _SENTENCE, _CLAUSE, _SPACE):
        boundaries = [match.end() for match in pattern.finditer(window)
                      if NARRATION_ANCHOR_CHARS // 2 <= match.end() <= NARRATION_ANCHOR_CHARS
                      and len(normalized[:match.end()].strip()) >= NARRATION_ANCHOR_CHARS // 2]
        if boundaries:
            cut = boundaries[-1]
            break
    opening = normalized[:cut].strip()
    remainder = normalized[len(opening):].lstrip()
    return [opening, *split_tts_text(remainder, NARRATION_SEGMENT_CHARS)]
