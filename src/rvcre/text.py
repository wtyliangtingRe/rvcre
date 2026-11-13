"""Split narration at natural boundaries while retaining every spoken character."""

import re


TTS_SEGMENT_CHARS = 600
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
