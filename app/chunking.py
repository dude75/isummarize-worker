"""Нарезка длинного транскрипта и сборка prompt (map-reduce)."""

from __future__ import annotations

MAX_PROMPT_CHARS = 100_000
CHUNK_OVERLAP_CHARS = 400

USER_SINGLE = (
    "Ниже транскрипт. Следуй правилам из system. "
    "Верни только результат саммари, без преамбулы.\n\n"
    "<transcript>\n{text}\n</transcript>"
)

USER_CHUNK = (
    "Ниже часть {index} из {total} транскрипта. Следуй правилам из system. "
    "Верни только результат саммари этой части, без преамбулы.\n\n"
    "<transcript>\n{text}\n</transcript>"
)

USER_REDUCE = (
    "Ниже промежуточные саммари частей транскрипта. Следуй правилам из system. "
    "Верни только итоговый результат саммари, без преамбулы.\n\n"
    "{summaries}"
)


def user_single(text: str) -> str:
    return USER_SINGLE.format(text=text)


def user_chunk(text: str, index: int, total: int) -> str:
    return USER_CHUNK.format(text=text, index=index, total=total)


def user_reduce(summaries: list[str]) -> str:
    body = "\n\n".join(
        f"## Часть {index} из {len(summaries)}\n{item}"
        for index, item in enumerate(summaries, start=1)
    )
    return USER_REDUCE.format(summaries=body)


def prompt_chars(system: str, user: str) -> int:
    return len(system) + len(user)


def fits(system: str, user: str, limit: int | None = None) -> bool:
    budget = MAX_PROMPT_CHARS if limit is None else limit
    return prompt_chars(system, user) <= budget


def _overlap_for(max_chars: int) -> int:
    if max_chars <= 0:
        return 0
    return min(CHUNK_OVERLAP_CHARS, max(0, max_chars // 10))


def _units(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text] if text else []
    units: list[str] = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
            continue
        for line in para.split("\n"):
            if len(line) <= max_chars:
                units.append(line)
                continue
            for offset in range(0, len(line), max_chars):
                units.append(line[offset : offset + max_chars])
    return units


def _pack(units: list[str], max_chars: int) -> list[str]:
    if not units:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    sep = "\n\n"
    for unit in units:
        extra = len(unit) + (len(sep) if buf else 0)
        if buf and size + extra > max_chars:
            chunks.append(sep.join(buf))
            buf = [unit]
            size = len(unit)
        else:
            buf.append(unit)
            size += extra
    if buf:
        chunks.append(sep.join(buf))
    return chunks


def _with_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, current in zip(chunks[:-1], chunks[1:], strict=True):
        prefix = prev[-overlap:] if len(prev) >= overlap else prev
        if current.startswith(prefix):
            out.append(current)
        else:
            out.append(prefix + current)
    return out


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []
    if len(text) <= max_chars:
        return [text]
    overlap = _overlap_for(max_chars)
    packed_limit = max(1, max_chars - overlap)
    packed = _pack(_units(text, packed_limit), packed_limit)
    return _with_overlap(packed, overlap)


def max_text_chars(system: str, template: str, **sample: object) -> int:
    sample_kwargs = dict(sample)
    sample_kwargs.setdefault("text", "")
    overhead = prompt_chars(system, template.format(**sample_kwargs))
    return MAX_PROMPT_CHARS - overhead


def plan_chunks(system: str, text: str) -> list[str] | None:
    """None = один вызов на весь текст. Иначе список чанков. [] = text_too_long."""
    if fits(system, user_single(text)):
        return None
    budget = max_text_chars(system, USER_CHUNK, index=9999, total=9999, text="")
    if budget <= 0:
        return []
    chunks = split_text(text, budget)
    if not chunks:
        return []
    for index, chunk in enumerate(chunks, start=1):
        if not fits(system, user_chunk(chunk, index, len(chunks))):
            return []
    return chunks
