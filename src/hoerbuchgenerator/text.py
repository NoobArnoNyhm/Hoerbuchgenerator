import re


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    words = segment.split()
    pieces: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[index:index + max_chars] for index in range(0, len(word), max_chars))
            continue

        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    lines: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
        elif lines:
            paragraphs.append(" ".join(lines))
            lines = []

    if lines:
        paragraphs.append(" ".join(lines))
    return paragraphs


def chunk_text(text: str, max_chars: int = 600, min_chars: int = 5) -> list[str]:
    if max_chars < 1 or min_chars < 0 or min_chars > max_chars:
        raise ValueError("Ungueltige Chunk-Grenzen")

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise ValueError("Der Text darf nicht leer sein")

    segments: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            segments.extend(_split_long_segment(sentence.strip(), max_chars))

    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current} {segment}" if current else segment
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = segment
    if current:
        chunks.append(current)

    index = 0
    while index < len(chunks) - 1:
        if len(chunks[index]) < min_chars and len(chunks[index]) + 1 + len(chunks[index + 1]) <= max_chars:
            chunks[index:index + 2] = [f"{chunks[index]} {chunks[index + 1]}"]
        else:
            index += 1

    return chunks
