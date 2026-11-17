from bisect import bisect_left, bisect_right
import math

from hoerbuchgenerator.audio_types import AudioError, WordInterval


def select_reference_window(
    words: list[WordInterval],
    min_seconds: float = 20,
    target_seconds: float = 30,
    max_seconds: float = 40,
) -> tuple[float, float]:
    if not 0 < min_seconds <= target_seconds <= max_seconds:
        raise ValueError("Ungueltige Dauergrenzen fuer das Referenzsample")
    if not words:
        raise AudioError("Whisper hat keine Wortzeitstempel geliefert")

    previous_start = -1.0
    previous_end = -1.0
    for word in words:
        if (
            not math.isfinite(word.start)
            or not math.isfinite(word.end)
            or word.start < 0
            or word.end <= word.start
            or word.start < previous_start
            or word.end < previous_end
        ):
            raise AudioError("Whisper-Wortzeitstempel sind ungueltig oder unsortiert")
        previous_start, previous_end = word.start, word.end

    ends = [word.end for word in words]
    best_window: tuple[float, float] | None = None
    best_score: tuple[float, float] | None = None
    shortest_fallback: tuple[float, float] | None = None
    longest_fallback_duration = -1.0

    for start_index, word in enumerate(words):
        start = word.start
        last_index = bisect_right(ends, start + max_seconds, lo=start_index)
        if last_index == start_index:
            continue
        last_index -= 1

        minimum_index = bisect_left(
            ends,
            start + min_seconds,
            lo=start_index,
            hi=last_index + 1,
        )
        if minimum_index <= last_index:
            target_index = bisect_left(
                ends,
                start + target_seconds,
                lo=minimum_index,
                hi=last_index + 1,
            )
            for end_index in {minimum_index, last_index, target_index - 1, target_index}:
                if minimum_index <= end_index <= last_index:
                    end = ends[end_index]
                    score = (abs(end - start - target_seconds), -(end - start))
                    if best_score is None or score < best_score:
                        best_window = (start, end)
                        best_score = score

        fallback_end = ends[last_index]
        fallback_duration = fallback_end - start
        if fallback_duration > longest_fallback_duration:
            shortest_fallback = (start, fallback_end)
            longest_fallback_duration = fallback_duration

    if best_window is not None:
        return best_window
    if shortest_fallback is not None:
        return shortest_fallback
    raise AudioError("Kein vollstaendiges Wort passt in das Referenzsample")
