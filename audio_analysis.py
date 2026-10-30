from pathlib import Path
import numpy as np
import soundfile as sf

def analyze_wav(file_path):
    audio, sample_rate = sf.read(str(file_path), always_2d=True)
    if audio.size == 0:
        raise ValueError("The audio file is empty.")
    mono = audio.mean(axis=1).astype(np.float32)
    duration = len(mono) / sample_rate
    frame_size = max(1, int(sample_rate * 0.5))  # half-second windows
    rms_values = []
    for start in range(0, len(mono), frame_size):
        chunk = mono[start:start + frame_size]
        if len(chunk):
            rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
    rms = np.asarray(rms_values)
    # Relative threshold: distinguish quieter intervals from louder intervals.
    threshold = max(0.008, float(np.percentile(rms, 35)) * 1.6)
    active_ratio = float(np.mean(rms > threshold)) if len(rms) else 0.0
    silence_ratio = 1.0 - active_ratio
    mean_rms = float(np.mean(rms)) if len(rms) else 0.0
    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": int(sample_rate),
        "mean_rms": round(mean_rms, 5),
        "sound_activity_percent": round(active_ratio * 100, 1),
        "low_activity_percent": round(silence_ratio * 100, 1),
        "windows": rms.tolist(),
        "threshold": threshold
    }

def suggestions(report):
    activity = report["sound_activity_percent"]
    if activity > 75:
        return [
            "The recording contains frequent sound activity. Check whether it reflects discussion, group work, or background noise.",
            "If discussion was intended, use a short recap or exit question to check understanding."
        ]
    if activity < 25:
        return [
            "The recording contains long low-volume intervals. This may be normal for lecture or individual work.",
            "Consider a brief check-for-understanding question or think-pair-share if it fits the lesson."
        ]
    return [
        "The recording shows a mixture of quieter and more active intervals.",
        "Compare these indicators with the lesson plan and student feedback before changing teaching methods."
    ]
