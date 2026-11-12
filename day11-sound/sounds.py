"""Day 11 audio: one short looping clip per sound -> data/audio/<key>.m4a (AAC, mono, 22 kHz).

Real recordings (Wikimedia Commons, downloaded by fetch.sh into data/raw/audio/):
  dogs   "A dog making noises and barking.flac"   Amada44, CC BY-SA 3.0
  heli   "Helicopter flying overhead.wav"         Doggo19292, attribution
  horn   "Car Horn.wav"                           15HPanska_Ruttner_Jan, CC0
  talk   "442697-SBssa-Crowd Talking 003.wav"     SoundSnap, CC0
Synthesized here (no recording needed, they are machine-like sounds anyway):
  music (kick, bass, chord stabs, hi-hats), bang (thuds from upstairs), build (jackhammer), icecream (music-box jingle)
Needs macOS afconvert (decode + AAC encode).
Run: ../.venv/bin/python sounds.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "audio"
OUT = HERE / "data" / "audio"
OUT.mkdir(parents=True, exist_ok=True)
SR = 22050
rng = np.random.default_rng(11)


def lowpass(x, hz, order=2):
    b, a = butter(order, hz / (SR / 2))
    return lfilter(b, a, x)


def bandpass(x, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return lfilter(b, a, x)


def load(name):
    mono = RAW / f"{name}_mono.wav"
    src = next(RAW.glob(f"{name}.*"))
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(src), str(mono)], check=True)
    _, x = wavfile.read(mono)
    return x.astype(float) / 32768


def loop(x, a, b, fade=0.4):
    """Seconds a..b of x as a seamless loop: the tail past b is crossfaded into the start."""
    i, n, f = int(a * SR), int((b - a) * SR), int(fade * SR)
    seg = x[i:i + n + f].copy()
    out = seg[:n].copy()
    r = np.linspace(0, 1, f)
    out[:f] = seg[:f] * r + seg[n:n + f] * (1 - r)
    return out


def t(sec):
    return np.arange(int(sec * SR)) / SR


def place(buf, at, x):
    i = int(at * SR)
    m = min(len(x), len(buf) - i)
    buf[i:i + m] += x[:m]


def music():  # 120 bpm party: kick, bass, offbeat chord stabs and hi-hats, a little muffled
    L = 4.0   # two bars: Am then F, so it reads as a song, not a pulse
    buf = np.zeros(int(L * SR))
    k = t(0.3)
    kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-k * 30)) / SR) * np.exp(-k * 10)
    hat_n = t(0.04)
    for bar, (root, chord) in enumerate([(55.0, [220.0, 261.6, 329.6]), (43.65, [174.6, 220.0, 261.6])]):
        for beat in range(4):
            at = bar * 2 + beat * 0.5
            place(buf, at, 0.9 * kick)
            b = t(0.2)
            saw = sum(np.sin(2 * np.pi * root * h * b) / h for h in range(1, 8))
            place(buf, at + 0.25, 0.45 * saw * np.minimum(1, b * 80) * np.exp(-b * 7))
            st = t(0.16)
            stab = sum(np.sin(2 * np.pi * f * hh * st) / hh for f in chord for hh in range(1, 5))
            place(buf, at + 0.25, 0.22 * lowpass(stab, 2200) * np.minimum(1, st * 200) * np.exp(-st * 14))
            for off in (0, 0.25):
                hat = bandpass(rng.normal(0, 1, len(hat_n)), 5000, 9000) * np.exp(-hat_n * 120)
                place(buf, at + off, 0.12 * hat)
    return lowpass(buf, 5000, 2)


def bang():  # irregular heavy thuds, like furniture dragged and dropped upstairs
    L = 3.0
    buf = np.zeros(int(L * SR))
    for at, amp in [(0.15, 1), (0.6, .8), (1.45, 1), (1.75, .7), (2.3, .9)]:
        d = t(0.4)
        thud = np.sin(2 * np.pi * (60 + 30 * np.exp(-d * 30)) * d) * np.exp(-d * 14)
        knock = lowpass(rng.normal(0, 1, len(d)), 900) * np.exp(-d * 60) * 0.6
        place(buf, at, amp * (thud + knock))
    return buf


def build():  # jackhammer: 18 strikes/s for 1.8 s, then a pause
    L = 3.0
    buf = np.zeros(int(L * SR))
    d = t(0.05)
    for i in range(int(1.8 * 18)):
        hit = bandpass(rng.normal(0, 1, len(d)), 300, 3500) * np.exp(-d * 90)
        hit += 0.6 * np.sin(2 * np.pi * 140 * d) * np.exp(-d * 70)
        place(buf, 0.1 + i / 18, hit * (0.85 + 0.3 * rng.random()))
    env = np.ones(len(buf))
    return buf * env


def icecream():  # an original music-box jingle (not any truck's real tune)
    L = 4.0
    buf = np.zeros(int(L * SR))
    C = 523.25
    melody = [0, 4, 7, 12, 9, 7, 4, 7, 5, 4, 2, 0]   # semitones above C5
    for i, s in enumerate(melody):
        f = C * 2 ** (s / 12)
        d = t(0.9)
        bell = (np.sin(2 * np.pi * f * d) + 0.4 * np.sin(2 * np.pi * 2 * f * d) * np.exp(-d * 6)
                + 0.15 * np.sin(2 * np.pi * 3.01 * f * d) * np.exp(-d * 10)) * np.exp(-d * 4)
        place(buf, i * (L - 0.6) / len(melody), bell * np.minimum(1, d * 400))
    return buf


def a_weighted_rms(x):
    """RMS after A-weighting (how loud it sounds, not how big the waveform is)."""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    f2 = f ** 2
    ra = (12194 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    y = np.fft.irfft(np.fft.rfft(x) * ra / ra[np.argmin(abs(f - 1000))], len(x))
    return np.sqrt((y ** 2).mean())


# Perceived-loudness trims on top of A-weighting, set by ear: jackhammer and barking cut through
# a mix; bass and rotor rumble get lost on laptop speakers.
GAINS = {}
TRIM = {"bang": 0.7, "build": 0.6, "dogs": 0.65, "music": 1.35, "heli": 1.5, "icecream": 0.85, "horn": 0.8}


def finish(x, key, target=0.055):
    x = x - x.mean()
    x = x * (target * TRIM.get(key, 1) / a_weighted_rms(x))
    # bass-heavy clips need big waveforms to sound as loud: store them at full scale and let the
    # page turn them back up (makeup gain in audio/gains.json) instead of clipping here
    peak = np.abs(x).max()
    makeup = max(1.0, peak / 0.95)
    x = x / makeup
    GAINS[key] = round(float(makeup), 3)
    e = int(0.01 * SR)  # 10 ms de-click at the loop seam
    x[:e] *= np.linspace(0, 1, e)
    x[-e:] *= np.linspace(1, 0, e)
    wav = OUT / f"{key}.wav"
    wavfile.write(wav, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    m4a = OUT / f"{key}.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "40000", str(wav), str(m4a)], check=True)
    wav.unlink()
    print(f"{key:9s} {len(x) / SR:4.1f}s  {m4a.stat().st_size / 1000:5.0f} KB")


def main():
    finish(music(), "music")
    finish(bang(), "bang")
    finish(build(), "build")
    finish(icecream(), "icecream")
    finish(loop(load("talk"), 20, 30), "talk")
    finish(loop(load("heli"), 0.3, 9.5), "heli")
    dog = load("dog")[int(8.4 * SR):int(14 * SR)]
    finish(np.concatenate([dog, np.zeros(int(1.5 * SR))]), "dogs")
    horn = load("horn")[int(0.35 * SR):int(3.3 * SR)]
    finish(np.concatenate([horn, np.zeros(int(2.5 * SR))]), "horn")
    (OUT / "gains.json").write_text(json.dumps(GAINS))
    print("makeup gains:", GAINS)


if __name__ == "__main__":
    main()
