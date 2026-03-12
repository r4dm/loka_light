#!/usr/bin/env python3
"""
loka_ytp.py — A YouTube Poop style video about the loka_light universe.
Glitchy, absurdist, hypnotic journey through multipolar algebra,
Σ→0 cascades, pseudo-quantum circuits, and the N-polar beyond.

Usage: python3 loka_ytp.py
Output: loka_ytp.mp4

Optimised for Apple M1 Max (VideoToolbox HW encode, 10-core threading).
"""

import cmath
import math
import os
import random
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    VideoClip,
    concatenate_videoclips,
)

random.seed(42)
np.random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────────────────
W, H = 1280, 720
FPS = 24
OUTPUT = "loka_ytp.mp4"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 65)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
DARK_BG = (8, 6, 18)
TERMINAL_GREEN = (0, 230, 64)
LOKA_PURPLE = (120, 40, 200)
LOKA_TEAL = (0, 200, 180)
LOKA_GOLD = (255, 200, 40)
SIGMA_BLUE = (30, 120, 255)
GLITCH_COLORS = [RED, CYAN, MAGENTA, YELLOW, GREEN, LOKA_PURPLE, LOKA_TEAL]

# polarity palette for C_n rings
POLE_COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
    (255, 160, 40), (180, 80, 255), (80, 200, 160),
]


def _font(size):
    for name in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _bold(size):
    for name in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Bold.otf",
    ]:
        if os.path.exists(name):
            try:
                return ImageFont.truetype(name, size, index=1 if name.endswith(".ttc") else 0)
            except Exception:
                continue
    return _font(size)


FONT_SM = _font(22)
FONT_MD = _font(36)
FONT_LG = _font(64)
FONT_XL = _font(96)
FONT_HUGE = _font(130)
FONT_BOLD = _bold(48)
FONT_TITLE = _bold(72)

# ── AUDIO ────────────────────────────────────────────────────────────────────
SR = 44100


def _tone(freq, dur, vol=0.3, wave_type="sine"):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    if wave_type == "square":
        s = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "saw":
        s = 2.0 * (t * freq - np.floor(t * freq + 0.5))
    else:
        s = np.sin(2 * np.pi * freq * t)
    return (s * vol * 32767).astype(np.int16)


def _noise(dur, vol=0.15):
    return (np.random.uniform(-1, 1, int(SR * dur)) * vol * 32767).astype(np.int16)


def _glitch_audio(dur, base=200):
    out = np.array([], dtype=np.int16)
    t = 0.0
    while t < dur:
        cl = random.uniform(0.02, 0.12)
        r = random.random()
        if r < 0.3:
            c = _tone(random.uniform(100, 2000), cl, random.uniform(0.1, 0.4), "square")
        elif r < 0.6:
            c = _noise(cl, random.uniform(0.05, 0.25))
        elif r < 0.8:
            c = _tone(random.uniform(50, 800), cl, random.uniform(0.2, 0.5), "saw")
        else:
            c = np.zeros(int(SR * cl), dtype=np.int16)
        out = np.concatenate([out, c])
        t += cl
    return out[: int(SR * dur)]


def _stutter_audio(dur):
    base = _tone(random.uniform(200, 600), 0.03, 0.3, "square")
    return np.tile(base, int(dur / 0.03) + 1)[: int(SR * dur)]


def _drone(dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    s = np.sin(2 * np.pi * 55 * t) * 0.3 + np.sin(2 * np.pi * 110 * t) * 0.15
    s *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    return (s * 32767).astype(np.int16)


def _error_audio(dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    s = np.sign(np.sin(2 * np.pi * 150 * t)) * 0.4
    s *= 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * t))
    return (s * 32767).astype(np.int16)


def _startup_audio(dur):
    out = np.array([], dtype=np.int16)
    freqs = [130.8, 164.8, 196.0, 246.9, 329.6, 392.0, 523.3, 659.3]
    nl = dur / len(freqs)
    for f in freqs:
        c = _tone(f, nl, 0.2)
        fade = int(0.005 * SR)
        if len(c) > fade * 2:
            c[:fade] = (c[:fade] * np.linspace(0, 1, fade)).astype(np.int16)
            c[-fade:] = (c[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
        out = np.concatenate([out, c])
    return out[: int(SR * dur)]


def _typewriter(dur, cps=15):
    buf = np.zeros(int(SR * dur), dtype=np.float64)
    click = _tone(4000, 0.008, 0.15, "square").astype(np.float64)
    step = int(SR / cps)
    for i in range(0, len(buf) - len(click), step):
        pos = max(0, i + random.randint(-100, 100))
        end = min(len(buf), pos + len(click))
        buf[pos:end] += click[: end - pos]
    return np.clip(buf, -32767, 32767).astype(np.int16)


def _cn_chord(n, dur, vol=0.15):
    """Harmonic chord from C_n roots of unity mapped to frequencies."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    s = np.zeros_like(t)
    base = 220.0
    for k in range(n):
        freq = base * (1 + k * 0.5)
        phase = 2 * math.pi * k / n
        s += np.sin(2 * np.pi * freq * t + phase)
    s /= max(1, n)
    return (s * vol * 32767).astype(np.int16)


def _wav(samples, path):
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples.tobytes())


def _audio_clip(samples, dur):
    path = tempfile.mktemp(suffix=".wav")
    padded = np.zeros(int(SR * dur), dtype=np.int16)
    ln = min(len(samples), len(padded))
    padded[:ln] = samples[:ln]
    _wav(padded, path)
    return AudioFileClip(path)


# ── FRAME TOOLS ──────────────────────────────────────────────────────────────
def _text_frame(text, bg=DARK_BG, fg=WHITE, font=None, align="center",
                y_offset=0, scanlines=True, vignette=True):
    if font is None:
        font = FONT_MD
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    lh, lw = [], []
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        lw.append(bb[2] - bb[0])
        lh.append(bb[3] - bb[1])
    total = sum(lh) + (len(lines) - 1) * 8
    sy = (H - total) // 2 + y_offset
    for i, ln in enumerate(lines):
        x = {"center": (W - lw[i]) // 2, "left": 60}.get(align, W - lw[i] - 60)
        draw.text((x, sy), ln, fill=fg, font=font)
        sy += lh[i] + 8
    arr = np.array(img)
    if scanlines:
        arr[::3, :, :] = (arr[::3, :, :] * 0.85).astype(np.uint8)
    if vignette:
        af = arr.astype(np.float64)
        yc = np.linspace(-1, 1, H)[:, None]
        xc = np.linspace(-1, 1, W)[None, :]
        v = np.clip(1.0 - (np.sqrt(xc**2 + yc**2) - 0.7) * 0.8, 0.3, 1.0)
        arr = (af * v[:, :, None]).astype(np.uint8)
    return arr


def _glitch(frame, intensity=0.5):
    a = frame.copy()
    h, w = a.shape[:2]
    for _ in range(int(5 + intensity * 20)):
        y = random.randint(0, h - 1)
        sh = random.randint(1, int(10 + intensity * 40))
        a[y: min(y + sh, h)] = np.roll(a[y: min(y + sh, h)], random.randint(-int(intensity * 100), int(intensity * 100)), axis=1)
    if random.random() < intensity:
        a[:, :, 0] = np.roll(a[:, :, 0], random.randint(-15, 15), axis=1)
        a[:, :, 2] = np.roll(a[:, :, 2], random.randint(-15, 15), axis=1)
    for _ in range(int(intensity * 8)):
        bx, by = random.randint(0, w - 1), random.randint(0, h - 1)
        a[by: by + random.randint(5, 30), bx: bx + random.randint(20, 150)] = random.choice(GLITCH_COLORS)
    return a


def _deep_fry(frame, intensity=1.0):
    from PIL import ImageEnhance
    img = Image.fromarray(frame)
    img = ImageEnhance.Contrast(img).enhance(1.5 + intensity)
    img = ImageEnhance.Color(img).enhance(1.5 + intensity * 2)
    img = ImageEnhance.Sharpness(img).enhance(2 + intensity * 3)
    a = np.array(img).astype(np.float64)
    a = np.clip(a + np.random.normal(0, 15 * intensity, a.shape), 0, 255).astype(np.uint8)
    q = 8 + int(intensity * 8)
    return (a // q) * q


def _shake(frame, mag=15):
    return np.roll(np.roll(frame, random.randint(-mag, mag), 1), random.randint(-mag, mag), 0)


def _zoom(frame, z=1.5, center=None):
    h, w = frame.shape[:2]
    cx, cy = center or (w // 2, h // 2)
    nw, nh = int(w / z), int(h / z)
    x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
    cropped = frame[y1: min(h, y1 + nh), x1: min(w, x1 + nw)]
    return np.array(Image.fromarray(cropped).resize((w, h), Image.Resampling.NEAREST))


def _invert(frame):
    return 255 - frame


def _pixel_sort(frame, threshold=128):
    a = frame.copy()
    for y in range(0, a.shape[0], random.randint(2, 6)):
        row = a[y].copy()
        b = np.mean(row, axis=1)
        idx = np.where(b > threshold)[0]
        if len(idx) > 1:
            sp = row[idx]
            a[y, idx] = sp[sp[:, 1].argsort()]
    return a


# ── CLIP BUILDER ─────────────────────────────────────────────────────────────
def _clip(frame_fn, dur, audio=None):
    clip = VideoClip(frame_fn, duration=dur).with_fps(FPS)
    if audio is not None:
        clip = clip.with_audio(_audio_clip(audio, dur))
    return clip


# ── SCENES ───────────────────────────────────────────────────────────────────

def scene_boot():
    """Loka system BIOS boot."""
    dur = 4.5
    lines = [
        "LOKA BIOS v1.4.2 — N-Polar Core",
        "Initialising polarity axes...... OK",
        "Loading C_n algebras: C3 C4 C6 C7",
        "SigmaGuard NX cascade: ARMED",
        "PseudoBlockM formation: ONLINE",
        "MultipolarOscillator: 6-pole mode",
        "Σ constraint: ∑(cᵢ·aᵢ) → 0",
        "Witnesses: CHSH / CGLMP loaded",
        "",
        "MULTIPOLARITY ENGAGED.",
    ]

    def make_frame(t):
        n = min(len(lines), int(t * 3) + 1)
        text = "\n".join(lines[:n])
        frame = _text_frame(text, bg=BLACK, fg=TERMINAL_GREEN, font=FONT_SM, align="left")
        if random.random() < 0.08:
            frame = (frame * random.uniform(0.5, 0.9)).astype(np.uint8)
        if int(t * 4) % 2 == 0:
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            cy = 200 + n * 28
            d.rectangle([60, cy, 75, cy + 22], fill=TERMINAL_GREEN)
            frame = np.array(img)
        return frame

    return _clip(make_frame, dur, _startup_audio(dur))


def scene_cn_ring():
    """Visualise C_n roots of unity forming on a circle, then multiplying."""
    dur = 5.0
    N = 6
    roots = [cmath.exp(2j * cmath.pi * k / N) for k in range(N)]
    pole_names = [f"C{k}" for k in range(N)]

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        cx, cy, r = W // 2, H // 2 + 20, 220
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(40, 40, 60), width=2)

        n_visible = min(N, int(t * 2.5) + 1)
        for k in range(n_visible):
            z = roots[k]
            px = cx + int(z.real * r)
            py = cy - int(z.imag * r)
            col = POLE_COLORS[k % len(POLE_COLORS)]
            d.ellipse([px - 12, py - 12, px + 12, py + 12], fill=col)
            d.text((px + 16, py - 10), pole_names[k], fill=col, font=FONT_SM)

        if n_visible >= 2 and t > 1.5:
            for k in range(n_visible - 1):
                z1, z2 = roots[k], roots[k + 1]
                d.line([
                    cx + int(z1.real * r), cy - int(z1.imag * r),
                    cx + int(z2.real * r), cy - int(z2.imag * r),
                ], fill=(60, 60, 100), width=1)

        if t < 2.5:
            d.text((W // 2 - 120, 50), f"C_{N} — Cyclic Loka", fill=WHITE, font=FONT_MD)
        elif t < 3.5:
            d.text((W // 2 - 200, 50), "roots of unity basis", fill=LOKA_TEAL, font=FONT_MD)
            angle = t * 2
            highlight = int(angle) % N
            z = roots[highlight]
            px, py = cx + int(z.real * r), cy - int(z.imag * r)
            d.ellipse([px - 18, py - 18, px + 18, py + 18], outline=WHITE, width=3)
        elif t < 4.2:
            d.text((W // 2 - 250, 50), "C2 × C3 = C5 (mod 6)", fill=LOKA_GOLD, font=FONT_MD)
            z_product = roots[5]
            ppx, ppy = cx + int(z_product.real * r), cy - int(z_product.imag * r)
            d.ellipse([ppx - 20, ppy - 20, ppx + 20, ppy + 20], outline=LOKA_GOLD, width=4)
        else:
            d.text((W // 2 - 200, 50), "ω = e^(2πi/6)", fill=MAGENTA, font=FONT_LG)

        frame = np.array(img)
        if t > 4.0:
            frame = _glitch(frame, 0.3)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        return frame

    return _clip(make_frame, dur, _cn_chord(N, dur))


def scene_sigma_cascade():
    """Sigma → 0 cascade: bar chart of amplitudes converging."""
    dur = 5.0
    N = 6
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.2, 1.0, N)
    stages = [raw.copy()]
    v = raw.copy()
    for _ in range(4):
        v = v - v.mean()
        stages.append(v.copy())

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        stage_idx = min(len(stages) - 1, int(t * 1.2))
        blend = min(1.0, (t * 1.2) - stage_idx)
        if stage_idx < len(stages) - 1:
            vals = stages[stage_idx] * (1 - blend) + stages[min(stage_idx + 1, len(stages) - 1)] * blend
        else:
            vals = stages[-1]

        sigma = float(np.sum(vals))

        bar_w = 80
        gap = 20
        total_w = N * bar_w + (N - 1) * gap
        sx = (W - total_w) // 2
        base_y = H // 2 + 50
        scale = 200.0

        for i in range(N):
            x = sx + i * (bar_w + gap)
            bh = int(abs(vals[i]) * scale)
            col = POLE_COLORS[i % len(POLE_COLORS)]
            if vals[i] >= 0:
                d.rectangle([x, base_y - bh, x + bar_w, base_y], fill=col)
            else:
                d.rectangle([x, base_y, x + bar_w, base_y + bh], fill=col)
            d.text((x + 10, base_y + 10), f"P{i}", fill=(180, 180, 180), font=FONT_SM)

        d.line([(sx - 20, base_y), (sx + total_w + 20, base_y)], fill=(80, 80, 80), width=2)

        if abs(sigma) < 0.01:
            sigma_color = GREEN
        elif abs(sigma) < 0.5:
            sigma_color = YELLOW
        else:
            sigma_color = RED
        d.text((W // 2 - 180, 50), f"Σ = {sigma:+.4f}", fill=sigma_color, font=FONT_LG)

        stage_label = f"NX section {stage_idx}" if stage_idx > 0 else "raw O1 (before)"
        d.text((W // 2 - 140, 130), stage_label, fill=(150, 150, 150), font=FONT_SM)

        if stage_idx >= len(stages) - 1:
            d.text((W // 2 - 80, H - 100), "Σ → 0 ✓", fill=GREEN, font=FONT_BOLD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        if t > 4.0:
            frame = _glitch(frame, (t - 4.0) * 0.6)
        return frame

    audio = np.concatenate([_drone(3.0), _cn_chord(6, 2.0)])
    return _clip(make_frame, dur, audio)


def scene_multipolarity_title():
    """Big dramatic title: MULTIPOLARITY."""
    dur = 3.0

    def make_frame(t):
        if t < 1.0:
            frame = _text_frame("beyond binary", fg=(120, 120, 140), font=FONT_MD)
        elif t < 2.0:
            frame = _text_frame("beyond binary\nbeyond ternary", fg=LOKA_TEAL, font=FONT_LG)
            frame = _glitch(frame, 0.2)
        else:
            frame = _text_frame("N-POLARITY", fg=WHITE, bg=LOKA_PURPLE, font=FONT_HUGE)
            frame = _deep_fry(frame, 0.5)
            frame = _shake(frame, 15)
        return frame

    audio = np.concatenate([_drone(2.0), _glitch_audio(1.0)])
    return _clip(make_frame, dur, audio)


def scene_wave_propagation():
    """Multipolar wave visualisation — N conjugates oscillating."""
    dur = 4.0
    N = 6

    phases = np.linspace(0, 2 * np.pi, N, endpoint=False)

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((50, 30), "MultiConjugateFunction", fill=LOKA_TEAL, font=FONT_SM)
        d.text((50, 60), f"n_conjugates = {N}", fill=(150, 150, 150), font=FONT_SM)

        lane_h = (H - 140) // N
        for k in range(N):
            col = POLE_COLORS[k % len(POLE_COLORS)]
            base_y = 120 + k * lane_h + lane_h // 2
            freq = 1.5 + k * 0.3
            amp = 0.6 + 0.3 * math.sin(t * 2 + k)
            points = []
            for x in range(60, W - 60, 3):
                norm_x = (x - 60) / (W - 120)
                phase_shift = phases[k]
                y = base_y + int(amp * (lane_h * 0.35) * math.sin(
                    2 * math.pi * freq * norm_x + t * 4 + phase_shift
                ))
                points.append((x, y))
            for i in range(len(points) - 1):
                d.line([points[i], points[i + 1]], fill=col, width=2)
            d.text((10, base_y - 8), f"ψ{k}", fill=col, font=FONT_SM)

        if t > 2.5:
            sigma_val = sum(
                math.sin(t * 4 + phases[k]) * (0.6 + 0.3 * math.sin(t * 2 + k))
                for k in range(N)
            )
            sc = GREEN if abs(sigma_val) < 0.5 else YELLOW
            d.text((W - 300, 30), f"|Σ| = {abs(sigma_val):.3f}", fill=sc, font=FONT_MD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        return frame

    return _clip(make_frame, dur, _cn_chord(N, dur, vol=0.12))


def scene_pseudo_block_m():
    """PseudoBlockM: 2-pole sources merging into N-pole space."""
    dur = 4.0

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 200, 30), "PseudoBlockM", fill=LOKA_GOLD, font=FONT_LG)

        n_sources = 3
        n_poles = 6
        src_x = 150
        dst_x = W - 250
        mid_x = W // 2

        for i in range(n_sources):
            y = 180 + i * 150
            d.rounded_rectangle([src_x - 60, y - 25, src_x + 60, y + 25], radius=8, fill=(50, 30, 30), outline=RED)
            d.text((src_x - 40, y - 12), f"S{i} (2P)", fill=WHITE, font=FONT_SM)

            if t > 0.5 + i * 0.4:
                progress = min(1.0, (t - 0.5 - i * 0.4) / 0.8)
                lx = int(src_x + 60 + (mid_x - src_x - 60) * progress)
                d.line([(src_x + 60, y), (lx, y)], fill=POLE_COLORS[i], width=3)

        d.rounded_rectangle([mid_x - 40, 150, mid_x + 40, H - 100], radius=12, fill=(30, 30, 60), outline=SIGMA_BLUE)
        d.text((mid_x - 20, H // 2 - 15), "Σ", fill=SIGMA_BLUE, font=FONT_MD)
        d.text((mid_x - 10, H // 2 + 25), "O1", fill=(120, 120, 120), font=FONT_SM)

        if t > 2.0:
            for k in range(n_poles):
                y = 150 + k * 80
                progress = min(1.0, (t - 2.0) / 1.0)
                rx = int(mid_x + 40 + (dst_x - mid_x - 40) * progress)
                col = POLE_COLORS[k % len(POLE_COLORS)]
                d.line([(mid_x + 40, 150 + k * ((H - 250) // (n_poles - 1))),
                        (rx, 150 + k * ((H - 250) // (n_poles - 1)))], fill=col, width=2)

            d.rounded_rectangle([dst_x - 50, 140, dst_x + 70, H - 90], radius=10, fill=(20, 40, 20), outline=GREEN)
            d.text((dst_x - 30, H // 2 - 10), f"C_{n_poles}", fill=GREEN, font=FONT_MD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        if t > 3.5:
            frame = _glitch(frame, 0.3)
        return frame

    audio = np.concatenate([_drone(2.5), _cn_chord(6, 1.5)])
    return _clip(make_frame, dur, audio)


def scene_sigma_guard():
    """SigmaGuard NX — dramatic Σ→0 enforcement."""
    dur = 3.5
    sigma_vals = [2.34, 0.87, 0.12, 0.003, 0.00001]

    def make_frame(t):
        idx = min(len(sigma_vals) - 1, int(t * 1.6))
        sv = sigma_vals[idx]

        if sv > 1.0:
            bg = (40, 10, 10)
        elif sv > 0.1:
            bg = (30, 30, 10)
        else:
            bg = (10, 30, 10)

        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 220, 80), "SigmaGuard NX", fill=SIGMA_BLUE, font=FONT_LG)
        d.text((W // 2 - 160, 170), f"section {idx + 1} / {len(sigma_vals)}", fill=(150, 150, 150), font=FONT_SM)

        bar_w = W - 200
        bar_h = 50
        bar_x = 100
        bar_y = H // 2 - bar_h // 2
        d.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=WHITE, width=2)

        fill_w = int(bar_w * min(1.0, sv / 3.0))
        if sv > 1.0:
            col = RED
        elif sv > 0.1:
            col = YELLOW
        else:
            col = GREEN
        d.rectangle([bar_x + 2, bar_y + 2, bar_x + 2 + fill_w, bar_y + bar_h - 2], fill=col)

        d.text((W // 2 - 180, bar_y + bar_h + 30), f"|Σ| = {sv:.5f}", fill=col, font=FONT_LG)

        if sv < 0.001:
            d.text((W // 2 - 130, H - 120), "Σ CONVERGED", fill=GREEN, font=FONT_BOLD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        if idx >= 3:
            frame = _glitch(frame, 0.15)
        return frame

    audio = np.concatenate([
        _tone(220, 0.7, 0.1), _tone(330, 0.7, 0.1),
        _tone(440, 0.7, 0.12), _tone(660, 0.7, 0.15),
        _cn_chord(6, 0.7),
    ])
    return _clip(make_frame, dur, audio)


def scene_polarity_scan():
    """Detector scanning for intrinsic polarity — resonance peak."""
    dur = 4.0
    true_n = 6
    scan_range = list(range(2, 13))
    responses = [max(0.0, 1.0 - 0.25 * abs(p - true_n) + 0.05 * random.random()) for p in scan_range]
    best_idx = responses.index(max(responses))

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 220, 40), "Polarity Detector", fill=LOKA_TEAL, font=FONT_LG)

        n_visible = min(len(scan_range), int(t * 4) + 1)
        bw = 60
        gap = 10
        total = len(scan_range) * bw + (len(scan_range) - 1) * gap
        sx = (W - total) // 2
        base_y = H - 120

        for i in range(n_visible):
            x = sx + i * (bw + gap)
            bh = int(responses[i] * 350)
            col = GREEN if i == best_idx and t > 2.5 else POLE_COLORS[i % len(POLE_COLORS)]
            d.rectangle([x, base_y - bh, x + bw, base_y], fill=col)
            d.text((x + 15, base_y + 8), str(scan_range[i]), fill=(180, 180, 180), font=FONT_SM)

        if t > 2.5:
            d.text((W // 2 - 200, 130), f"RESONANCE: N = {true_n}", fill=GREEN, font=FONT_BOLD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        return frame

    audio = np.concatenate([_typewriter(2.5, 8), _cn_chord(6, 1.5)])
    return _clip(make_frame, dur, audio)


def scene_witness_pack():
    """CHSH / CGLMP witness values flashing."""
    dur = 3.5
    witnesses = [
        ("CHSH (d=2)", 2.828, 2.0),
        ("CGLMP (d=3)", 2.915, 2.0),
        ("CGLMP (d=4)", 2.969, 2.0),
        ("CGLMP (d=5)", 3.003, 2.0),
    ]

    def make_frame(t):
        idx = min(len(witnesses) - 1, int(t * 1.5))
        name, qval, cval = witnesses[idx]

        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 240, 50), "Pseudo-Quantum Witnesses", fill=LOKA_PURPLE, font=FONT_MD)

        for i in range(idx + 1):
            y = 160 + i * 100
            n2, qv, cv = witnesses[i]
            d.text((100, y), n2, fill=WHITE, font=FONT_MD)
            violation = qv > cv
            vc = GREEN if violation else RED
            d.text((550, y), f"Q={qv:.3f}", fill=vc, font=FONT_MD)
            d.text((800, y), f"C={cv:.3f}", fill=(120, 120, 120), font=FONT_MD)
            if violation:
                d.text((1000, y), "VIOLATED", fill=GREEN, font=FONT_SM)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        if t > 2.5:
            frame = _glitch(frame, 0.3)
        return frame

    audio = np.concatenate([_tone(440, 2.0, 0.1), _glitch_audio(1.5)])
    return _clip(make_frame, dur, audio)


def scene_translation_gap():
    """n→2 projection gap — information lost in translation."""
    dur = 3.5

    def make_frame(t):
        if t < 1.5:
            img = Image.new("RGB", (W, H), DARK_BG)
            d = ImageDraw.Draw(img)
            d.text((W // 2 - 250, 100), "N=6 amplitude space", fill=LOKA_TEAL, font=FONT_MD)
            for k in range(6):
                x = 150 + k * 160
                bh = int(60 + 40 * math.sin(k * 1.2))
                col = POLE_COLORS[k]
                d.rectangle([x, H // 2 - bh, x + 100, H // 2 + bh], fill=col)
            d.text((W // 2 - 100, H - 120), "full picture", fill=GREEN, font=FONT_SM)
            frame = np.array(img)
        elif t < 2.5:
            img = Image.new("RGB", (W, H), (20, 10, 10))
            d = ImageDraw.Draw(img)
            d.text((W // 2 - 200, 100), "projected to N=2", fill=RED, font=FONT_MD)
            for k in range(2):
                x = W // 2 - 150 + k * 200
                bh = int(80 + 30 * math.sin(k))
                col = POLE_COLORS[k]
                d.rectangle([x, H // 2 - bh, x + 100, H // 2 + bh], fill=col)
            d.text((W // 2 - 200, H - 120), "INFORMATION LOST", fill=RED, font=FONT_BOLD)
            frame = np.array(img)
            frame = _glitch(frame, 0.4)
        else:
            frame = _text_frame("translation gap\nvisibility ≠ 1", fg=RED, font=FONT_LG)
            frame = _deep_fry(frame, 0.8)
            frame = _shake(frame, 15)
        return frame

    audio = np.concatenate([_drone(2.0), _error_audio(1.5)])
    return _clip(make_frame, dur, audio)


def scene_quantum_circuit():
    """Pseudo-quantum circuit: H → phase → H → measure."""
    dur = 4.0
    gates = ["INIT |0⟩", "H (superpose)", "Φ (phase)", "H (interfere)", "MEASURE"]

    def make_frame(t):
        img = Image.new("RGB", (W, H), BLACK)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 280, 30), "Pseudo-Quantum Circuit", fill=LOKA_PURPLE, font=FONT_MD)

        n_vis = min(len(gates), int(t * 1.5) + 1)
        wire_y = H // 2
        d.line([(80, wire_y), (W - 80, wire_y)], fill=(60, 60, 60), width=3)

        gate_w = 150
        total_w = len(gates) * gate_w
        sx = (W - total_w) // 2

        for i in range(n_vis):
            gx = sx + i * gate_w + gate_w // 2
            active = i == n_vis - 1
            box_col = LOKA_PURPLE if active else (40, 30, 60)
            border_col = MAGENTA if active else (80, 60, 120)
            d.rounded_rectangle(
                [gx - 55, wire_y - 35, gx + 55, wire_y + 35],
                radius=8, fill=box_col, outline=border_col, width=2
            )
            d.text((gx - 45, wire_y - 12), gates[i][:10], fill=WHITE, font=FONT_SM)

        if n_vis >= len(gates):
            outcome = random.choice(["P0", "P1", "P2"])
            d.text((W // 2 - 80, wire_y + 80), f"→ {outcome}", fill=LOKA_GOLD, font=FONT_LG)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        if t > 3.0:
            frame = _glitch(frame, 0.3)
        return frame

    audio = np.concatenate([_tone(330, 2.5, 0.1), _cn_chord(3, 1.5)])
    return _clip(make_frame, dur, audio)


def scene_flash_montage():
    """Rapid flash cuts of loka concepts."""
    dur = 2.5
    flashes = [
        ("C_n", RED, FONT_HUGE),
        ("Σ → 0", CYAN, FONT_HUGE),
        ("P_PERP", MAGENTA, FONT_XL),
        ("NX CASCADE", YELLOW, FONT_XL),
        ("HARLOKA", GREEN, FONT_HUGE),
        ("k-CONJUGATE", LOKA_PURPLE, FONT_XL),
        ("TATTVA", LOKA_TEAL, FONT_XL),
        ("DHARMA", WHITE, FONT_HUGE),
        ("POLARITY", RED, FONT_XL),
        ("MULTIPOLARITY", LOKA_GOLD, FONT_HUGE),
    ]

    def make_frame(t):
        idx = int(t * 6) % len(flashes)
        text, col, font = flashes[idx]
        bg = random.choice([BLACK, (20, 0, 0), (0, 0, 20), (10, 0, 20)])
        frame = _text_frame(text, fg=col, bg=bg, font=font, scanlines=False, vignette=False)
        if random.random() < 0.5:
            frame = _invert(frame)
        if random.random() < 0.3:
            frame = _deep_fry(frame, 0.5)
        return _shake(frame, 15)

    audio = _glitch_audio(dur)
    audio = np.clip(audio.astype(np.int32) * 2, -32767, 32767).astype(np.int16)
    return _clip(make_frame, dur, audio)


def scene_matrix_polarities():
    """Matrix-style falling polarity symbols."""
    dur = 3.0
    tokens = ["C₀", "C₁", "C₂", "C₃", "C₄", "C₅", "P⊥", "Σ", "ψ",
              "ω", "Φ", "NX", "O1", "O2", "O3", "∑cᵢ", "→0",
              "H", "Δ", "ρ", "⊗"]
    n_cols = 28
    col_speeds = [random.uniform(100, 400) for _ in range(n_cols)]
    col_tokens = [[random.choice(tokens) for _ in range(50)] for _ in range(n_cols)]
    col_offsets = [random.uniform(0, 500) for _ in range(n_cols)]

    def make_frame(t):
        img = Image.new("RGB", (W, H), BLACK)
        d = ImageDraw.Draw(img)
        for c in range(n_cols):
            x = int(c * (W / n_cols)) + 5
            speed = col_speeds[c]
            off = col_offsets[c]
            for row in range(15):
                y = int((off + row * 40 + t * speed) % (H + 200)) - 100
                if 0 <= y < H:
                    tok = col_tokens[c][row % len(col_tokens[c])]
                    br = max(60, 255 - row * 15)
                    col = (0, br, int(br * 0.5)) if row > 0 else WHITE
                    d.text((x, y), tok, fill=col, font=FONT_SM)
        frame = np.array(img)
        frame[::2, :, :] = (frame[::2, :, :] * 0.9).astype(np.uint8)
        if t > 2.0:
            frame = _glitch(frame, (t - 2.0) * 0.5)
        return frame

    return _clip(make_frame, dur, _typewriter(dur, 30))


def scene_existential():
    """Quiet existential moment for a multipolar system."""
    dur = 4.5
    texts = [
        ("If Σ → 0…", (150, 150, 180)),
        ("does anything remain?", (120, 120, 140)),
        ("only the differential.", LOKA_TEAL),
        ("the structure survives\nthe sum does not.", WHITE),
    ]

    def make_frame(t):
        idx = min(len(texts) - 1, int(t * 1.1))
        text, color = texts[idx]
        font = FONT_LG if idx < 3 else FONT_MD
        frame = _text_frame(text, bg=(4, 4, 12), fg=color, font=font)
        if idx == 1:
            br = 0.7 + 0.3 * math.sin(t * 3)
            frame = (frame * br).astype(np.uint8)
        return frame

    audio = _drone(dur)
    audio = (audio * 0.35).astype(np.int16)
    return _clip(make_frame, dur, audio)


def scene_sigma_noise_compare():
    """Σ-consistent vs generic noise — visual comparison."""
    dur = 3.5

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 200, 30), "Σ-noise comparison", fill=WHITE, font=FONT_MD)

        mid = W // 2
        d.line([(mid, 100), (mid, H - 50)], fill=(60, 60, 60), width=2)

        d.text((mid // 2 - 80, 100), "Σ-consistent", fill=GREEN, font=FONT_SM)
        d.text((mid + mid // 2 - 60, 100), "generic", fill=RED, font=FONT_SM)

        np.random.seed(int(t * 10) + 42)
        for i in range(40):
            y = 150 + i * 12
            v_sigma = np.random.normal(0, 0.02)
            v_generic = np.random.normal(0, 0.3)
            x1 = mid // 2 + int(v_sigma * 300)
            x2 = mid + mid // 2 + int(v_generic * 300)
            d.ellipse([x1 - 3, y - 3, x1 + 3, y + 3], fill=GREEN)
            d.ellipse([x2 - 3, y - 3, x2 + 3, y + 3], fill=RED)

        d.text((mid // 2 - 60, H - 80), "δΣ ≈ 0.001", fill=GREEN, font=FONT_SM)
        d.text((mid + mid // 2 - 60, H - 80), "δΣ ≈ 2.34", fill=RED, font=FONT_SM)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        return frame

    audio = np.concatenate([_tone(330, 2.0, 0.1), _glitch_audio(1.5)])
    return _clip(make_frame, dur, audio)


def scene_cascade_pipeline():
    """The full M→N→NX→RX pipeline as animated flow."""
    dur = 4.5

    stages = [
        ("M", "formation", RED, 0.0),
        ("O1", "summation", YELLOW, 0.8),
        ("N", "Σ removal", SIGMA_BLUE, 1.6),
        ("NX", "cascade", LOKA_TEAL, 2.4),
        ("RX", "decode", GREEN, 3.2),
    ]

    def make_frame(t):
        img = Image.new("RGB", (W, H), DARK_BG)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 250, 30), "Pseudomultipolar Cascade", fill=WHITE, font=FONT_MD)

        n_vis = sum(1 for _, _, _, threshold in stages if t > threshold)
        total = len(stages)
        box_w = 150
        gap = (W - 100 - total * box_w) // max(1, total - 1)
        sx = 50

        for i, (name, desc, col, threshold) in enumerate(stages):
            if t <= threshold:
                continue
            x = sx + i * (box_w + gap)
            y = H // 2 - 40

            progress = min(1.0, (t - threshold) / 0.6)
            alpha = int(255 * progress)
            fill = tuple(int(c * progress) for c in col)

            d.rounded_rectangle([x, y, x + box_w, y + 80], radius=10, fill=fill, outline=col, width=2)
            d.text((x + 10, y + 10), name, fill=WHITE, font=FONT_BOLD)
            d.text((x + 10, y + 50), desc, fill=(200, 200, 200), font=FONT_SM)

            if i > 0 and t > threshold + 0.3:
                prev_x = sx + (i - 1) * (box_w + gap) + box_w
                d.line([(prev_x, y + 40), (x, y + 40)], fill=col, width=3)
                arrow_x = x - 5
                d.polygon([(arrow_x, y + 30), (arrow_x, y + 50), (arrow_x + 10, y + 40)], fill=col)

        if t > 4.0:
            d.text((W // 2 - 150, H - 100), "message decoded ✓", fill=GREEN, font=FONT_BOLD)

        frame = np.array(img)
        frame[::3, :, :] = (frame[::3, :, :] * 0.85).astype(np.uint8)
        return frame

    audio = np.concatenate([_startup_audio(3.0), _cn_chord(6, 1.5)])
    return _clip(make_frame, dur, audio)


def scene_attention_loka():
    """Loka multiplication table as a color matrix."""
    dur = 3.5
    N = 6
    table = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            table[i, j] = (i + j) % N

    def make_frame(t):
        img = Image.new("RGB", (W, H), BLACK)
        d = ImageDraw.Draw(img)

        d.text((W // 2 - 200, 30), "C₆ Multiplication Table", fill=LOKA_GOLD, font=FONT_MD)

        cell = 70
        sx = (W - N * cell) // 2
        sy = 120

        for i in range(N):
            d.text((sx - 40, sy + i * cell + 20), f"C{i}", fill=WHITE, font=FONT_SM)
            d.text((sx + i * cell + 20, sy - 30), f"C{i}", fill=WHITE, font=FONT_SM)

        phase = min(t / 2.0, 1.0)
        n_cells = int(phase * N * N)
        for idx in range(n_cells):
            i, j = idx // N, idx % N
            x, y = sx + j * cell, sy + i * cell
            result = table[i, j]
            col = POLE_COLORS[result % len(POLE_COLORS)]

            if t > 2.5:
                if random.random() < 0.1:
                    col = random.choice(GLITCH_COLORS)
            d.rectangle([x, y, x + cell - 2, y + cell - 2], fill=col)
            d.text((x + 20, y + 20), str(result), fill=BLACK, font=FONT_SM)

        frame = np.array(img)
        if t > 2.5:
            frame = _glitch(frame, (t - 2.5) * 0.6)
            frame = _shake(frame, 8)
        return frame

    audio = np.concatenate([_cn_chord(6, 2.5), _glitch_audio(1.0)])
    return _clip(make_frame, dur, audio)


def scene_stutter_sigma():
    """Stuttering 'Σ → 0' like a broken record."""
    dur = 2.5
    stutter = ["Σ", "Σ →", "Σ →", "Σ → 0", "Σ → 0", "Σ → 0", "Σ → 0 → 0 → 0", "Σ → ∞ ?!"]

    def make_frame(t):
        idx = min(int(t * 4), len(stutter) - 1)
        text = stutter[idx]
        fg = SIGMA_BLUE if idx < 6 else RED
        font = FONT_LG if idx < 6 else FONT_XL
        frame = _text_frame(text, fg=fg, font=font)
        if idx >= 6:
            frame = _shake(frame, 10 + idx * 3)
            frame = _glitch(frame, 0.5)
        return frame

    return _clip(make_frame, dur, _stutter_audio(dur))


def scene_outro():
    """Loka system shutting down."""
    dur = 4.0

    def make_frame(t):
        if t < 1.5:
            frame = _text_frame("loka_light v1.4\nN-Polar Information Flow", fg=WHITE, font=FONT_MD)
        elif t < 2.0:
            frame = _text_frame("ΣDOWN INITIATED", fg=RED, font=FONT_LG)
        elif t < 2.7:
            frame = _text_frame("but Σ was already 0…\nwhat's left to remove?", fg=(120, 120, 150), font=FONT_MD)
        elif t < 3.3:
            progress = (t - 2.7) / 0.6
            img = Image.new("RGB", (W, H), BLACK)
            d = ImageDraw.Draw(img)
            cx, cy = W // 2, H // 2
            rw = int((1 - progress) * W // 2)
            rh = int((1 - progress) * H // 2)
            if rw > 0 and rh > 0:
                d.rectangle([cx - rw, cy - rh, cx + rw, cy + rh], fill=LOKA_PURPLE)
            frame = np.array(img)
        else:
            img = Image.new("RGB", (W, H), BLACK)
            d = ImageDraw.Draw(img)
            br = int(255 * max(0, 1.0 - (t - 3.3) * 2))
            if br > 0:
                d.ellipse([W // 2 - 3, H // 2 - 3, W // 2 + 3, H // 2 + 3], fill=(br, 0, br))
            frame = np.array(img)
        return frame

    audio = np.concatenate([_drone(2.5), _tone(220, 0.5, 0.15), _tone(110, 1.0, 0.1)])
    return _clip(make_frame, dur, audio)


# ── INTERSTITIALS ────────────────────────────────────────────────────────────
def _black(dur=0.15):
    return VideoClip(lambda t: np.zeros((H, W, 3), dtype=np.uint8), duration=dur).with_fps(FPS)


def _static(dur=0.3):
    def f(t):
        return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    audio = _noise(dur, 0.2)
    return VideoClip(f, duration=dur).with_fps(FPS).with_audio(_audio_clip(audio, dur))


# ── ASSEMBLY ─────────────────────────────────────────────────────────────────
def build():
    print("=" * 60)
    print("  LOKA_YTP GENERATOR")
    print("  N-Polar absurdist cinema from pure code")
    print("=" * 60)
    print()

    scenes = []

    def add(name, builder):
        print(f"  rendering: {name}...")
        clip = builder()
        scenes.append(clip)
        if random.random() < 0.4:
            scenes.append(_static(random.uniform(0.1, 0.3)))
        else:
            scenes.append(_black(random.uniform(0.08, 0.2)))

    add("Boot Sequence", scene_boot)
    add("Multipolarity Title", scene_multipolarity_title)
    add("C_n Ring", scene_cn_ring)
    add("Wave Propagation", scene_wave_propagation)
    add("PseudoBlockM", scene_pseudo_block_m)
    add("Flash Montage", scene_flash_montage)
    add("Sigma Cascade", scene_sigma_cascade)
    add("SigmaGuard NX", scene_sigma_guard)
    add("Stutter Sigma", scene_stutter_sigma)
    add("Polarity Scan", scene_polarity_scan)
    add("Matrix Polarities", scene_matrix_polarities)
    add("Cascade Pipeline", scene_cascade_pipeline)
    add("Loka Table", scene_attention_loka)
    add("Quantum Circuit", scene_quantum_circuit)
    add("Witness Pack", scene_witness_pack)
    add("Translation Gap", scene_translation_gap)
    add("Sigma Noise", scene_sigma_noise_compare)
    add("Existential", scene_existential)
    add("Outro", scene_outro)

    print()
    print("  concatenating clips...")
    final = concatenate_videoclips(scenes, method="compose")

    total_dur = final.duration
    print(f"  duration: {total_dur:.1f}s  |  {W}x{H} @ {FPS}fps")
    print()

    # M1 Max: try hardware VideoToolbox encoder, fall back to libx264
    try:
        print("  trying h264_videotoolbox (HW accel)...")
        final.write_videofile(
            OUTPUT,
            fps=FPS,
            codec="h264_videotoolbox",
            audio_codec="aac",
            bitrate="8000k",
            threads=10,
            logger="bar",
        )
    except Exception as e:
        print(f"  VideoToolbox unavailable ({e}), falling back to libx264...")
        final.write_videofile(
            OUTPUT,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=10,
            logger="bar",
        )

    print()
    print("=" * 60)
    print(f"  Done! Output: {OUTPUT}")
    print(f"  File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    build()
