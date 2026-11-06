#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
섀도잉 스튜디오 - PC에서 실행하고 휴대폰으로 사용합니다.

  Windows : '섀도잉 스튜디오.bat' 더블클릭
  Mac     : 터미널에서  python3 shadow_server.py

실행하면
  1) PC 브라우저가 열리고, 화면에 휴대폰용 QR코드가 나옵니다.
  2) 휴대폰 카메라로 QR을 찍으면 휴대폰에서 바로 사용합니다(https 주소라 마이크 사용 가능).
  3) 유튜브 링크를 붙여넣으면 실제 영상 소리를 받아 문장별로 자르고,
     내 녹음을 원본과 비교해 억양, 톤, 빠르기, 강세, 발음을 점수로 보여줍니다.

옵션
  python shadow_server.py --model base.en   # 받아쓰기 모델 (기본 small.en / 빠르게: base.en / 정확하게: medium.en)
  python shadow_server.py --no-tunnel       # 휴대폰 주소 만들지 않기 (PC에서만 사용)
  python shadow_server.py --port 8000
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "shadow_data")
SR = 16000          # 분석용 샘플레이트
HOP = 160           # 10ms
FPS = SR // HOP
MAX_MINUTES = 20
CFG = {"model": os.environ.get("WHISPER_MODEL", "small.en"), "port": 8000}
INFO = {"phone_url": None, "token": None}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


# ───────────────────────── 패키지 준비 ─────────────────────────
REQUIRED = [("yt-dlp", "yt_dlp"), ("faster-whisper", "faster_whisper"), ("numpy", "numpy"),
            ("pycloudflared", "pycloudflared"), ("qrcode", "qrcode")]


def ensure_packages():
    need = [p for p, m in REQUIRED if importlib.util.find_spec(m) is None]
    if need:
        log("처음 실행이라 필요한 프로그램을 설치합니다 (몇 분 걸릴 수 있음): " + ", ".join(need))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *need])
    else:  # 유튜브 변경에 대응하려면 yt-dlp는 항상 최신이 좋음
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "-q", "yt-dlp"],
                           timeout=90, capture_output=True)
        except Exception:
            pass


if __name__ == "__main__":
    ensure_packages()

import numpy as np  # noqa: E402


class UserError(Exception):
    pass


# ───────────────────────── 파일/오디오 유틸 ─────────────────────────
def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav_bytes(b):
    with wave.open(io.BytesIO(b)) as w:
        ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise UserError("녹음 형식을 읽을 수 없습니다.")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR and len(x):
        t_new = np.arange(int(len(x) * SR / rate)) / SR
        x = np.interp(t_new, np.arange(len(x)) / rate, x).astype(np.float32)
    return x


def decode_any(path):
    """영상/음성 파일 -> 16kHz 모노 float32 (av 버전 차이에 안전하게)"""
    import av
    chunks = []
    with av.open(path) as c:
        if not c.streams.audio:
            raise UserError("이 파일에는 소리가 없습니다.")
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for fr in c.decode(audio=0):
            for o in rs.resample(fr) or []:
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None) or []:
            chunks.append(o.to_ndarray().reshape(-1))
    if not chunks:
        raise UserError("소리를 읽지 못했습니다.")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


_audio_cache = {}


def load_audio(vid):
    if vid not in _audio_cache:
        with open(os.path.join(DATA, vid, "audio.wav"), "rb") as f:
            _audio_cache[vid] = read_wav_bytes(f.read())
        while len(_audio_cache) > 3:
            _audio_cache.pop(next(iter(_audio_cache)))
    return _audio_cache[vid]


def lesson_path(vid):
    return os.path.join(DATA, vid, "lesson.json")


def load_lesson(vid):
    p = lesson_path(vid)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def safe_vid(v):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,40}", v or ""):
        raise UserError("잘못된 영상 번호입니다.")
    return v


# ───────────────────────── 받아쓰기(Whisper) ─────────────────────────
_model = None
MODEL_LOCK = threading.Lock()


def transcribe(audio, vad, progress=None, status=None):
    """단어별 [텍스트, 시작초, 끝초] 목록"""
    global _model
    with MODEL_LOCK:
        if _model is None:
            from faster_whisper import WhisperModel
            if status:
                status("받아쓰기 모델 준비 중 (처음 한 번은 다운로드로 몇 분 걸립니다)")
            log(f"받아쓰기 모델 로딩: {CFG['model']}")
            _model = WhisperModel(CFG["model"], device="cpu", compute_type="int8")
        segs, info = _model.transcribe(audio, language="en", word_timestamps=True, vad_filter=vad,
                                       condition_on_previous_text=False, beam_size=5)
        words = []
        for sg in segs:
            for w in sg.words or []:
                t = w.word.strip()
                if t:
                    words.append([t, round(float(w.start), 3), round(float(w.end), 3)])
            if progress and info.duration:
                progress(min(1.0, sg.end / info.duration))
    return words


def make_segments(words, total):
    """단어 타임스탬프를 섀도잉하기 좋은 문장 구간으로 묶기"""
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt[1] - w[2]) if nxt else 99
        n, t = len(cur), w[0]
        sent_end = re.search(r"[.?!…][\"')\]]*$", t) is not None
        if (nxt is None or gap > 1.5 or (sent_end and n >= 3) or (gap > 0.7 and n >= 3) or n >= 18
                or (n >= 10 and t.endswith((",", ";", ":")))):
            groups.append(cur)
            cur = []
    merged = []
    for g in groups:
        if merged and len(g) < 3 and g[0][1] - merged[-1][-1][2] < 0.8 and len(merged[-1]) + len(g) <= 22:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    if len(merged) > 1 and len(merged[0]) < 3 and merged[1][0][1] - merged[0][-1][2] < 0.8:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    segs = []
    for i, g in enumerate(merged):
        s, e = g[0][1] - 0.15, g[-1][2] + 0.3
        if i > 0:
            s = min(g[0][1], max(s, merged[i - 1][-1][2] + 0.02))
        if i + 1 < len(merged):
            e = max(g[-1][2], min(e, merged[i + 1][0][1] - 0.02))
        s, e = max(0.0, s), min(total, e)
        segs.append({"s": round(s, 2), "e": round(e, 2), "text": " ".join(w[0] for w in g),
                     "words": [[w[0], round(w[1], 3), round(w[2], 3)] for w in g]})
    return segs


# ───────────────────────── 영상 → 학습 데이터 ─────────────────────────
YT_ID = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([A-Za-z0-9_-]{11})")
JOBS = {}


def video_key(url):
    m = YT_ID.search(url)
    if m:
        return m.group(1)
    return "u_" + hashlib.sha1(url.encode()).hexdigest()[:12]


def download(url, workdir, status):
    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                status(f"영상 소리 받는 중 {int(d.get('downloaded_bytes', 0) * 100 / tot)}%")

    opts = {"format": "bestaudio/best", "outtmpl": os.path.join(workdir, "src.%(ext)s"), "quiet": True,
            "no_warnings": True, "noplaylist": True, "progress_hooks": [hook]}
    status("영상 정보 확인 중")
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=False)
            if info.get("_type") == "playlist":
                raise UserError("재생목록이 아니라 영상 하나의 링크를 넣어 주세요.")
            dur = info.get("duration") or 0
            if dur > MAX_MINUTES * 60:
                raise UserError(f"{MAX_MINUTES}분 이하 영상만 가능합니다. (이 영상은 {int(dur // 60)}분)")
            y.download([info.get("webpage_url") or url])
    except UserError:
        raise
    except Exception as e:
        msg = str(e)
        if "Sign in" in msg or "bot" in msg:
            raise UserError("유튜브가 잠시 다운로드를 막았습니다. 몇 분 뒤 다시 시도하거나 프로그램을 다시 실행해 주세요.")
        if "Private" in msg or "unavailable" in msg:
            raise UserError("비공개이거나 삭제된 영상입니다.")
        raise UserError("영상 소리를 받지 못했습니다: " + re.sub(r"\x1b\[[0-9;]*m", "", msg)[:200])
    files = [f for f in os.listdir(workdir) if f.startswith("src.") and not f.endswith((".part", ".ytdl"))]
    if not files:
        raise UserError("영상 소리 파일을 찾지 못했습니다.")
    return info.get("title") or url, os.path.join(workdir, files[0])


def start_job(vid, url=None, file=None, title=None):
    if JOBS.get(vid, {}).get("state") == "running":
        return
    JOBS[vid] = {"state": "running", "msg": "준비 중", "p": 0}
    threading.Thread(target=run_job, args=(vid, url, file, title), daemon=True).start()


def run_job(vid, url, file, title):
    d = os.path.join(DATA, vid)
    os.makedirs(d, exist_ok=True)

    def st(msg, p=None):
        JOBS[vid]["msg"] = msg
        if p is not None:
            JOBS[vid]["p"] = p

    src = file
    try:
        if url:
            title, src = download(url, d, st)
        st("소리 변환 중")
        try:
            audio = decode_any(src)
        except UserError:
            raise
        except Exception:
            raise UserError("이 파일에서 소리를 읽지 못했습니다. 다른 형식(mp4, m4a, mp3)으로 올려 주세요.")
        dur = len(audio) / SR
        if dur < 1:
            raise UserError("소리가 너무 짧습니다.")
        if dur > MAX_MINUTES * 60:
            raise UserError(f"{MAX_MINUTES}분 이하만 가능합니다.")
        write_wav(os.path.join(d, "audio.wav"), audio)
        st("받아쓰기 중 0%", 0)
        words = transcribe(audio, vad=True, status=st,
                           progress=lambda p: st(f"받아쓰기 중 {int(p * 100)}%", p))
        if not words:
            raise UserError("영어 대사를 찾지 못했습니다.")
        segs = make_segments(words, dur)
        lesson = {"id": vid, "title": title or vid, "source": url or "upload", "duration": round(dur, 2),
                  "created": time.strftime("%Y-%m-%d %H:%M"), "youtube": bool(url and YT_ID.search(url)),
                  "segments": segs}
        with open(lesson_path(vid), "w", encoding="utf-8") as f:
            json.dump(lesson, f, ensure_ascii=False)
        JOBS[vid] = {"state": "done"}
        log(f"준비 완료: {title} ({len(segs)}문장)")
    except UserError as e:
        JOBS[vid] = {"state": "error", "msg": str(e)}
        log(f"실패: {e}")
    except Exception as e:
        traceback.print_exc()
        JOBS[vid] = {"state": "error", "msg": f"오류: {type(e).__name__}: {e}"}
    finally:
        if src and os.path.exists(src) and os.path.basename(src).startswith("src."):
            try:
                os.remove(src)
            except Exception:
                pass
        if JOBS[vid].get("state") == "error" and not os.path.exists(lesson_path(vid)):
            shutil.rmtree(d, ignore_errors=True)


# ───────────────────────── 발화 분석 ─────────────────────────
def frames_of(x, size, hop):
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _drop_short_runs(mask, min_len):
    out = mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _nan_median_smooth(v, k=5):
    pad = k // 2
    p = np.pad(v, pad, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(p, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(win, axis=1)
    m[np.isnan(v)] = np.nan
    return m


def features(x):
    """10ms 간격: 시간, 피치(반음, 화자 중앙값 기준), 음량(dB)"""
    x = np.asarray(x, np.float32)
    W, tmin, tmax = 480, SR // 450, SR // 60
    fr = frames_of(x, W + tmax, HOP)
    n = len(fr)
    x0 = fr[:, :W]
    d = np.zeros((n, tmax + 1), np.float32)
    for tau in range(1, tmax + 1):
        df = x0 - fr[:, tau:tau + W]
        d[:, tau] = np.einsum("ij,ij->i", df, df)
    cum = np.cumsum(d[:, 1:], axis=1)
    cm = np.ones_like(d)
    cm[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1) / np.maximum(cum, 1e-9)
    sub = cm[:, tmin:]
    f0 = np.zeros(n)
    for i in range(n):
        row = sub[i]
        below = np.flatnonzero(row < 0.15)
        if below.size:
            t = int(below[0])
            while t + 1 < row.size and row[t + 1] < row[t]:
                t += 1
        else:
            t = int(np.argmin(row))
            if row[t] > 0.3:
                continue
        tt = float(t + tmin)
        if 0 < t < row.size - 1:
            a, b, c = row[t - 1], row[t], row[t + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                tt += 0.5 * (a - c) / den
        f0[i] = SR / tt
    e = frames_of(x, 400, HOP)
    db = 10 * np.log10(np.mean(e.astype(np.float64) ** 2, axis=1) + 1e-10)
    db = np.pad(db, (0, max(0, n - len(db))), constant_values=-100)[:n]
    voiced = (f0 > 0) & (db > db.max() - 30)
    if voiced.sum() > 5:
        med = np.median(f0[voiced])
        voiced &= (f0 < med * 1.9) & (f0 > med * 0.52)
    voiced = _drop_short_runs(voiced, 3)
    st = np.full(n, np.nan)
    if voiced.sum() > 3:
        med = np.median(f0[voiced])
        st[voiced] = 12 * np.log2(f0[voiced] / med)
        st = _nan_median_smooth(st, 5)
    times = (np.arange(n) * HOP + W / 2) / SR
    return times, st, db


def _norm(w):
    w = w.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", w).strip("'")


def _clean(w):
    return re.sub(r"^[^\w']+|[^\w']+$", "", w)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def _wsim(a, b):
    if a == b:
        return 1.0
    A, B = a.replace("'", ""), b.replace("'", "")
    if A == B:
        return 1.0
    if min(len(A), len(B)) <= 2:
        return 0.0
    return 1 - _lev(A, B) / max(len(A), len(B))


def align_words(ref, usr):
    """ref 각 단어에 대해 (usr 인덱스 또는 None, 'ok'|'close'|'miss')"""
    n, m = len(ref), len(usr)
    D = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), np.int8)
    D[1:, 0] = np.arange(1, n + 1)
    P[1:, 0] = 2
    D[0, 1:] = np.arange(1, m + 1) * 0.7
    P[0, 1:] = 3
    S = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _wsim(ref[i - 1], usr[j - 1])
            S[i, j] = s
            a = D[i - 1, j - 1] + ((1 - s) if s >= 0.7 else 2)
            b = D[i - 1, j] + 1
            c = D[i, j - 1] + 0.7
            if a <= b and a <= c:
                D[i, j], P[i, j] = a, 1
            elif b <= c:
                D[i, j], P[i, j] = b, 2
            else:
                D[i, j], P[i, j] = c, 3
    out = [(None, "miss")] * n
    i, j = n, m
    while i > 0 or j > 0:
        p = P[i, j]
        if p == 1:
            s = S[i, j]
            out[i - 1] = (j - 1, "ok" if s == 1 else "close" if s >= 0.7 else "miss")
            i, j = i - 1, j - 1
        elif p == 2 or j == 0:
            out[i - 1] = (None, "miss")
            i -= 1
        else:
            j -= 1
    return [(j if st != "miss" else None, st, usr[j] if j is not None else None) for j, st in out]


def _clip01(v):
    return float(max(0.0, min(1.0, v)))


def _ratio_score(r, tol, span):
    return 100 * _clip01(1 - max(0.0, abs(np.log2(max(r, 1e-3))) - tol) / span)


def _span(times, db):
    on = np.flatnonzero(db > db.max() - 25)
    if not on.size:
        return 0.0, float(times[-1])
    return float(times[on[0]]), float(times[on[-1]])


def _word_mask(times, words):
    m = np.zeros(len(times), bool)
    for _, s, e in words:
        m |= (times >= s - 0.03) & (times <= e + 0.03)
    return m


def _z(v):
    v = np.array(v, float)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    v = np.where(np.isnan(v), np.nanmean(v), v)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-6 else np.zeros_like(v)


def _prominence(times, st, db, words):
    E, P, Dn = [], [], []
    for w, s, e in words:
        m = (times >= s) & (times <= e)
        if not m.any():
            m = np.zeros(len(times), bool)
            m[int(np.argmin(np.abs(times - (s + e) / 2)))] = True
        E.append(np.percentile(db[m], 80))
        pv = st[m]
        pv = pv[~np.isnan(pv)]
        P.append(pv.max() if pv.size else np.nan)
        Dn.append((e - s) / (0.05 * max(1, len(_norm(w))) + 0.12))
    return _z(E) + 0.6 * _z(P) + 0.4 * _z(Dn)


def _corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze(ref_x, ref_words, usr_x, usr_words):
    ref_words = [w for w in ref_words if _norm(w[0])]
    usr_words = [w for w in usr_words if _norm(w[0])]
    ut, ust, udb = features(usr_x)
    if udb.max() < -58 or (udb > udb.max() - 25).sum() < 25:
        return {"ok": False, "reason": "목소리가 거의 녹음되지 않았습니다. 마이크 가까이에서 또렷하게 다시 말해 주세요."}
    rt, rst, rdb = features(ref_x)

    # 1) 단어 맞추기 (발음/전달력)
    al = align_words([_norm(w[0]) for w in ref_words], [_norm(w[0]) for w in usr_words])
    pairs = [(i, j) for i, (j, s, _) in enumerate(al) if j is not None]
    n_ok = sum(1 for _, s, _ in al if s == "ok")
    n_close = sum(1 for _, s, _ in al if s == "close")
    s_acc = 100 * (n_ok + 0.6 * n_close) / max(1, len(ref_words))

    # 2) 시간 정렬 (원본 시간 -> 내 시간)
    anchors = []
    for i, j in pairs:
        anchors += [(ref_words[i][1], usr_words[j][1]), (ref_words[i][2], usr_words[j][2])]
    anchors.sort()
    A = []
    for a, u in anchors:
        if not A or (a > A[-1][0] + 0.005 and u > A[-1][1] + 0.005):
            A.append((a, u))
    r0, r1 = (ref_words[0][1], ref_words[-1][2]) if ref_words else _span(rt, rdb)
    u0, u1 = (usr_words[0][1], usr_words[-1][2]) if usr_words else _span(ut, udb)
    if len(A) < 2:
        A = [(r0, u0), (r1, u1)]
    ra, ua = np.array([a for a, _ in A]), np.array([u for _, u in A])

    def to_user(t):
        return np.interp(t, ra, ua)

    rmask = _word_mask(rt, ref_words) if ref_words else np.ones(len(rt), bool)
    umask = _word_mask(ut, usr_words) if usr_words else (ut >= u0) & (ut <= u1)
    rs = np.where(rmask, rst, np.nan)
    us = np.where(umask, ust, np.nan)
    if np.any(~np.isnan(rs)):
        rs = rs - np.nanmedian(rs)
    if np.any(~np.isnan(us)):
        us = us - np.nanmedian(us)

    sel = (rt >= r0 - 0.05) & (rt <= r1 + 0.05)
    tr = rt[sel]
    a = rs[sel]
    ui = np.clip(np.round((to_user(tr) - ut[0]) * FPS).astype(int), 0, len(ut) - 1)
    b = us[ui]
    both = ~np.isnan(a) & ~np.isnan(b)
    fb = []   # (분야, 문구)

    # 3) 억양 (피치 곡선 모양)
    s_int = None
    if both.sum() >= 12:
        aa, bb = a[both] - a[both].mean(), b[both] - b[both].mean()
        r = _corr(aa, bb)
        mae = float(np.mean(np.abs(aa - bb)))
        s_int = 0.65 * 100 * _clip01((r + 0.3) / 1.1) + 0.35 * 100 * _clip01(1 - (mae - 1.0) / 4)
        # 문장 끝 억양: 마지막 단어가 앞 단어보다 올라가는지/내려가는지 비교
        def wmean(t, arr, s0, e0):
            v = arr[(t >= s0) & (t <= e0)]
            v = v[~np.isnan(v)]
            return float(v.mean()) if v.size >= 3 else np.nan
        tail = [(i, j) for i, j in pairs if i >= len(ref_words) - 2]
        if tail:
            i, j = tail[-1]
            prev = [(i2, j2) for i2, j2 in pairs if i2 < i]
            if prev:
                i2, j2 = prev[-1]
                rd = wmean(rt, rs, *ref_words[i][1:]) - wmean(rt, rs, *ref_words[i2][1:])
                ud = wmean(ut, us, *usr_words[j][1:]) - wmean(ut, us, *usr_words[j2][1:])
                lw = _clean(ref_words[i][0])
                if not (np.isnan(rd) or np.isnan(ud)):
                    if rd <= -2 and ud >= 1.5:
                        s_int -= 15
                        fb.append(("억양", f"문장 끝 억양이 반대입니다. 원본은 '{lw}'에서 내려가며 끝나는데 올렸습니다. 마지막을 툭 떨어뜨리세요."))
                    elif rd >= 2 and ud <= 0:
                        s_int -= 15
                        fb.append(("억양", f"원본은 '{lw}'에서 끝을 올립니다(질문, 놀람). 마지막 단어에서 음을 올려 주세요."))
        s_int = max(0.0, s_int)
        # 단어별 높낮이 차이
        devs = []
        for i, j in pairs:
            m = (tr >= ref_words[i][1]) & (tr <= ref_words[i][2]) & both
            if m.sum() >= 4:
                devs.append((float(np.mean((b[m] - b[both].mean()) - (a[m] - a[both].mean()))), i))
        if devs:
            dv, i = max(devs, key=lambda x: abs(x[0]))
            if abs(dv) >= 2.5:
                fb.append(("억양", f"'{_clean(ref_words[i][0])}'에서 음이 원본보다 {'높습니다' if dv > 0 else '낮습니다'} "
                                  f"(약 {abs(dv):.0f}반음). 원본 그래프 모양을 보고 맞춰 보세요."))
        if r < 0.3:
            fb.append(("억양", "전체 높낮이 흐름이 원본과 다릅니다. 0.75x로 들으며 어디서 올라가고 내려가는지 손으로 따라 그려 보세요."))
    else:
        fb.append(("억양", "목소리 높낮이를 충분히 잡지 못했습니다. 모음을 조금 더 길고 또렷하게 소리 내 보세요."))

    # 4) 톤 (음높이 폭, 세기 변화 = 표현력)
    s_tone = None
    rv, uv = rs[~np.isnan(rs)], us[~np.isnan(us)]
    if rv.size > 10 and uv.size > 10:
        rr = float(np.percentile(rv, 90) - np.percentile(rv, 10))
        ur = float(np.percentile(uv, 90) - np.percentile(uv, 10))
        pr = (ur + 0.5) / (rr + 0.5)
        rde = float(np.std(rdb[rmask & (rdb > rdb.max() - 35)])) if (rmask & (rdb > rdb.max() - 35)).sum() > 5 else 1
        ude = float(np.std(udb[umask & (udb > udb.max() - 35)])) if (umask & (udb > udb.max() - 35)).sum() > 5 else 1
        er = (ude + 0.5) / (rde + 0.5)
        s_tone = 0.7 * _ratio_score(pr, 0.1, 1.5) + 0.3 * _ratio_score(er, 0.1, 1.5)
        if pr < 0.7:
            fb.append(("톤", f"말이 단조롭습니다. 원본은 음높이가 {rr:.0f}반음 폭으로 움직이는데 {ur:.0f}반음입니다. 강조 단어에서 확 올려 주세요."))
        elif pr > 1.6:
            fb.append(("톤", "억양이 원본보다 과장됐습니다. 오르내림을 조금 줄여 자연스럽게 말해 보세요."))
        if er < 0.7:
            fb.append(("톤", "소리 크기 변화가 작습니다. 중요한 단어는 크게, 나머지는 흘리듯 약하게 말하세요."))

    # 5) 빠르기 (길이, 끊김, 늘어짐)
    if len(pairs) >= 2:
        (i0, j0), (i1, j1) = pairs[0], pairs[-1]
        rdur, udur = ref_words[i1][2] - ref_words[i0][1], usr_words[j1][2] - usr_words[j0][1]
    else:
        rdur, udur = r1 - r0, u1 - u0
    rdur, udur = max(rdur, 0.2), max(udur, 0.2)
    ratio = udur / rdur
    s_speed = _ratio_score(ratio, 0.08, 1.0)
    pauses, drags = [], []
    for (i, j), (i2, j2) in zip(pairs, pairs[1:]):
        if i2 == i + 1:
            ug = usr_words[j2][1] - usr_words[j][2]
            rg = ref_words[i2][1] - ref_words[i][2]
            if ug > 0.35 and ug - rg > 0.3:
                pauses.append((ref_words[i][0], ref_words[i2][0], ug))
    for i, j in pairs:
        rd, ud = ref_words[i][2] - ref_words[i][1], usr_words[j][2] - usr_words[j][1]
        if rd > 0.05 and ud > 0.35 and (ud / rd) / ratio > 1.8:
            drags.append(ref_words[i][0])
    s_speed = max(0.0, s_speed - 12 * len(pauses))
    if ratio > 1.15:
        fb.append(("빠르기", f"원본보다 {ratio:.1f}배 느립니다 (원본 {rdur:.1f}초, 나 {udur:.1f}초). 단어를 이어 붙여 한 호흡에 말해 보세요."))
    elif ratio < 0.87:
        fb.append(("빠르기", f"원본보다 빠릅니다 (원본 {rdur:.1f}초, 나 {udur:.1f}초). 강조 단어에서 조금 더 머무르세요."))
    for w1, w2, g in pauses[:2]:
        fb.append(("빠르기", f"'{_clean(w1)}'와 '{_clean(w2)}' 사이에서 {g:.1f}초 끊겼습니다. 원본은 붙여서(연음으로) 말합니다."))
    if drags:
        fb.append(("빠르기", "'" + "', '".join(_clean(w) for w in drags[:3]) + "'를 너무 길게 끌었습니다."))

    # 6) 강세 (어떤 단어에 힘을 주는지)
    ref_prom_all = _prominence(rt, rst, rdb, ref_words) if ref_words else np.array([])
    ref_stress = set()
    if len(ref_words) >= 3:
        thr = max(0.5, float(np.quantile(ref_prom_all, 0.7)))
        ref_stress = {i for i, p in enumerate(ref_prom_all) if p >= thr}
    me_stress = {}
    s_stress = None
    if len(pairs) >= 3:
        pr_ = _prominence(rt, rst, rdb, [ref_words[i] for i, _ in pairs])
        pu_ = _prominence(ut, ust, udb, [usr_words[j] for _, j in pairs])
        s_stress = 100 * _clip01((_corr(pr_, pu_) + 0.3) / 1.0)
        uthr = max(0.5, float(np.quantile(pu_, 0.7)))
        missed, over = [], []
        for k, (i, j) in enumerate(pairs):
            me_stress[i] = bool(pu_[k] >= uthr)
            if i in ref_stress and pu_[k] < 0:
                missed.append(ref_words[i][0])
            if pu_[k] > 1.0 and pr_[k] < 0 and i not in ref_stress:
                over.append(ref_words[i][0])
        for w in missed[:2]:
            fb.append(("강세", f"원본은 '{_clean(w)}'에 힘을 줍니다. 그 단어를 더 크고, 높고, 조금 길게 말하세요."))
        if over:
            fb.append(("강세", f"'{_clean(over[0])}'는 원본에서 약하게 지나가는 단어인데 세게 말했습니다."))

    # 7) 발음 피드백
    miss = [_clean(ref_words[i][0]) for i, (_, s, _) in enumerate(al) if s == "miss"]
    close = [f"{_clean(ref_words[i][0])} → {h}" for i, (_, s, h) in enumerate(al) if s == "close"]
    if miss:
        fb.append(("발음", "알아듣지 못한 단어: " + ", ".join(miss[:6]) + (" …" if len(miss) > 6 else "")))
    if close:
        fb.append(("발음", "다르게 들린 단어: " + ", ".join(close[:4])))

    scores = {"intonation": s_int, "tone": s_tone, "speed": s_speed, "stress": s_stress, "accuracy": s_acc}
    weights = {"intonation": .3, "tone": .15, "speed": .2, "stress": .2, "accuracy": .15}
    tot = sum(weights[k] for k, v in scores.items() if v is not None)
    overall = sum(weights[k] * v for k, v in scores.items() if v is not None) / tot if tot else 0
    overall = min(overall, s_acc + 10)
    if s_acc < 40:
        fb.insert(0, ("발음", "문장의 상당 부분이 인식되지 않아 다른 점수의 정확도가 낮습니다. 먼저 단어를 정확히 따라 말해 보세요."))

    cat_key = {"억양": "intonation", "톤": "tone", "빠르기": "speed", "강세": "stress", "발음": "accuracy"}
    fb.sort(key=lambda f: scores.get(cat_key[f[0]]) if scores.get(cat_key[f[0]]) is not None else 50)
    if overall >= 90 and all(v is None or v >= 80 for v in scores.values()):
        fb.insert(0, ("좋아요", "원본과 거의 같습니다. 이제 원본과 동시에 소리 내는 섀도잉(겹쳐 말하기)으로 연습해 보세요."))
    elif not fb:
        fb.append(("좋아요", "큰 문제는 없습니다. 원본을 한 번 더 듣고 리듬을 더 똑같이 맞춰 보세요."))

    # 그래프 데이터 (원본 시간축 기준)
    step = max(1, len(tr) // 240)
    rnd = lambda v: None if np.isnan(v) else round(float(v), 2)  # noqa: E731
    chart = {"t": [round(float(t - r0), 3) for t in tr[::step]], "ref": [rnd(v) for v in a[::step]],
             "me": [rnd(v) for v in b[::step]],
             "words": [[_clean(w), round(s - r0, 3), round(e - r0, 3)] for w, s, e in ref_words]}
    words_view = [{"w": w[0], "st": al[i][1], "rs": i in ref_stress, "ms": me_stress.get(i)} for i, w in enumerate(ref_words)]
    rnd1 = lambda v: None if v is None else int(round(v))  # noqa: E731
    return {"ok": True, "overall": int(round(overall)), "scores": {k: rnd1(v) for k, v in scores.items()},
            "feedback": [{"cat": c, "text": t} for c, t in fb[:7]], "chart": chart, "words": words_view,
            "heard": " ".join(w[0] for w in usr_words),
            "speed": {"ref": round(rdur, 2), "me": round(udur, 2), "ratio": round(ratio, 2)}}


def analyze_request(vid, seg_i, wav_bytes):
    lesson = load_lesson(vid)
    if not lesson:
        raise UserError("학습 데이터를 찾지 못했습니다.")
    seg = lesson["segments"][seg_i]
    audio = load_audio(vid)
    s, e = seg["s"], seg["e"]
    ref_x = audio[int(s * SR):int(e * SR)]
    ref_words = [[w, ws - s, we - s] for w, ws, we in seg["words"]]
    usr = read_wav_bytes(wav_bytes)
    if len(usr) < SR * 0.3:
        return {"ok": False, "reason": "녹음이 너무 짧습니다."}
    usr = usr[:SR * 40]
    peak = float(np.max(np.abs(usr))) if len(usr) else 0
    if peak > 0.001:
        usr = usr * min(20.0, 0.5 / peak)
    t, _, db = features(usr)
    if db.max() < -58 or (db > db.max() - 25).sum() < 25:
        return {"ok": False, "reason": "목소리가 거의 녹음되지 않았습니다. 마이크 권한과 볼륨을 확인해 주세요."}
    a0, a1 = _span(t, db)
    lo, hi = max(0, int((a0 - 0.3) * SR)), min(len(usr), int((a1 + 0.4) * SR))
    usr = usr[lo:hi]
    usr_words = transcribe(usr, vad=False)
    return analyze(ref_x, ref_words, usr, usr_words)


# ───────────────────────── 웹 서버 ─────────────────────────
def list_videos():
    out = []
    if os.path.isdir(DATA):
        for v in os.listdir(DATA):
            ls = load_lesson(v) if os.path.isdir(os.path.join(DATA, v)) else None
            if ls:
                out.append({"id": v, "title": ls["title"], "n": len(ls["segments"]), "created": ls.get("created", ""),
                            "youtube": ls.get("youtube", False), "duration": ls.get("duration")})
    out.sort(key=lambda x: x["created"], reverse=True)
    return out


def qr_svg(text):
    try:
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
        s = img.to_string()
        return s.decode() if isinstance(s, bytes) else s
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "ShadowStudio"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _q(self):
        u = urllib.parse.urlparse(self.path)
        return u.path, dict(urllib.parse.parse_qsl(u.query))

    def _authed(self, q):
        return secrets.compare_digest(q.get("k", ""), INFO["token"] or "")

    def do_GET(self):
        path, q = self._q()
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if not self._authed(q):
            return self._send(403, {"error": "접속 키가 맞지 않습니다. PC 화면의 QR코드로 다시 접속해 주세요."})
        try:
            if path.startswith("/media/") and path.endswith(".wav"):
                vid = safe_vid(path[7:-4])
                p = os.path.join(DATA, vid, "audio.wav")
                if not os.path.exists(p):
                    return self._send(404, {"error": "없음"})
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "audio/wav")
            if path == "/api/info":
                url = INFO["phone_url"]
                return self._send(200, {"phone_url": url, "qr": qr_svg(url) if url else None, "model": CFG["model"]})
            if path == "/api/videos":
                return self._send(200, {"videos": list_videos()})
            if path == "/api/load":
                url = (q.get("url") or "").strip()
                if not re.match(r"https?://", url):
                    raise UserError("영상 링크(https://...)를 붙여넣어 주세요.")
                vid = video_key(url)
                if load_lesson(vid):
                    return self._send(200, {"state": "done", "id": vid})
                start_job(vid, url=url)
                return self._send(200, {"state": "running", "id": vid})
            if path == "/api/status":
                vid = safe_vid(q.get("id"))
                if load_lesson(vid) and JOBS.get(vid, {}).get("state") != "running":
                    return self._send(200, {"state": "done", "id": vid})
                return self._send(200, dict(JOBS.get(vid, {"state": "error", "msg": "작업이 없습니다."}), id=vid))
            if path == "/api/lesson":
                ls = load_lesson(safe_vid(q.get("id")))
                return self._send(200, ls) if ls else self._send(404, {"error": "학습 데이터가 없습니다."})
            if path == "/api/delete":
                vid = safe_vid(q.get("id"))
                shutil.rmtree(os.path.join(DATA, vid), ignore_errors=True)
                _audio_cache.pop(vid, None)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "없는 주소"})
        except UserError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        path, q = self._q()
        if not self._authed(q):
            return self._send(403, {"error": "접속 키가 맞지 않습니다."})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if path == "/api/analyze":
                if n <= 0 or n > 20_000_000:
                    raise UserError("녹음 파일 크기가 올바르지 않습니다.")
                body = self.rfile.read(n)
                res = analyze_request(safe_vid(q.get("id")), int(q.get("seg", 0)), body)
                return self._send(200, res)
            if path == "/api/upload":
                if n <= 0 or n > 500_000_000:
                    raise UserError("파일은 500MB 이하만 가능합니다.")
                name = os.path.basename(q.get("name") or "upload")
                ext = os.path.splitext(name)[1][:6] or ".bin"
                h = hashlib.sha1()
                os.makedirs(DATA, exist_ok=True)
                tmp = os.path.join(DATA, f"_up_{secrets.token_hex(4)}{ext}")
                with open(tmp, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        h.update(chunk)
                        f.write(chunk)
                        left -= len(chunk)
                vid = "f_" + h.hexdigest()[:12]
                if load_lesson(vid):
                    os.remove(tmp)
                    return self._send(200, {"state": "done", "id": vid})
                d = os.path.join(DATA, vid)
                os.makedirs(d, exist_ok=True)
                src = os.path.join(d, "src" + ext)
                shutil.move(tmp, src)
                start_job(vid, file=src, title=os.path.splitext(name)[0])
                return self._send(200, {"state": "running", "id": vid})
            return self._send(404, {"error": "없는 주소"})
        except UserError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def load_token():
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "token.txt")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(secrets.token_urlsafe(9))
    with open(p) as f:
        return f.read().strip()


def start_tunnel(port):
    try:
        from pycloudflared import try_cloudflare
        log("휴대폰용 주소 만드는 중...")
        urls = try_cloudflare(port=port, verbose=False)
        url = f"{urls.tunnel}/?k={INFO['token']}"
        INFO["phone_url"] = url
        with open(os.path.join(HERE, "휴대폰 주소.txt"), "w", encoding="utf-8") as f:
            f.write(url + "\n")
        print("\n" + "=" * 60)
        print(" 휴대폰에서 이 주소로 접속하세요 (PC 화면의 QR을 찍어도 됩니다)")
        print(" " + url)
        print("=" * 60)
        try:
            import qrcode
            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.print_ascii(invert=True)
        except Exception:
            pass
        print(" ※ 주소는 실행할 때마다 바뀝니다. 이 창을 닫으면 휴대폰에서도 꺼집니다.\n", flush=True)
    except Exception as e:
        log(f"휴대폰용 주소를 만들지 못했습니다({type(e).__name__}). PC에서는 그대로 사용할 수 있습니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--port", type=int, default=CFG["port"])
    ap.add_argument("--no-tunnel", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    CFG["model"], CFG["port"] = a.model, a.port
    INFO["token"] = load_token()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError:
        log(f"{a.port}번 포트가 이미 사용 중입니다. 이미 실행 중이면 그 창을 사용하세요.")
        return
    local = f"http://localhost:{a.port}/?k={INFO['token']}"
    log(f"섀도잉 스튜디오 실행 중 → {local}")
    log(f"받아쓰기 모델: {a.model}  (이 창을 닫으면 종료)")
    if not a.no_tunnel:
        threading.Thread(target=start_tunnel, args=(a.port,), daemon=True).start()
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(local)).start()
    threading.Thread(target=lambda: _warm_model(), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("종료합니다.")


def _warm_model():
    try:
        transcribe(np.zeros(SR, np.float32), vad=False)
        log("받아쓰기 모델 준비 완료")
    except Exception as e:
        log(f"받아쓰기 모델을 불러오지 못했습니다: {e}")


if __name__ == "__main__":
    main()
