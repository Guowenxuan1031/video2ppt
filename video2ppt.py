#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2ppt.py —— 把课堂录屏（PPT 讲解视频）自动整理成 PPT

思路（只用 ffmpeg + numpy + Pillow + python-pptx，纯本地、不需要 GPU / 深度学习模型）：

1. 以 1 帧/秒 抽样整段视频，逐帧计算"画面是否静止"（相邻帧平均差异）。
2. 静止的连续区间就是一个"候选页面"：PPT 页面不动，而播放视频、动画、切换幻灯片
   都会让画面明显变化，因此视频内容会被自然排除。
   老师在同一页上长期讲解（含逐步出字）也属于"静止区间"，取区间末尾的一帧，
   正好拿到这一页最后、最完整的样子。
3. 每帧统计"浅色页面占比"、"文字/边缘密度"、"深色文字占比"、"背景亮度"，
   过滤掉桌面、黑屏、"无信号"、播放器等非 PPT 画面。
4. 对候选帧重定位幻灯片页面矩形并裁剪（全屏放映时去掉黑边，编辑窗口中只保留
   幻灯片画布），于是"缩小放映的 PPT"和"全屏放映的 PPT"都能得到干净的一页。
5. 用 480x360 灰度图（允许几像素平移对齐）的差异比例去重：老师回顾同一页时只保留一次，
   并且若后面出现的版本更完整/更清晰，则替换前面那张（保持原来的顺序位置）。
6. 按时间顺序把截图贴进 pptx（页面比例按多数截图自动选择 4:3 / 16:9）。

依赖（本地安装即可，无需联网模型）：
    pip install numpy pillow python-pptx
    ffmpeg 需要在本机 PATH 中（Windows 可下载 https://ffmpeg.org）

用法：
    python video2ppt.py 录屏.mp4 -o 整理后.pptx
    python video2ppt.py 录屏.mp4 -o 整理后.pptx --debug --keep-all

常用参数：
    --fps 1              抽样帧率（越大越细，速度越慢）
    --static-thr 2.0     静止判定阈值（灰度差，0-255；画面抖动大时调大）
    --min-sec 3          一个页面至少静止多少秒才收录
    --dedupe-thr 0.04    去重阈值（两页不同像素比例，越小越严格）
    --no-crop            不裁剪，直接整屏截图
    --keep-all           保存所有候选截图 + 候选拼图，便于人工检查
    --start / --duration 只分析一段（调试用，单位秒）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
W_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def run_quiet(cmd: List[str]) -> str:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError("命令失败: %s\n%s" % (" ".join(cmd), r.stderr[-2000:]))
    return r.stdout


def probe_video(path: str) -> Tuple[int, int, float]:
    """返回 (宽, 高, 时长秒)。"""
    out = run_quiet([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", path,
    ])
    j = json.loads(out)
    st = j["streams"][0]
    dur = float(j.get("format", {}).get("duration") or 0.0)
    if dur <= 0:
        dur = float(st.get("duration") or 0.0)
    return int(st["width"]), int(st["height"]), dur


def analysis_size(width: int, height: int, aw: int = 640) -> Tuple[int, int]:
    """分析用的缩小尺寸，宽固定，高按比例并对齐到 8 的倍数。"""
    ah = int(round(aw * height / float(width)))
    ah = max(8, (ah // 8) * 8)
    return aw, ah


def iter_frames(path: str, fps: float, aw: int, ah: int, start: float = 0.0,
                duration: Optional[float] = None):
    """按 fps 抽样，逐帧产出 RGB numpy 数组。"""
    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={aw}:{ah}:flags=bilinear",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = aw * ah * 3
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(ah, aw, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# ----------------------------------------------------------------------------
# 幻灯片页面定位
# ----------------------------------------------------------------------------
def largest_component_bbox(mask: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """8 邻域连通域里面积最大的一个，返回 (x0,y0,x1,y1) 与面积。"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None, 0
    seen = np.zeros((h, w), dtype=bool)
    best_area = 0
    best = None
    for y_start, x_start in zip(ys, xs):
        if seen[y_start, x_start]:
            continue
        stack = [(int(y_start), int(x_start))]
        seen[y_start, x_start] = True
        area = 0
        ymin = ymax = int(y_start)
        xmin = xmax = int(x_start)
        while stack:
            y, x = stack.pop()
            area += 1
            if y < ymin:
                ymin = y
            elif y > ymax:
                ymax = y
            if x < xmin:
                xmin = x
            elif x > xmax:
                xmax = x
            y0 = y - 1 if y > 0 else 0
            y1 = y + 1 if y + 1 < h else h - 1
            x0 = x - 1 if x > 0 else 0
            x1 = x + 1 if x + 1 < w else w - 1
            for ny in range(y0, y1 + 1):
                row = mask[ny]
                srow = seen[ny]
                for nx in range(x0, x1 + 1):
                    if row[nx] and not srow[nx]:
                        srow[nx] = True
                        stack.append((ny, nx))
        if area > best_area:
            best_area = area
            best = (xmin, ymin, xmax + 1, ymax + 1)
    return best, best_area


def refine_box(gray: np.ndarray, bg: float, box: List[int], band: int = 20) -> List[int]:
    """在一帧 640 宽灰度图上把页面矩形的四条边对齐到真实边界（±band 像素内搜索）。"""
    h, w = gray.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    near = np.abs(gray - bg) <= 8.0

    ry0 = min(h - 2, max(0, y0 + 12))
    ry1 = max(ry0 + 1, min(h - 1, y1 - 12))

    def col_frac(x):
        return float(near[ry0:ry1, x].mean())

    if x1 - x0 > 2 * band + 8:
        lo = max(0, x0 - band)
        hi = min(w - 1, x0 + band)
        for x in range(lo, hi + 1):
            if col_frac(x) >= 0.45:
                x0 = x
                break
        lo = max(0, x1 - band)
        hi = min(w - 1, x1 + band)
        for x in range(hi, lo - 1, -1):
            if col_frac(x) >= 0.45:
                x1 = x + 1
                break

    rx0 = min(w - 2, max(0, x0 + 12))
    rx1 = max(rx0 + 1, min(w - 1, x1 - 12))

    def row_frac(y):
        return float(near[y, rx0:rx1].mean())

    if y1 - y0 > 2 * band + 8:
        lo = max(0, y0 - band)
        hi = min(h - 1, y0 + band)
        for y in range(lo, hi + 1):
            if row_frac(y) >= 0.45:
                y0 = y
                break
        lo = max(0, y1 - band)
        hi = min(h - 1, y1 + band)
        for y in range(hi, lo - 1, -1):
            if row_frac(y) >= 0.45:
                y1 = y + 1
                break

    return [x0, y0, x1, y1]


def detect_page(gray: np.ndarray) -> Optional[List[int]]:
    """
    在一帧 640 宽灰度图上找"幻灯片页面"矩形（坐标基于该灰度图）。
    页面是画面中最大的一块接近背景亮度的矩形：全屏放映时黑边被排除，
    PowerPoint 编辑窗口里灰色画布/缩略图栏被排除。
    """
    h, w = gray.shape
    sub = gray[::3, ::3]
    bg = float(np.percentile(sub, 97))
    if bg < 40:
        return None
    bs = 8
    hs, ws = h // bs, w // bs
    if hs < 8 or ws < 8:
        return None
    small = gray[:hs * bs, :ws * bs].reshape(hs, bs, ws, bs).mean(axis=(1, 3))
    mask = small >= (bg - 9.0)
    frac = float(mask.mean())
    if frac < 0.03:
        return None
    if frac > 0.90:  # 整屏都是浅色（全屏 16:9 幻灯片 / 白底演示）
        box = [0, 0, w, h]
    else:
        bbox, area = largest_component_bbox(mask)
        if bbox is None or area < 0.02 * hs * ws:
            return None
        x0, y0, x1, y1 = bbox
        box = [x0 * bs, y0 * bs, min(w, x1 * bs), min(h, y1 * bs)]
    return refine_box(gray, bg, box)


def crop_signature(img: Image.Image) -> np.ndarray:
    """统一到 480x360 并轻微模糊，用于相似度比较（容忍缩放/抗锯齿差异）。"""
    from PIL import ImageFilter
    g = img.convert("L").resize((480, 360), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(g, dtype=np.uint8)


def diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """两张签名图的差异比例（|Δ|>25 的像素占比）。"""
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > 25).mean())


def aligned_diff(a: np.ndarray, b: np.ndarray, max_shift: int = 8, step: int = 2) -> float:
    """
    允许上下左右少量平移后再比差异：同一页的"全屏放映截图"和"编辑窗口截图"
    在裁剪时可能有几个像素的偏移，先对齐再比较，去重才不会漏。
    不同页面（哪怕只差一个字）对齐后差异仍然明显，所以不会误合并。
    """
    h, w = a.shape
    best = 1.0
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            ay0, by0 = max(0, dy), max(0, -dy)
            ay1, by1 = min(h, h + dy), min(h, h - dy)
            ax0, bx0 = max(0, dx), max(0, -dx)
            ax1, bx1 = min(w, w + dx), min(w, w - dx)
            if ay1 - ay0 < 50 or ax1 - ax0 < 50:
                continue
            d = float((np.abs(a[ay0:ay1, ax0:ax1].astype(np.int16) -
                              b[by0:by1, bx0:bx1].astype(np.int16)) > 25).mean())
            if d < best:
                best = d
    return best


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for bit in bits:
        v = (v << 1) | int(bit)
    return v


# ----------------------------------------------------------------------------
# 第一遍：逐帧统计 + 静止区间
# ----------------------------------------------------------------------------
def pass1_analyze(path: str, fps: float, aw: int, ah: int, start: float,
                  duration: Optional[float], cache: str) -> dict:
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    stats = {k: [] for k in ("bg", "page_frac", "edge", "ink")}
    diffs: List[float] = [0.0]
    prev_small: Optional[np.ndarray] = None
    t_begin = time.time()
    n = 0
    for rgb in iter_frames(path, fps, aw, ah, start, duration):
        gray = rgb.astype(np.float32) @ W_GRAY
        hs, ws = ah // 8, aw // 8
        small = gray[:hs * 8, :ws * 8].reshape(hs, 8, ws, 8).mean(axis=(1, 3))
        if prev_small is not None:
            diffs.append(float(np.abs(small - prev_small).mean()))
        prev_small = small

        bg = float(np.percentile(gray[::3, ::3], 97))
        stats["bg"].append(bg)
        stats["page_frac"].append(float((gray[:hs * 8, :ws * 8] >= bg - 9).mean()))
        gx = np.abs(np.diff(small, axis=1))
        gy = np.abs(np.diff(small, axis=0))
        stats["edge"].append(float(((gx > 12).mean() + (gy > 12).mean()) * 0.5))
        stats["ink"].append(float((small < bg - 70).mean()))

        n += 1
        if n % 300 == 0:
            print(f"  分析中… {n} 帧 / {time.time() - t_begin:.0f}s", flush=True)

    arr = {
        "n": np.array([n], dtype=np.int64),
        "fps": np.array([fps], dtype=np.float32),
        "start": np.array([start], dtype=np.float32),
        "diffs": np.asarray(diffs, dtype=np.float32),
    }
    for k, v in stats.items():
        arr[k] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(cache, **arr)
    print(f"  分析完成：{n} 帧，用时 {time.time() - t_begin:.0f}s")
    return arr


def build_runs(diffs: np.ndarray, n: int, fps: float, static_thr: float,
               min_sec: float) -> List[Tuple[int, int]]:
    """把"相邻帧差异都很小"的连续帧合并成区间 [s, e]（含端点）。"""
    runs = []
    s = 0
    for i in range(1, n):
        if diffs[i] > static_thr:
            runs.append((s, i - 1))
            s = i
    runs.append((s, n - 1))
    min_frames = max(1, int(round(min_sec * fps)))
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_frames]
    return runs


def run_score(arr: dict, s: int, e: int) -> dict:
    def med(key):
        return float(np.median(arr[key][s:e + 1]))
    return {
        "page_frac": med("page_frac"),
        "bg": med("bg"),
        "edge": med("edge"),
        "ink": med("ink"),
    }


def looks_like_slide(feat: dict, args) -> bool:
    if feat["page_frac"] < args.page_frac_min:
        return False
    if feat["bg"] < args.bg_min:
        return False
    if feat["edge"] < args.edge_min:
        return False
    if feat["ink"] < args.ink_min:
        return False
    if feat["ink"] > 0.55:          # 大面积深色，多半是黑屏/窗口边框/无信号画面
        return False
    return True


STD_ASPECTS = (4 / 3, 16 / 9, 16 / 10, 3 / 2)


def plausible_crop(st: dict, args) -> Tuple[bool, str]:
    """裁剪结果是否像"一张幻灯片"：比例接近常见幻灯片，且不是彩色照片/壁纸。"""
    if st["ink"] > 0.6:
        return False, "too_dark"
    if st["sat_hi"] > 0.5:
        return False, "too_colorful"
    asp_err = min(abs(st["aspect"] / a - 1.0) for a in STD_ASPECTS)
    if asp_err > 0.05:
        return False, "aspect"
    if st["edge"] < args.edge_min * 0.3 and st["ink"] < args.ink_min:
        return False, "blank"
    return True, ""


# ----------------------------------------------------------------------------
# 第二遍：抽代表帧、裁剪、去重
# ----------------------------------------------------------------------------
def extract_frame(path: str, t: float, out_png: str) -> bool:
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}",
           "-i", path, "-frames:v", "1", "-y", out_png]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


def trim_letterbox(img: Image.Image) -> Image.Image:
    """去掉四周纯黑边（全屏放映 4:3 幻灯片时两侧的黑条）。"""
    g = np.asarray(img.convert("L"))
    h, w = g.shape
    rows = np.where((g < 24).mean(axis=1) < 0.9)[0]
    cols = np.where((g < 24).mean(axis=0) < 0.9)[0]
    if len(rows) and len(cols):
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        if (y1 - y0) > h * 0.3 and (x1 - x0) > w * 0.3:
            return img.crop((x0, y0, x1, y1))
    return img


def detect_bars(gray: np.ndarray) -> Optional[List[int]]:
    """
    全屏放映的幻灯片两侧/上下通常是纯黑边框；找到这些黑边并返回内部矩形。
    没有黑边（或整屏全黑）时返回 None。
    """
    h, w = gray.shape
    dark = gray < 45.0
    left = 0
    while left < w and dark[:, left].mean() >= 0.9:
        left += 1
    right = 0
    while right < w and dark[:, w - 1 - right].mean() >= 0.9:
        right += 1
    top = 0
    while top < h and dark[top, :].mean() >= 0.9:
        top += 1
    bottom = 0
    while bottom < h and dark[h - 1 - bottom, :].mean() >= 0.9:
        bottom += 1
    if left + right + top + bottom == 0:
        return None
    x0, x1 = left, w - right
    y0, y1 = top, h - bottom
    if (x1 - x0) < 0.3 * w or (y1 - y0) < 0.3 * h:
        return None
    return [x0, y0, x1, y1]


def aspect_err(box: List[int], w: int, h: int) -> float:
    asp = (box[2] - box[0]) / float(box[3] - box[1])
    return min(abs(asp / a - 1.0) for a in STD_ASPECTS)


def snap_aspect(box: List[int], w: int, h: int, tol: float = 0.06) -> List[int]:
    """
    把页面框微调到标准幻灯片比例（4:3 / 16:9 / 16:10 / 3:2）。
    检测出来的框常常比真实页面小 1~2%（页面边缘有细线/阴影），
    按比例向外扩一点既能把整页补回来，也便于不同放映方式之间做去重比较。
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return box
    a = bw / bh
    r = min(STD_ASPECTS, key=lambda t: abs(a / t - 1.0))
    if abs(a / r - 1.0) > tol:
        return box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if a < r:  # 太“瘦”，补宽
        nw, nh = bh * r, bh
    else:      # 太“胖”，补高
        nw, nh = bw, bw / r
    nw = min(nw, w)
    nh = min(nh, h)
    if a < r:
        nh = nw / r
    else:
        nw = nh * r
    nx0 = min(max(0.0, cx - nw / 2.0), w - nw)
    ny0 = min(max(0.0, cy - nh / 2.0), h - nh)
    return [int(round(nx0)), int(round(ny0)),
            int(round(nx0 + nw)), int(round(ny0 + nh))]


def crop_slide(img: Image.Image, args) -> Tuple[Image.Image, str]:
    """
    定位并裁剪幻灯片页面：优先用全屏放映的黑边（最精确），否则用最大浅色矩形；
    再把框微调到标准幻灯片比例；都不像标准幻灯片时退化为去掉黑边的整屏。
    """
    W, H = img.size
    if args.no_crop:
        return trim_letterbox(img), "full"
    aw, ah = analysis_size(W, H, args.analysis_width)
    small = img.resize((aw, ah), Image.BILINEAR)
    gray = np.asarray(small.convert("L"), dtype=np.float32)

    bars = detect_bars(gray)
    page = detect_page(gray)
    chosen = None
    if bars is not None and aspect_err(bars, aw, ah) <= 0.06:
        chosen = (bars, "page")
    elif page is not None and aspect_err(page, aw, ah) <= 0.06:
        chosen = (page, "page")
    elif bars is not None:
        chosen = (bars, "page")
    elif page is not None:
        chosen = (page, "page")

    if chosen is not None:
        box, mode = chosen
        box = snap_aspect(box, aw, ah)
        x0, y0, x1, y1 = box
        if (x1 - x0) >= aw * 0.18 and (y1 - y0) >= ah * 0.18:
            sx, sy = W / float(aw), H / float(ah)
            cbox = (max(0, int(round(x0 * sx))), max(0, int(round(y0 * sy))),
                    min(W, int(round(x1 * sx))), min(H, int(round(y1 * sy))))
            if cbox[2] - cbox[0] > 320 and cbox[3] - cbox[1] > 240:
                return img.crop(cbox), mode
    return trim_letterbox(img), "full"


def image_stats(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat_hi = float((((mx - mn) > 0.5 * np.maximum(mx, 1)) & (mx > 60)).mean())
    g = np.asarray(img.convert("L"), dtype=np.float32)
    ink = float((g < 110).mean())
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    edge = float(((gx > 18).mean() + (gy > 18).mean()) * 0.5)
    return {"ink": ink, "edge": edge, "sat_hi": sat_hi,
            "aspect": img.size[0] / float(img.size[1]),
            "area": img.size[0] * img.size[1]}


def dedupe(cands: List[dict], args, log: List[dict]) -> List[dict]:
    kept: List[dict] = []
    for c in cands:
        best = None
        best_ratio = 1.0
        for k in kept:
            if abs(c["aspect"] - k["aspect"]) > 0.10:
                continue
            if diff_ratio(c["sig"], k["sig"]) > 0.25:   # 快速排除明显不同的
                continue
            r = aligned_diff(c["sig"], k["sig"])
            if r <= args.dedupe_thr and r < best_ratio:
                best = k
                best_ratio = r
        if best is not None:
            # 同一页：默认保留第一次出现的顺序位置；
            # 若后来这版内容更完整 / 分辨率更高，就用它替换画面。
            better = (c["ink"] > best["ink"] * 1.05) or (c["area"] > best["area"] * 1.15)
            log.append({"time": c["time"], "dup_of": best["time"], "ratio": best_ratio,
                        "replaced": int(better)})
            if better:
                best["time"] = c["time"]
                best["image"] = c["image"]
                best["sig"] = c["sig"]
                best["dhash"] = c["dhash"]
                best.update({k: c[k] for k in ("ink", "edge", "aspect", "area")})
        else:
            kept.append(c)
            log.append({"time": c["time"], "dup_of": None, "ratio": 0.0, "replaced": 0})
    return kept


def contact_sheet(items: List[dict], out_png: str, cols: int = 6, tw: int = 240) -> None:
    if not items:
        return
    th = int(tw * 0.75)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 22) + 6), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x = 6 + c * (tw + 6)
        y = 6 + r * (th + 22)
        thumb = it["image"].convert("RGB").resize((tw, th), Image.BILINEAR)
        sheet.paste(thumb, (x, y))
        label = "%02d  %s" % (i + 1, fmt_time(it["time"]))
        d.text((x + 2, y + th + 3), label, fill=(230, 230, 230))
    sheet.save(out_png)


def fmt_time(t: float) -> str:
    t = int(round(t))
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


# ----------------------------------------------------------------------------
# 生成 pptx
# ----------------------------------------------------------------------------
def build_pptx(items: List[dict], out_path: str, source_name: str) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    aspects = [it["aspect"] for it in items]
    med_aspect = float(np.median(aspects)) if aspects else 4 / 3
    if abs(med_aspect - 16 / 9) < abs(med_aspect - 4 / 3):
        slide_w, slide_h = 13.333, 7.5
    else:
        slide_w, slide_h = 10.0, 7.5
    slide_aspect = slide_w / slide_h

    prs = Presentation()
    prs.slide_width = Inches(slide_w)
    prs.slide_height = Inches(slide_h)
    blank = prs.slide_layouts[6]

    for idx, it in enumerate(items, 1):
        slide = prs.slides.add_slide(blank)
        img_w, img_h = it["image"].size
        img_aspect = img_w / float(img_h)
        if abs(img_aspect - slide_aspect) <= 0.02:
            slide.shapes.add_picture(it["path"], 0, 0,
                                     width=prs.slide_width, height=prs.slide_height)
        else:
            if img_aspect > slide_aspect:
                w = prs.slide_width
                h = int(w / img_aspect)
                x, y = 0, int((prs.slide_height - h) / 2)
            else:
                h = prs.slide_height
                w = int(h * img_aspect)
                x, y = int((prs.slide_width - w) / 2), 0
            slide.shapes.add_picture(it["path"], x, y, width=w, height=h)
        slide.notes_slide.notes_text_frame.text = (
            "来源：%s\n时间：%s\n截图尺寸：%dx%d" %
            (source_name, fmt_time(it["time"]), img_w, img_h))
    prs.save(out_path)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="把课堂录屏里真正属于 PPT 的画面整理成一份 PPT（截图版）。")
    ap.add_argument("input", help="输入视频文件")
    ap.add_argument("-o", "--output", help="输出 pptx 路径（默认 <视频名>_整理.pptx）")
    ap.add_argument("--workdir", default="_video2ppt", help="中间文件目录")
    ap.add_argument("--fps", type=float, default=1.0, help="抽样帧率（默认 1）")
    ap.add_argument("--static-thr", type=float, default=2.0,
                    help="静止判定阈值：相邻抽样帧平均灰度差（0-255）")
    ap.add_argument("--min-sec", type=float, default=3.0, help="候选页面最短静止时长（秒）")
    ap.add_argument("--dedupe-thr", type=float, default=0.015,
                    help="去重阈值：两页（对齐后）差异像素比例上限（越小越严格）")
    ap.add_argument("--page-frac-min", type=float, default=0.12,
                    help="浅色页面至少占画面比例（过滤桌面等）")
    ap.add_argument("--bg-min", type=float, default=90.0, help="页面背景最低亮度")
    ap.add_argument("--edge-min", type=float, default=0.004, help="最低文字/边缘密度")
    ap.add_argument("--ink-min", type=float, default=0.0015, help="最低深色内容比例")
    ap.add_argument("--analysis-width", type=int, default=640, help="分析/定位用缩放宽度")
    ap.add_argument("--no-crop", action="store_true", help="不裁剪幻灯片页面，直接整屏截图")
    ap.add_argument("--png", action="store_true",
                    help="截图保存为 PNG（默认用 JPEG，体积约为 PNG 的 1/3）")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="JPEG 质量（默认 90）")
    ap.add_argument("--keep-all", action="store_true", help="保存全部候选截图与拼图")
    ap.add_argument("--debug", action="store_true", help="输出候选与去重明细")
    ap.add_argument("--start", type=float, default=0.0, help="只处理从该秒开始的部分（调试）")
    ap.add_argument("--duration", type=float, default=None, help="只处理该时长（调试）")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        print("找不到输入文件:", args.input)
        return 2
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or (stem + "_整理.pptx")
    work = args.workdir
    slide_dir = os.path.join(work, "slides")
    cand_dir = os.path.join(work, "candidates")
    os.makedirs(slide_dir, exist_ok=True)
    # 清掉上一次运行留下的截图，避免新旧文件混在一起
    for name in os.listdir(slide_dir):
        if name.lower().endswith((".png", ".jpg")):
            os.remove(os.path.join(slide_dir, name))
    if os.path.isdir(cand_dir):
        for name in os.listdir(cand_dir):
            if name.lower().endswith((".png", ".jpg")):
                os.remove(os.path.join(cand_dir, name))
    if args.keep_all:
        os.makedirs(cand_dir, exist_ok=True)

    W, H, dur = probe_video(args.input)
    aw, ah = analysis_size(W, H, args.analysis_width)
    print(f"视频：{W}x{H}，时长 {fmt_time(dur)}")
    print(f"分析：{args.fps:g} 帧/秒，缩小到 {aw}x{ah}")

    tag = f"_{int(args.start)}-{int(args.start + (args.duration or 0))}" if (args.start or args.duration) else ""
    cache = os.path.join(work, f"analysis_{args.fps:g}fps_{aw}x{ah}{tag}.npz")
    arr = pass1_analyze(args.input, args.fps, aw, ah, args.start, args.duration, cache)
    n = int(arr["n"][0])
    if n == 0:
        print("没有读到帧")
        return 3
    fps = float(arr["fps"][0])
    t_start = float(arr["start"][0])

    runs = build_runs(arr["diffs"], n, fps, args.static_thr, args.min_sec)
    print(f"静止区间：{len(runs)} 个")

    # 过滤 + 每个区间取末尾一帧（这一页最后、最完整的画面）
    cands = []
    if args.debug:
        dbg = open(os.path.join(work, "runs_debug.csv"), "w", newline="", encoding="utf-8-sig")
        dw = csv.writer(dbg)
        dw.writerow(["start", "end", "sec", "page_frac", "bg", "edge", "ink", "keep"])
    for (s, e) in runs:
        feat = run_score(arr, s, e)
        ok = looks_like_slide(feat, args)
        if args.debug:
            dw.writerow([fmt_time(t_start + s / fps), fmt_time(t_start + e / fps),
                         round((e - s + 1) / fps, 1),
                         round(feat["page_frac"], 3), round(feat["bg"], 1),
                         round(feat["edge"], 4), round(feat["ink"], 4), int(ok)])
        if not ok:
            continue
        rep = s + int(round(0.97 * (e - s)))
        rep = min(e, max(s, rep))
        cands.append({"frame": rep, "time": t_start + rep / fps,
                      "run": (s, e), "feat": feat})
    if args.debug:
        dbg.close()
    print(f"符合 PPT 画面特征的候选页面：{len(cands)} 个")

    # 抽代表帧、裁剪
    extracted = []
    for i, c in enumerate(cands):
        png = os.path.join(work, "frames", "f_%05d.png" % i)
        os.makedirs(os.path.dirname(png), exist_ok=True)
        if not extract_frame(args.input, c["time"], png):
            print("  抽帧失败 @", fmt_time(c["time"]))
            continue
        try:
            img = Image.open(png).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print("  读取失败", png, exc)
            continue
        crop, mode = crop_slide(img, args)
        st = image_stats(crop)
        ok, why = plausible_crop(st, args)
        if not ok:
            if args.debug:
                print("  丢弃 %s（%s）" % (fmt_time(c["time"]), why))
            continue
        c2 = dict(c)
        c2.update(st)
        c2["image"] = crop
        c2["mode"] = mode
        c2["sig"] = crop_signature(crop)
        c2["dhash"] = dhash(crop)
        c2["index"] = i
        extracted.append(c2)
        if args.keep_all:
            crop.save(os.path.join(cand_dir, "c_%05d_%s.png" % (i, fmt_time(c["time"]).replace(":", "-"))))
        if (i + 1) % 20 == 0:
            print(f"  抽帧裁剪… {i + 1}/{len(cands)}", flush=True)

    if args.keep_all:
        contact_sheet(extracted, os.path.join(work, "candidates_sheet.png"))
        print("候选拼图：", os.path.join(work, "candidates_sheet.png"))

    log: List[dict] = []
    kept = dedupe(extracted, args, log)
    print(f"去重后保留：{len(kept)} 页")

    # 落盘最终截图
    final = []
    ext = ".png" if args.png else ".jpg"
    for i, it in enumerate(kept, 1):
        path = os.path.join(slide_dir, "slide_%03d_%s%s" %
                            (i, fmt_time(it["time"]).replace(":", "-"), ext))
        if args.png:
            it["image"].save(path)
        else:
            it["image"].save(path, quality=int(args.jpeg_quality), optimize=True)
        it["path"] = path
        final.append(it)
    if args.keep_all:
        contact_sheet(final, os.path.join(work, "slides_sheet.png"))

    # 索引表
    with open(os.path.join(work, "slides_index.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["序号", "画面时间", "裁剪方式", "宽", "高", "深色占比", "边缘密度"])
        for i, it in enumerate(final, 1):
            w.writerow([i, fmt_time(it["time"]), it["mode"],
                        it["image"].size[0], it["image"].size[1],
                        round(it["ink"], 4), round(it["edge"], 4)])
    with open(os.path.join(work, "dedupe_log.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["时间", "与哪一页重复", "差异比例", "是否替换为更完整版本"])
        for row in log:
            w.writerow([fmt_time(row["time"]),
                        fmt_time(row["dup_of"]) if row["dup_of"] is not None else "",
                        round(row["ratio"], 4) if row["dup_of"] is not None else "",
                        row["replaced"]])

    if not final:
        print("没有识别到任何 PPT 页面，请调整阈值后重试")
        return 4

    build_pptx(final, out_path, os.path.basename(args.input))
    # 中间抽帧文件用完就删，节省磁盘（最终截图都已保存在 slides 目录）
    shutil.rmtree(os.path.join(work, "frames"), ignore_errors=True)
    print("完成：", out_path)
    print("截图目录：", slide_dir)
    print("索引：", os.path.join(work, "slides_index.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
