#!/usr/bin/env python3
"""
Video Blur Studio - SAM2ベースの人物セグメンテーション & 背景ぼかしツール
ロトブラシ3.0同等以上の精度を目指した本格的なビデオ編集アプリケーション

Requirements:
  - Python 3.10+
  - PyQt6
  - torch (CUDA対応)
  - sam2 (Meta SAM 2.1)
  - opencv-python
  - numpy
"""

import sys
import os

# ── PyQt6 imports ──────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QStatusBar, QToolBar,
    QDockWidget, QListWidget, QListWidgetItem, QSpinBox, QComboBox,
    QProgressBar, QGroupBox, QCheckBox, QSplitter, QMessageBox,
    QScrollArea, QFrame, QToolButton, QMenu, QSizePolicy, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QStyle, QListView
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF, QSize, QRect
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, QAction,
    QIcon, QKeySequence, QCursor, QWheelEvent, QMouseEvent,
    QPainterPath, QFont, QFontDatabase
)

import cv2
import numpy as np
import json
import time
import gc
import shutil
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

_LIBC = None
_LIBC_CHECKED = False


def _malloc_trim():
    """glibc が抱えたままの解放済みヒープを OS へ返す

    Python や numpy が free したメモリでも、glibc は次の確保に備えて
    アリーナに保持し続けるため RSS が下がらない。実測では伝播 1 回ぶんで
    1GB 以上がこの状態で残っていた。処理の区切りで trim を呼んで返却する。
    glibc 以外の環境では何もしない。
    """
    global _LIBC, _LIBC_CHECKED
    if not _LIBC_CHECKED:
        _LIBC_CHECKED = True
        try:
            import ctypes
            lib = ctypes.CDLL("libc.so.6")
            lib.malloc_trim.argtypes = [ctypes.c_size_t]
            lib.malloc_trim.restype = ctypes.c_int
            _LIBC = lib
        except Exception as e:
            logger.info(f"malloc_trim は利用できません: {e}")
            _LIBC = None
    if _LIBC is None:
        return False
    try:
        return bool(_LIBC.malloc_trim(0))
    except Exception:
        return False


def _available_mb():
    """OS が今すぐ渡せるメモリ量 (MB)。取得できなければ 0。"""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _rss_mb():
    """このプロセスの実メモリ使用量 (MB)。取得できなければ 0。"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


# ── ロギング設定 ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# データクラスと列挙型
# ══════════════════════════════════════════════════════════════════════════════

class ToolMode(Enum):
    SELECT_PERSON = auto()      # 人物をクリックして選択
    ADD_MASK = auto()           # マスクを手動追加（ブラシ）
    REMOVE_MASK = auto()        # マスクを手動削除（消しゴム）
    PAN = auto()                # パン
    ZOOM = auto()               # ズーム
    NEGATIVE_POINT = auto()     # ネガティブポイント（除外領域）


@dataclass
class PersonTrack:
    """トラッキング対象の人物情報"""
    track_id: int
    name: str
    color: QColor
    points: dict = field(default_factory=dict)      # {frame_idx: [(x, y, label)]}
    masks: dict = field(default_factory=dict)        # {frame_idx: np.ndarray}
    is_visible: bool = True
    is_locked: bool = False


@dataclass
class ProjectState:
    """プロジェクト全体の状態"""
    video_path: str = ""
    output_path: str = ""
    total_frames: int = 0
    fps: float = 30.0
    width: int = 0                # 作業解像度（プロキシ）。編集・マスクは全てこの座標系
    height: int = 0
    source_width: int = 0         # 元動画の解像度。エクスポートはこの解像度で書き出す
    source_height: int = 0
    proxy_scale: float = 1.0      # width / source_width（1.0 = プロキシ未使用）
    current_frame: int = 0
    blur_strength: int = 25
    blur_type: str = "gaussian"   # gaussian, box, motion
    edge_feather: int = 5
    person_tracks: list = field(default_factory=list)
    manual_edits: dict = field(default_factory=dict)  # {frame_idx: np.ndarray}
    auto_save_enabled: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# SAM2 バックエンド (セグメンテーションエンジン)
# ══════════════════════════════════════════════════════════════════════════════

# 保存用マスクの縮小倍率。1.0 で縮小なし、0.5 で 1/2 解像度 (メモリ 1/4)。
# 読み取り側 (_refresh_display / ExportThread 等) は shape 不一致時に
# cv2.resize(NEAREST) で元解像度へ復元するため整合が取れる。
MASK_STORE_SCALE = 0.5

# ── プロキシ解像度 ──
# UHD/4K を読み込んだときは、この枠に収まるまで縮小した「プロキシ」で編集する。
# マスク・手動編集・ポイントは全てプロキシ座標系で保持し、
# エクスポート時に ExportThread が元解像度へ戻して書き出す。
PROXY_MAX_W = 1920
PROXY_MAX_H = 1080


def calc_proxy_size(width, height):
    """(width, height) を FHD 枠に収める作業解像度を返す。縮小不要なら原寸のまま。

    戻り値: (proxy_w, proxy_h, scale)。scale は 1.0 で縮小なし。
    """
    if width <= 0 or height <= 0:
        return width, height, 1.0
    scale = min(PROXY_MAX_W / width, PROXY_MAX_H / height, 1.0)
    if scale >= 1.0:
        return width, height, 1.0
    # 偶数に丸める（H.264/mp4v エンコーダが奇数サイズを嫌うため）
    proxy_w = max(2, int(round(width * scale)) // 2 * 2)
    proxy_h = max(2, int(round(height * scale)) // 2 * 2)
    return proxy_w, proxy_h, proxy_w / width


def _to_size(mask, width, height):
    """マスクを (width, height) の uint8 (0/1) に揃える。

    マスクは MASK_STORE_SCALE で縮小して保持しているため、
    読み出し側は必ずこれを通してから合成する。
    """
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    m = mask.astype(np.uint8, copy=False)
    if m.shape[:2] != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m


def build_mask_float(state, all_masks, frame_idx, work_w, work_h):
    """指定フレームの人物マスクを 0.0-1.0 の float32 で組み立てる。

    マスクも手動編集も MASK_STORE_SCALE で縮小して保持しているため、
    ここで作業解像度へ揃えてからフェザリングをかける。
    """
    combined = np.zeros((work_h, work_w), dtype=np.uint8)
    for track in state.person_tracks:
        m = all_masks.get(track.track_id, {}).get(frame_idx)
        if m is not None:
            combined = np.maximum(combined, _to_size(m, work_w, work_h))

    edit = state.manual_edits.get(frame_idx)
    if edit is not None:
        combined = np.maximum(combined, _to_size(edit, work_w, work_h))

    mask_float = combined.astype(np.float32)
    if state.edge_feather > 0:
        fk = state.edge_feather * 2 + 1
        mask_float = cv2.GaussianBlur(mask_float, (fk, fk), 0)

    peak = float(mask_float.max())
    if peak > 0:
        mask_float /= peak
    return mask_float


class GpuBlender:
    """ぼかしと合成を GPU (PyTorch) で行う

    エクスポートとプレビューの重い部分――大きなカーネルのガウシアンぼかしと
    フレーム全体のアルファ合成――は OpenCV/numpy では CPU で回るため、
    ここだけ GPU へ逃がす。OpenCV が CUDA 無効ビルドでも、
    torch は CUDA 対応で入っているのでそれを使う。

    CUDA が無い / 途中で失敗した場合は process() が None を返し、
    呼び出し側は従来どおり CPU で処理する。環境変数 VBS_GPU_BLUR=0 で無効化できる。
    """

    def __init__(self):
        self.torch = None
        self.F = None
        self.device = None
        self.enabled = False
        self._kernels = {}
        self._warned = False

        if os.environ.get("VBS_GPU_BLUR", "1") == "0":
            logger.info("GPU ぼかしは VBS_GPU_BLUR=0 により無効です")
            return
        try:
            import torch
            import torch.nn.functional as F
            if not torch.cuda.is_available():
                logger.info("CUDA が使えないため、ぼかしは CPU で処理します")
                return
            self.torch = torch
            self.F = F
            self.device = torch.device("cuda")
            self.enabled = True
            logger.info("ぼかし・合成を GPU で処理します")
        except Exception as e:
            logger.info(f"GPU ぼかしを初期化できませんでした（CPU で処理します）: {e}")

    def warmup(self):
        """CUDA コンテキストとカーネルを先に作っておく

        最初の 1 回だけコンテキスト生成に数百 ms かかるため、
        短いエクスポートだとその分で GPU 化の利得が消えてしまう。
        起動時に小さな処理を 1 度流して払っておく。
        """
        if not self.enabled:
            return
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.float32)
            self.process(dummy, mask, "gaussian", 3)
        except Exception:
            pass

    def close(self):
        """保持しているカーネルを解放する"""
        self._kernels.clear()

    # ── 畳み込みカーネル ──
    def _kernel(self, blur_type, k):
        """(3,1,1,k) 形状の分離可能カーネルを作って使い回す"""
        key = (blur_type, k)
        cached = self._kernels.get(key)
        if cached is not None:
            return cached
        torch = self.torch
        if blur_type == "gaussian":
            # sigma は OpenCV が ksize から決める既定値と同じ式にする
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
            x = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2
            g = torch.exp(-(x * x) / (2 * sigma * sigma))
            g = g / g.sum()
        else:
            g = torch.full((k,), 1.0 / k, dtype=torch.float32, device=self.device)
        weight = g.view(1, 1, 1, k).repeat(3, 1, 1, 1)
        self._kernels[key] = weight
        return weight

    def _conv_h(self, x, weight, k):
        x = self.F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect")
        return self.F.conv2d(x, weight, groups=3)

    def _conv_v(self, x, weight, k):
        x = self.F.pad(x, (0, 0, k // 2, k // 2), mode="reflect")
        return self.F.conv2d(x, weight.transpose(2, 3), groups=3)

    def _blur(self, x, blur_type, k):
        if blur_type == "motion":
            # 横方向だけの一様ぼかし（CPU 版の kernel[k//2, :] と同じ）
            return self._conv_h(x, self._kernel("box", k), k)
        kind = "gaussian" if blur_type not in ("box", "motion") else blur_type
        weight = self._kernel(kind, k)
        return self._conv_v(self._conv_h(x, weight, k), weight, k)

    def process(self, frame, mask_float, blur_type, strength):
        """ぼかして合成した uint8 フレームを返す。処理できなければ None。

        mask_float が frame と別サイズなら GPU 上で拡大する。
        作業解像度の小さいマスクを渡せば、CPU 側の拡大も転送量も減る。
        """
        if not self.enabled:
            return None

        k = max(1, int(strength)) * 2 + 1
        h, w = frame.shape[:2]
        # reflect パディングは入力より大きい幅を取れない
        if k // 2 >= min(h, w):
            return None

        torch = self.torch
        try:
            with torch.inference_mode():
                src = torch.from_numpy(np.ascontiguousarray(frame)).to(
                    self.device, non_blocking=True)
                x = src.permute(2, 0, 1).unsqueeze(0).float()

                blurred = self._blur(x, blur_type, k)

                m = torch.from_numpy(np.ascontiguousarray(mask_float)).to(
                    self.device, non_blocking=True)
                m = m.unsqueeze(0).unsqueeze(0)
                if m.shape[-2:] != (h, w):
                    m = self.F.interpolate(
                        m, size=(h, w), mode="bilinear", align_corners=False)

                # result = blurred + (frame - blurred) * mask
                out = torch.addcmul(blurred, x - blurred, m)
                # CPU 版の astype(uint8) と同じく切り捨てで揃える
                out = out.clamp_(0, 255).to(torch.uint8)
                return out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        except Exception as e:
            if not self._warned:
                logger.warning(
                    f"GPU でのぼかしに失敗したため CPU に切り替えます: {e}")
                self._warned = True
            self.enabled = False
            self._kernels.clear()
            return None


def blend_blur(frame, mask_float, blur_type, strength, buf=None, blender=None):
    """背景をぼかし、人物マスクで元フレームと合成する。

    mask_float は 0.0-1.0。frame と大きさが違えば frame に合わせて拡大する。
    blender に GpuBlender を渡すと GPU で処理し、使えなければ CPU に落ちる。

    CPU 経路は result = blurred + (frame - blurred) * mask を in-place で計算し、
    3ch に広げた float コピーを作らずに済ませる（4K で 1 枚あたり約 100MB の節約）。
    buf に float32 の作業バッファを渡すと、フレームごとの再確保を避けられる。
    """
    if blender is not None:
        out = blender.process(frame, mask_float, blur_type, strength)
        if out is not None:
            return out

    # ── CPU 経路 ──
    if mask_float.shape[:2] != frame.shape[:2]:
        mask_float = cv2.resize(
            mask_float, (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    k = max(1, int(strength)) * 2 + 1
    if blur_type == "box":
        blurred = cv2.blur(frame, (k, k))
    elif blur_type == "motion":
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        blurred = cv2.filter2D(frame, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

    if buf is None or buf.shape != frame.shape or buf.dtype != np.float32:
        buf = np.empty(frame.shape, dtype=np.float32)

    np.subtract(frame, blurred, out=buf, dtype=np.float32)
    np.multiply(buf, mask_float[:, :, np.newaxis], out=buf)
    np.add(buf, blurred, out=buf)
    return buf.astype(np.uint8)


def _downscale_mask_for_storage(mask, scale=MASK_STORE_SCALE):
    if mask is None or scale >= 1.0:
        return mask
    h, w = mask.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    resized = cv2.resize(
        mask.astype(np.uint8), (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# 変更履歴（Undo / Redo）
# ══════════════════════════════════════════════════════════════════════════════

def _pack_mask(arr):
    """マスクをビットパックして履歴用に保持する（bool 比 1/8、uint8 比 1/8）。"""
    if arr is None:
        return None
    a = np.asarray(arr)
    return (a.shape, a.dtype.str, np.packbits(a.astype(bool).ravel()))


def _unpack_mask(blob):
    """_pack_mask で畳んだマスクを元の shape / dtype へ戻す。"""
    if blob is None:
        return None
    shape, dtype_str, packed = blob
    count = int(np.prod(shape))
    flat = np.unpackbits(packed, count=count)
    return flat.reshape(shape).astype(np.dtype(dtype_str))


class UndoStack:
    """スロット単位の差分スナップショットを積む変更履歴。

    1 操作 = 1 スナップショットで、触れたスロット（人物ごと・フレームごと）の
    変更前の値だけを保持する。マスクはビットパックして格納するため、
    30 件保持してもメモリは実測で数十 MB に収まる。
    """

    MAX_ENTRIES = 30                       # 遡れる操作数
    MAX_BYTES = 192 * 1024 * 1024          # 履歴全体のメモリ上限

    def __init__(self, max_entries=MAX_ENTRIES):
        self.max_entries = max_entries
        self._undo = []
        self._redo = []

    # ── スタック操作 ──
    def push(self, entry):
        self._undo.append(entry)
        self._redo.clear()
        self._trim()

    def push_redo(self, entry):
        self._redo.append(entry)
        while len(self._redo) > self.max_entries:
            self._redo.pop(0)

    def pop_undo(self):
        return self._undo.pop() if self._undo else None

    def pop_redo(self):
        return self._redo.pop() if self._redo else None

    def push_undo_only(self, entry):
        """Redo 実行時に、Redo 前の状態を Undo 側へ戻す（_redo をクリアしない）。"""
        self._undo.append(entry)
        self._trim()

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def depth(self):
        return len(self._undo)

    def undo_label(self):
        return self._undo[-1]["label"] if self._undo else ""

    def redo_label(self):
        return self._redo[-1]["label"] if self._redo else ""

    # ── メモリ管理 ──
    def nbytes(self):
        return sum(e.get("nbytes", 0) for e in self._undo) + \
               sum(e.get("nbytes", 0) for e in self._redo)

    def _trim(self):
        while len(self._undo) > self.max_entries:
            self._undo.pop(0)
        # 件数が上限内でも、巨大な伝播スナップショットで膨らんだ場合は古い順に捨てる
        while len(self._undo) > 1 and self.nbytes() > self.MAX_BYTES:
            self._undo.pop(0)


class BoundedFrameStore:
    """SAM2 の inference_state["images"] を差し替える、上限つきフレーム置き場

    SAM2 標準の AsyncVideoFrameLoader は全フレームを 1024x1024x3 の
    float32 として保持し続ける。実測で 1 フレームあたり約 24-31MB、
    1920x1080・900 フレームで 22-32GB に達し、動画を続けて扱うと
    RAM を使い果たす（実際に CUDA error: unknown error で落ちた）。

    ここでは必要になった時点で JPEG を読み、直近 max_frames 枚だけ
    LRU で保持する。伝播はフレーム順に進むので数十枚でほぼヒットし、
    メモリはフレーム数に依存しない一定量で収まる。

    SAM2 側が要求するのは __getitem__ / __len__ と
    video_height / video_width だけなので、その形だけ合わせている。
    """

    DEFAULT_MAX_FRAMES = 64

    def __init__(self, img_paths, image_size, img_mean, img_std,
                 video_height, video_width, max_frames=None):
        self.img_paths = list(img_paths)
        self.image_size = image_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.video_height = video_height
        self.video_width = video_width
        self.max_frames = max_frames or self.DEFAULT_MAX_FRAMES
        self.exception = None
        self.thread = None          # 非同期ローダは使わないが属性は揃えておく
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._loader = None         # _load_img_as_tensor をキャッシュ

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self._loader is None:
            from sam2.utils.misc import _load_img_as_tensor
            self._loader = _load_img_as_tensor

        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached

        img, height, width = self._loader(
            self.img_paths[index], self.image_size)
        img -= self.img_mean
        img /= self.img_std
        self.video_height = height
        self.video_width = width

        with self._lock:
            self._cache[index] = img
            while len(self._cache) > self.max_frames:
                self._cache.popitem(last=False)
        return img

    def clear(self):
        with self._lock:
            self._cache.clear()

    def nbytes(self):
        with self._lock:
            return sum(t.numel() * t.element_size() for t in self._cache.values())


class SAM2Engine:
    """SAM2モデルのラッパー"""

    def __init__(self):
        self.model = None
        self.predictor = None
        self.image_predictor = None
        self.inference_state = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.model_loaded = False
        self.model_cfg = None
        self.checkpoint = None
        # inference_state が動画の一部だけを保持している場合の先頭フレーム番号。
        # アプリ側は常にグローバルなフレーム番号で扱い、
        # predictor へ渡す直前にここで区間内の番号へ変換する。
        self.frame_offset = 0
        self.frame_count = 0
        # inference_state がメモリに保持するフレーム数の上限
        self.frame_cache_limit = BoundedFrameStore.DEFAULT_MAX_FRAMES
        # CUDA が回復不能になったかどうか（一度立つと GPU は使わない）
        self.cuda_broken = False

    @staticmethod
    def _check_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _find_config_yaml(model_size):
        """
        sam2 パッケージ内の yaml ファイルを直接探して絶対パスを返す。
        Hydra の検索パスに一切依存しない。
        """
        base_names = {
            "tiny":      "sam2.1_hiera_t.yaml",
            "small":     "sam2.1_hiera_s.yaml",
            "base_plus": "sam2.1_hiera_b+.yaml",
            "large":     "sam2.1_hiera_l.yaml",
        }
        target = base_names.get(model_size, base_names["large"])

        try:
            import sam2 as sam2_pkg
            sam2_dir = os.path.dirname(sam2_pkg.__file__)
            logger.info(f"SAM2 パッケージ場所: {sam2_dir}")

            for root, dirs, files in os.walk(sam2_dir):
                if target in files:
                    found = os.path.join(root, target)
                    logger.info(f"Config yaml 発見: {found}")
                    return found
        except ImportError:
            pass

        return None

    @staticmethod
    def _build_model_directly(yaml_path, checkpoint, device, for_video=False):
        """
        Hydra を完全にバイパスして SAM2 モデルを構築する。
        yaml を直接読み込み、OmegaConf → Python dict → SAM2 クラスを手動初期化。
        """
        import torch
        import yaml as pyyaml
        from omegaconf import OmegaConf

        logger.info(f"Config を直接読み込み: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw = pyyaml.safe_load(f)

        cfg = OmegaConf.create(raw)

        # SAM2 の内部モデルクラスをインスタンス化
        from sam2.modeling.sam2_base import SAM2Base
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        import sam2.modeling.sam2_base  # noqa - ensure registered

        # hydra.utils.instantiate の代わりに手動で構築
        from hydra.utils import instantiate

        # Hydra の instantiate は GlobalHydra 不要（OmegaConf dict を渡すだけ）
        # ただし _target_ キーが必要
        model = instantiate(cfg.model, _recursive_=True)

        # チェックポイントをロード
        with open(checkpoint, "rb") as f:
            state = torch.load(f, map_location=device, weights_only=True)

        model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()

        if for_video:
            predictor = SAM2VideoPredictor(
                fill_hole_area=cfg.get("fill_hole_area", 8),
                non_overlap_masks=cfg.get("non_overlap_masks", True),
                sam_model=model,
            )
            return predictor
        else:
            return model

    def load_model(self, model_size="large", checkpoint_path=None):
        """
        SAM2モデルをロード
        model_size: tiny, small, base_plus, large

        ロード戦略:
          1. まず通常の build_sam2 (Hydra経由) を試行
          2. 失敗したら Hydra をバイパスして yaml を直接読み込み
        """
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor, build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            checkpoints = {
                "tiny": "sam2.1_hiera_tiny.pt",
                "small": "sam2.1_hiera_small.pt",
                "base_plus": "sam2.1_hiera_base_plus.pt",
                "large": "sam2.1_hiera_large.pt",
            }

            # ── チェックポイント探索 ──
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.checkpoint = checkpoint_path
            else:
                ckpt_name = checkpoints.get(model_size, checkpoints["large"])
                script_dir = os.path.dirname(os.path.abspath(__file__))
                search_paths = [
                    os.path.join(script_dir, "checkpoints", ckpt_name),
                    os.path.join(".", "checkpoints", ckpt_name),
                    os.path.join("..", "checkpoints", ckpt_name),
                    os.path.expanduser(os.path.join("~", ".cache", "sam2", ckpt_name)),
                ]

                self.checkpoint = None
                for p in search_paths:
                    if os.path.exists(p):
                        self.checkpoint = os.path.abspath(p)
                        break

                if not self.checkpoint:
                    logger.error(f"チェックポイントが見つかりません: {ckpt_name}")
                    logger.error(f"探索パス: {search_paths}")
                    return False

            logger.info(f"チェックポイント: {self.checkpoint}")
            logger.info(f"SAM2 モデルをロード中: {model_size} ({self.device})")

            # ── GPU最適化 ──
            if self.device == "cuda":
                torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            # ── yaml ファイルの絶対パスを取得 ──
            yaml_path = self._find_config_yaml(model_size)
            if yaml_path is None:
                logger.error("Config yaml が見つかりません")
                return False

            # ── 戦略1: Hydra 経由の正規ロードを試行 ──
            loaded = False
            hydra_configs_to_try = [
                f"configs/sam2.1/{os.path.basename(yaml_path)}",
                os.path.basename(yaml_path),
                f"sam2.1/{os.path.basename(yaml_path)}",
            ]

            for cfg_str in hydra_configs_to_try:
                try:
                    logger.info(f"[戦略1] Hydra 経由で試行: {cfg_str}")
                    # vos_optimized=True は内部で CUDAグラフ + torch.compile を
                    # 強制的に適用するが、環境によって AssertionError が出るため無効化。
                    # 代わりに image_encoder のみ手動で torch.compile を適用する。
                    self.predictor = build_sam2_video_predictor(
                        cfg_str, self.checkpoint, device=self.device,
                    )
                    logger.info("通常モードで構築（torch.compile は後で手動適用）")

                    sam2_model = build_sam2(
                        cfg_str, self.checkpoint, device=self.device
                    )
                    self.image_predictor = SAM2ImagePredictor(sam2_model)
                    self.model_cfg = cfg_str
                    loaded = True
                    logger.info(f"[戦略1] 成功: {cfg_str}")
                    break
                except Exception as e:
                    logger.warning(f"[戦略1] '{cfg_str}' 失敗: {e}")
                    continue

            # ── 戦略2: Hydra バイパス (yaml 直接読み込み) ──
            if not loaded:
                logger.info("[戦略2] Hydra をバイパスして yaml を直接読み込み")
                try:
                    self.predictor = self._build_model_directly(
                        yaml_path, self.checkpoint, self.device, for_video=True
                    )
                    model_for_image = self._build_model_directly(
                        yaml_path, self.checkpoint, self.device, for_video=False
                    )
                    self.image_predictor = SAM2ImagePredictor(model_for_image)
                    self.model_cfg = yaml_path
                    loaded = True
                    logger.info("[戦略2] 成功")
                except Exception as e:
                    logger.error(f"[戦略2] 失敗: {e}")
                    import traceback
                    traceback.print_exc()

            # ── 戦略3: Hydra を手動初期化してから正規ロード ──
            if not loaded:
                logger.info("[戦略3] Hydra を手動初期化して再試行")
                try:
                    from hydra.core.global_hydra import GlobalHydra
                    from hydra import initialize_config_dir

                    # yaml の親ディレクトリを config_dir として登録
                    config_dir = os.path.dirname(yaml_path)
                    # さらに2階層上（sam2パッケージルート）も試す
                    sam2_pkg_dir = os.path.dirname(
                        os.path.dirname(config_dir)
                    )

                    for init_dir in [sam2_pkg_dir, config_dir]:
                        try:
                            if GlobalHydra.instance().is_initialized():
                                GlobalHydra.instance().clear()

                            abs_dir = os.path.abspath(init_dir)
                            logger.info(f"[戦略3] Hydra 初期化: {abs_dir}")

                            with initialize_config_dir(
                                config_dir=abs_dir, version_base=None
                            ):
                                for cfg_str in hydra_configs_to_try:
                                    try:
                                        self.predictor = build_sam2_video_predictor(
                                            cfg_str, self.checkpoint,
                                            device=self.device
                                        )
                                        sam2_model = build_sam2(
                                            cfg_str, self.checkpoint,
                                            device=self.device
                                        )
                                        self.image_predictor = SAM2ImagePredictor(
                                            sam2_model
                                        )
                                        self.model_cfg = cfg_str
                                        loaded = True
                                        logger.info(f"[戦略3] 成功: {cfg_str}")
                                        break
                                    except Exception:
                                        continue
                            if loaded:
                                break
                        except Exception as e:
                            logger.warning(f"[戦略3] init_dir={init_dir} 失敗: {e}")
                            continue
                except Exception as e:
                    logger.error(f"[戦略3] 失敗: {e}")
                    import traceback
                    traceback.print_exc()

            if not loaded:
                logger.error("全ての戦略で失敗しました")
                return False

            # ── torch.compile について ──
            # RTX 5080 (sm_120) + PyTorch 2.10 では torch.compile の
            # CUDAグラフ処理で AssertionError が発生するため無効化。
            # PyTorch の RTX 50xx サポートが安定したら再有効化を検討。
            # 現状は SAM2 の素の推論 + bfloat16 + TF32 で動作。
            logger.info("torch.compile: スキップ (RTX 50xx 互換性問題のため)")
            logger.info("bfloat16 + TF32 による高速化のみ適用")

            self.model_loaded = True
            logger.info("SAM2 モデルのロード完了")
            return True

        except ImportError as e:
            logger.error(f"SAM2 が見つかりません。インストールしてください: {e}")
            return False
        except Exception as e:
            logger.error(f"モデルのロードに失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _autocast(self):
        """CUDA autocast コンテキスト（別スレッドでも有効にするため共通化）"""
        import torch
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda"))

    def init_video(self, frames_dir, frame_offset=0):
        """ビデオフレームでinference_stateを初期化

        frame_offset: frames_dir の 000000.jpg が動画全体の何フレーム目かを指定する。
                      分割処理で区間だけを読み込むときに使う。
        """
        if not self.model_loaded:
            return False

        # 新しい状態を作る前に古い方を手放す。
        # 先に作ってから差し替えると、一瞬 2 本ぶんのフレームが載る。
        self.release_state()

        try:
            with self._autocast():
                self.inference_state = self.predictor.init_state(
                    video_path=frames_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
            # 標準のローダは全フレームを抱え込むので、上限つきの置き場に差し替える
            self._install_bounded_frame_store(frames_dir)
            self.frame_offset = frame_offset
            try:
                self.frame_count = len([
                    f for f in os.listdir(frames_dir)
                    if f.lower().endswith(('.jpg', '.jpeg'))
                ])
            except OSError:
                self.frame_count = 0
            return True
        except Exception as e:
            logger.error(f"ビデオ初期化エラー: {e}")
            return False

    def to_local_frame(self, frame_idx):
        """グローバルなフレーム番号を inference_state 内の番号へ変換する"""
        return frame_idx - self.frame_offset

    def holds_frame(self, frame_idx):
        """そのフレームが現在の inference_state に含まれているか"""
        if self.inference_state is None:
            return False
        local = self.to_local_frame(frame_idx)
        if local < 0:
            return False
        return self.frame_count <= 0 or local < self.frame_count

    def _install_bounded_frame_store(self, frames_dir):
        """inference_state["images"] を BoundedFrameStore へ差し替える

        init_state が作る AsyncVideoFrameLoader は裏で全フレームを読み込み、
        1920x1080・900 フレームで 20GB 超を占有する。まだ数枚しか
        読んでいないこの時点で止めて、必要な分だけ読む置き場に入れ替える。
        """
        state = self.inference_state
        if state is None:
            return
        loader = state.get("images")
        try:
            frame_names = sorted(
                (f for f in os.listdir(frames_dir)
                 if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")),
                key=lambda f: int(os.path.splitext(f)[0]),
            )
            if not frame_names:
                return
            img_paths = [os.path.join(frames_dir, f) for f in frame_names]

            import torch
            img_mean = getattr(loader, "img_mean", None)
            img_std = getattr(loader, "img_std", None)
            if img_mean is None:
                img_mean = torch.tensor(
                    (0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
            if img_std is None:
                img_std = torch.tensor(
                    (0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

            image_size = getattr(loader, "image_size", None)
            if image_size is None:
                image_size = getattr(self.predictor, "image_size", 1024)

            store = BoundedFrameStore(
                img_paths, image_size, img_mean, img_std,
                state.get("video_height"), state.get("video_width"),
                max_frames=self.frame_cache_limit,
            )
            # 先に古いローダのスレッドを止めてから差し替える
            self._release_frame_loader(loader)
            state["images"] = store
            state["num_frames"] = len(store)
            logger.info(
                f"フレーム保持を上限つきに変更: {len(store)} フレーム中 "
                f"最大 {store.max_frames} 枚をメモリに保持"
            )
        except Exception as e:
            logger.warning(
                f"フレーム保持の差し替えに失敗しました（標準の方式を使います）: {e}")

    @staticmethod
    def _release_frame_loader(loader):
        """非同期フレームローダを止め、抱えている画像テンソルを捨てる

        AsyncVideoFrameLoader は各フレームを 1024x1024 の float32
        （1 枚あたり約 12MB）としてリストに溜め込み、
        デーモンスレッドが裏で読み続ける。停止手段が用意されていないので、
        exception を立てて __getitem__ を失敗させ、ループを抜けさせる。
        """
        if loader is None:
            return
        try:
            if isinstance(loader, BoundedFrameStore):
                loader.clear()
                return
            if hasattr(loader, "exception") and loader.exception is None:
                loader.exception = RuntimeError("frame loader released")
            thread = getattr(loader, "thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
            if hasattr(loader, "images"):
                loader.images = []
        except Exception as e:
            logger.warning(f"フレームローダの解放に失敗: {e}")

    def release_state(self):
        """inference_state を完全に解放する

        offload_video_to_cpu=True では動画フレームを CPU 側に抱え続けるため、
        次の動画を読む前にここを通さないと 2 本ぶんのフレームが同時に載る。
        """
        state = self.inference_state
        self.inference_state = None
        self.frame_offset = 0
        self.frame_count = 0
        if state is None:
            return

        try:
            self._release_frame_loader(state.get("images"))
        except Exception as e:
            logger.warning(f"inference_state の解放中にエラー: {e}")
        try:
            state.clear()          # output_dict や maskmem のテンソルを手放す
        except Exception:
            pass
        del state

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_model(self):
        """モデル本体も含めて解放する（アプリ終了時など）"""
        self.release_state()
        self.predictor = None
        self.image_predictor = None
        self.model = None
        self.model_loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def reset_state(self):
        """トラッキング状態（プロンプト）をリセットする

        inference_state が保持しているフレーム区間はそのままなので、
        frame_offset / frame_count は維持する。
        """
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)

    @staticmethod
    def is_cuda_failure(exc):
        """CUDA コンテキストが壊れた類の例外か判定する"""
        text = f"{type(exc).__name__}: {exc}"
        return any(sign in text for sign in (
            "CUDA error", "CUDA out of memory", "cudaError",
            "AcceleratorError", "device-side assert", "CUDA driver",
        ))

    def _handle_cuda_failure(self, exc):
        """CUDA が壊れたら以後 GPU を使わない状態に落とす

        一度 CUDA error が出るとコンテキストは復旧できず、以後の
        あらゆる CUDA 呼び出しが失敗する。テンソルのデストラクタまで
        例外を投げるようになり、放置すると PyQt の終了処理の途中で
        terminate されてプロセスが異常終了する（実際に発生した）。
        ここで検知して、GPU に触れないようにしておく。
        """
        if not self.is_cuda_failure(exc):
            return False
        if self.cuda_broken:
            return True
        self.cuda_broken = True
        logger.error(
            "CUDA が回復不能な状態になりました。"
            "以後 GPU 処理を停止します（復旧にはアプリの再起動が必要です）"
        )
        try:
            self.inference_state = None
            self.frame_offset = 0
            self.frame_count = 0
        except Exception:
            pass
        return True

    @staticmethod
    def _select_obj_mask(obj_id, out_obj_ids, out_mask_logits):
        """返ってきたマスク群から、目的の obj_id のぶんを取り出す

        SAM2 は inference_state に登録されている「全オブジェクト」の
        マスクを out_obj_ids と同じ並びで返す。out_mask_logits[0] を
        決め打ちすると、どのオブジェクトを指定しても常に最初に登録した
        オブジェクト（＝人物1）のマスクを拾ってしまう。
        """
        ids = list(out_obj_ids)
        if obj_id not in ids:
            return None
        return out_mask_logits[ids.index(obj_id)]

    def add_points(self, frame_idx, obj_id, points, labels):
        """
        ポイントプロンプトを追加
        points: np.ndarray shape (N, 2) - (x, y)座標
        labels: np.ndarray shape (N,) - 1=ポジティブ, 0=ネガティブ
        """
        import numpy as np
        if not self.model_loaded or self.inference_state is None or self.cuda_broken:
            return None

        if not self.holds_frame(frame_idx):
            logger.warning(
                f"フレーム {frame_idx} は現在の inference_state の範囲外です"
            )
            return None

        try:
            with self._autocast():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.to_local_frame(frame_idx),
                    obj_id=obj_id,
                    points=np.array(points, dtype=np.float32),
                    labels=np.array(labels, dtype=np.int32),
                )
            logits = self._select_obj_mask(obj_id, out_obj_ids, out_mask_logits)
            if logits is None:
                logger.error(f"obj_id {obj_id} のマスクが返りませんでした")
                return None
            mask = (logits > 0.0).cpu().numpy().squeeze()
            return _downscale_mask_for_storage(mask)
        except Exception as e:
            logger.error(f"ポイント追加エラー: {e}")
            self._handle_cuda_failure(e)
            return None

    def add_box(self, frame_idx, obj_id, box):
        """バウンディングボックスプロンプトを追加"""
        import numpy as np
        if not self.model_loaded or self.inference_state is None:
            return None

        if not self.holds_frame(frame_idx):
            logger.warning(
                f"フレーム {frame_idx} は現在の inference_state の範囲外です"
            )
            return None

        try:
            with self._autocast():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.to_local_frame(frame_idx),
                    obj_id=obj_id,
                    box=np.array(box, dtype=np.float32),
                )
            logits = self._select_obj_mask(obj_id, out_obj_ids, out_mask_logits)
            if logits is None:
                logger.error(f"obj_id {obj_id} のマスクが返りませんでした")
                return None
            mask = (logits > 0.0).cpu().numpy().squeeze()
            return _downscale_mask_for_storage(mask)
        except Exception as e:
            logger.error(f"ボックス追加エラー: {e}")
            self._handle_cuda_failure(e)
            return None

    def estimate_state_mb(self, num_frames):
        """伝播に必要な CPU メモリの概算 (MB)

        実測値ベース: フレーム保持は上限つきなので一定、
        伝播で溜まる状態がフレームあたり約 6MB。
        """
        return self.frame_cache_limit * 12 + num_frames * 6.0

    def propagate(self, start_frame=0, reverse=False, anchor_frame=None):
        """
        マスクを伝播

        start_frame : 前方伝播で、このフレーム以降のみ結果を返す（グローバル番号）。
                      reverse=True のときは適用しない（逆方向では意味を持たないため）。
        reverse     : True で anchor_frame から 0 に向かって逆方向へ伝播する。
                      動画の途中で修正ポイントを打ったとき、それより前の
                      フレームにも修正を届けるために使う。
        anchor_frame: 推論の開始フレーム（グローバル番号）。None なら SAM2 の
                      既定（最も早いプロンプトフレーム）から始まる。
                      ここを指定すると手前のフレームを再推論せずに済むが、
                      メモリの根拠が無いフレームを起点にすると精度が落ちるため、
                      プロンプトのあるフレームを渡すこと。
        yields: (frame_idx, {obj_id: mask})  frame_idx はグローバル番号
        """
        if not self.model_loaded or self.inference_state is None or self.cuda_broken:
            return

        kwargs = {"reverse": reverse}
        if anchor_frame is not None:
            kwargs["start_frame_idx"] = self.to_local_frame(anchor_frame)

        try:
            with self._autocast():
                for out_frame_idx, out_obj_ids, out_mask_logits in \
                        self.predictor.propagate_in_video(
                            self.inference_state, **kwargs):
                    global_idx = out_frame_idx + self.frame_offset
                    if not reverse and global_idx < start_frame:
                        continue
                    masks = {}
                    for i, obj_id in enumerate(out_obj_ids):
                        m = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        masks[obj_id] = _downscale_mask_for_storage(m)
                    yield global_idx, masks
        except Exception as e:
            logger.error(f"伝播エラー: {e}")
            self._handle_cuda_failure(e)
            import traceback
            traceback.print_exc()

    def segment_single_frame(self, frame_bgr, points=None, labels=None, box=None, prev_mask=None):
        """
        単一フレームでのセグメンテーション（プレビュー用）
        prev_mask: 既存のバイナリマスク (H, W) を渡すと、SAM にヒントとして入力して
                   既存領域を保ったまま点で追加/削除できる。
        """
        import numpy as np
        if not self.model_loaded:
            return None

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.image_predictor.set_image(frame_rgb)

            mask_input = None
            multimask = True
            if prev_mask is not None and prev_mask.any():
                # SAM は 256x256 の低解像度ロジットを期待する
                low = cv2.resize(
                    prev_mask.astype(np.float32), (256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                # バイナリを粗いロジットへ変換 (pos=+10, neg=-10)
                low = low * 20.0 - 10.0
                mask_input = low[None, :, :]
                # mask_input 使用時は multimask_output=False を推奨
                multimask = False

            masks, scores, _ = self.image_predictor.predict(
                point_coords=np.array(points, dtype=np.float32) if points else None,
                point_labels=np.array(labels, dtype=np.int32) if labels else None,
                box=np.array(box, dtype=np.float32) if box else None,
                mask_input=mask_input,
                multimask_output=multimask,
            )

            best_idx = int(np.argmax(scores))
            return _downscale_mask_for_storage(masks[best_idx])
        except Exception as e:
            logger.error(f"単一フレームセグメンテーションエラー: {e}")
            self._handle_cuda_failure(e)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# ビデオI/Oハンドラ
# ══════════════════════════════════════════════════════════════════════════════

class VideoHandler:
    """ビデオの読み書きを管理

    UHD/4K を開いたときは FHD 枠に収まる「プロキシ」へ縮小し、
    get_frame() は常にプロキシ解像度を返す。編集・マスク・プレビューは
    すべてこの解像度で行い、元解像度は get_source_frame() /
    open_source_reader() からエクスポート時にだけ読み出す。
    """

    # キャッシュはコマ数ではなくバイト数で制限する。
    # 4K の 100 コマは 2.5GB に達するため、コマ数上限では RAM を抑えられない。
    MAX_CACHE_BYTES = 192 * 1024 * 1024

    def __init__(self):
        self.cap = None
        self.path = None
        self.frames_dir = None
        self.frame_cache = {}
        self._cache_bytes = 0
        self.source_size = None     # (w, h) 元動画の解像度
        self.proxy_size = None      # (w, h) 縮小して扱う場合のみ。等倍なら None

    def open_video(self, path, use_proxy=True):
        """ビデオファイルを開く

        注意: 既存の cap を release するため、フレーム抽出スレッドなど
        この VideoHandler を使う処理が動いていない状態で呼ぶこと。
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return None

        self.path = path
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if use_proxy:
            proxy_w, proxy_h, scale = calc_proxy_size(src_w, src_h)
        else:
            proxy_w, proxy_h, scale = src_w, src_h, 1.0

        self.source_size = (src_w, src_h)
        self.proxy_size = (proxy_w, proxy_h) if scale < 1.0 else None

        info = {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": proxy_w,
            "height": proxy_h,
            "source_width": src_w,
            "source_height": src_h,
            "proxy_scale": scale,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        self.clear_cache()
        return info

    # ── プロキシ変換 ──
    def to_proxy(self, frame):
        """元解像度のフレームを作業解像度へ縮小する"""
        if frame is None or self.proxy_size is None:
            return frame
        if (frame.shape[1], frame.shape[0]) == self.proxy_size:
            return frame
        return cv2.resize(frame, self.proxy_size, interpolation=cv2.INTER_AREA)

    # ── キャッシュ ──
    def clear_cache(self):
        self.frame_cache.clear()
        self._cache_bytes = 0

    def _cache_put(self, frame_idx, frame):
        nbytes = frame.nbytes
        if nbytes > self.MAX_CACHE_BYTES:
            return
        while self.frame_cache and self._cache_bytes + nbytes > self.MAX_CACHE_BYTES:
            oldest = next(iter(self.frame_cache))
            self._cache_bytes -= self.frame_cache.pop(oldest).nbytes
        self.frame_cache[frame_idx] = frame
        self._cache_bytes += nbytes

    def get_frame(self, frame_idx):
        """作業解像度（プロキシ）のフレームを取得（キャッシュ付き）"""
        cached = self.frame_cache.get(frame_idx)
        if cached is not None:
            return cached.copy()

        if self.cap is None:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = self.to_proxy(frame)
        self._cache_put(frame_idx, frame)
        return frame.copy()

    def get_source_frame(self, frame_idx):
        """元解像度のフレームを取得（キャッシュしない）。エクスポート用。"""
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def open_source_reader(self):
        """元解像度を順次読みする専用の VideoCapture を開く。

        エクスポートは別スレッドで走るため、GUI 側が使う self.cap とは
        キャプチャを分ける（シーク競合の回避とシーケンシャル読みの高速化）。
        呼び出し側が release() する責任を持つ。
        """
        if not self.path:
            return None
        cap = cv2.VideoCapture(self.path)
        return cap if cap.isOpened() else None

    def extract_frames_to_dir(self, output_dir, callback=None,
                              start_frame=0, end_frame=None):
        """フレームをJPEGとしてディレクトリに出力（SAM2用）

        プロキシ有効時は縮小後のフレームを書き出す。SAM2 が読み込む画素数が
        4K→FHD で 1/4 になり、VRAM・RAM・ディスクいずれも大きく減る。

        start_frame / end_frame を渡すとその区間だけを出力する。
        ファイル名は区間内で 0 起番（SAM2 は連番を要求するため）なので、
        呼び出し側は SAM2Engine.init_video(frame_offset=start_frame) を
        併用してグローバルなフレーム番号と対応付ける。
        """
        if self.cap is None:
            return False

        os.makedirs(output_dir, exist_ok=True)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, start_frame)
        if end_frame is None:
            end_frame = total
        end_frame = min(end_frame, total)
        count = max(0, end_frame - start_frame)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]

        for i in range(count):
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(output_dir, f"{i:06d}.jpg"),
                self.to_proxy(frame), encode_params
            )
            if callback:
                callback(i, count)

        # 抽出直後は読み込み位置が末尾にあるので先頭へ戻しておく
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.path = None
        self.clear_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 処理スレッド
# ══════════════════════════════════════════════════════════════════════════════

class FrameExtractionThread(QThread):
    """フレーム抽出用バックグラウンドスレッド"""
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, video_handler, output_dir):
        super().__init__()
        self.video_handler = video_handler
        self.output_dir = output_dir

    def run(self):
        try:
            self.video_handler.extract_frames_to_dir(
                self.output_dir,
                callback=lambda i, t: self.progress.emit(i, t)
            )
            self.finished_signal.emit(True, self.output_dir)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class PropagationThread(QThread):
    """SAM2マスク伝播用バックグラウンドスレッド（キャンセル対応）

    passes に (anchor_frame, reverse) を並べると、その順で続けて伝播する。
    修正ポイントを打ったフレームを起点に前方・後方の両方へ広げる、
    という使い方をするためのもの。
    """
    progress = pyqtSignal(int, int, dict)   # frame_idx, expected, masks
    time_info = pyqtSignal(str)             # 残り時間等の情報文字列
    finished_signal = pyqtSignal(bool)

    def __init__(self, engine, total_frames, start_frame=0,
                 passes=None, expected_frames=None):
        super().__init__()
        self.engine = engine
        self.total_frames = total_frames
        self.start_frame = start_frame
        # 既定は従来どおり「start_frame から前方へ 1 回」
        self.passes = list(passes) if passes else [(None, False)]
        self.expected_frames = (
            expected_frames if expected_frames is not None
            else max(1, total_frames - start_frame)
        )
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            start_time = time.time()
            processed = 0

            for anchor, reverse in self.passes:
                if self._cancelled:
                    break

                for frame_idx, masks in self.engine.propagate(
                        start_frame=self.start_frame,
                        reverse=reverse, anchor_frame=anchor):
                    if self._cancelled:
                        logger.info(
                            f"マスク伝播がキャンセルされました "
                            f"(フレーム {frame_idx}/{self.total_frames})"
                        )
                        break

                    self.progress.emit(frame_idx, self.expected_frames, masks)
                    processed += 1

                    # 残り時間を計算
                    if processed >= 3:
                        elapsed = time.time() - start_time
                        fps = processed / elapsed
                        remaining = (
                            (self.expected_frames - processed) / fps
                            if fps > 0 else 0
                        )
                        mins, secs = divmod(int(max(0, remaining)), 60)
                        direction = "◀ 逆方向 " if reverse else ""
                        self.time_info.emit(
                            f"{direction}フレーム {frame_idx}  "
                            f"({processed}/{self.expected_frames}, {fps:.1f} fps)  "
                            f"残り約 {mins}分{secs:02d}秒"
                        )

            self.finished_signal.emit(not self._cancelled)
        except Exception as e:
            logger.error(f"伝播スレッドエラー: {e}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False)


class ExportThread(QThread):
    """ビデオエクスポート用バックグラウンドスレッド

    マスク・手動編集は作業解像度（プロキシ）で保持されているが、
    書き出しは元動画の解像度で行う。マスクは小さいまま合成・フェザリングし、
    最後に float のまま元解像度へ線形拡大するので、
    4K でも巨大な二値マスクを持ち回らずに済む。
    """
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, video_handler, state, all_masks, output_path, blender=None):
        super().__init__()
        self.video_handler = video_handler
        self.state = state
        self.all_masks = all_masks
        self.output_path = output_path
        self._blend_buf = None      # フレームごとの再確保を避ける作業バッファ
        # GPU 処理系は呼び出し側と共有する（CUDA コンテキストの作り直しを避ける）
        self._blender = blender
        self._owns_blender = blender is None

    def run(self):
        cap = None
        out = None
        try:
            if self._blender is None:
                self._blender = GpuBlender()
                self._blender.warmup()

            # 書き出しは元解像度。プロキシ未使用なら作業解像度と同じ。
            out_w = self.state.source_width or self.state.width
            out_h = self.state.source_height or self.state.height
            scale = self.state.proxy_scale or 1.0

            # マスクを保持している作業解像度
            work_w = self.state.width
            work_h = self.state.height

            cap = self.video_handler.open_source_reader()
            if cap is None:
                self.finished_signal.emit(False, "元動画を開けませんでした")
                return

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                self.output_path, fourcc, self.state.fps, (out_w, out_h)
            )
            if not out.isOpened():
                self.finished_signal.emit(False, "出力ファイルを作成できませんでした")
                return

            # プロキシで作ったマスクを元解像度へ戻すぶん、
            # 見た目を揃えるためぼかし半径も同じ倍率で拡大する
            strength = max(1, int(round(self.state.blur_strength / scale)))

            for i in range(self.state.total_frames):
                ret, frame = cap.read()      # 順次読み（シークなし）
                if not ret:
                    break

                # マスクは作業解像度のまま渡す。元解像度への拡大は
                # blend_blur 側（GPU が使えれば GPU 上）で行うので、
                # CPU での拡大と PCIe 転送量を減らせる
                mask_float = build_mask_float(
                    self.state, self.all_masks, i, work_w, work_h
                )

                out.write(self._apply_blur(frame, mask_float, strength))
                self.progress.emit(i, self.state.total_frames)

            out.release()
            out = None
            self.finished_signal.emit(True, self.output_path)
        except Exception as e:
            logger.error(f"エクスポートエラー: {e}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, str(e))
        finally:
            if out is not None:
                out.release()
            if cap is not None:
                cap.release()
            self._blend_buf = None
            if self._owns_blender and self._blender is not None:
                self._blender.close()
                self._blender = None
            gc.collect()

    def _apply_blur(self, frame, mask_float, strength):
        """背景にぼかしを適用（mask_float は作業解像度でも可）"""
        # CPU に落ちたときのために、float32 のバッファは使い回す
        if self._blend_buf is None or self._blend_buf.shape != frame.shape:
            self._blend_buf = np.empty(frame.shape, dtype=np.float32)
        return blend_blur(frame, mask_float, self.state.blur_type,
                          strength, self._blend_buf, self._blender)


# ══════════════════════════════════════════════════════════════════════════════
# カスタムビューポート（フレーム表示 + インタラクション）
# ══════════════════════════════════════════════════════════════════════════════

class SafeComboBox(QComboBox):
    """ドロップダウンが画面に残り続ける問題への対策付き QComboBox

    根本原因は WSLg の Wayland で Qt6 が grabbing popup を作れないこと
    （qt.qpa.wayland: Failed to create grabbing popup）で、これは
    select_qt_platform() が xcb を選ぶことで回避している。
    このクラスはその上に二重の保険をかける:

      1. hidePopup() でポップアップのウィンドウを明示的に閉じる
      2. 項目が選ばれた時点（activated）でも閉じにいく
         ── hidePopup() が呼ばれない経路があっても取りこぼさない
      3. それでも Wayland で動かされた場合は、コンテナを Popup から
         Tool ウィンドウへ変えて開閉が成立するようにする
    """

    POPUP_STYLE = """
        QComboBox {
            background: #2a2a2a; border: 1px solid #555;
            border-radius: 3px; color: #ddd; padding: 4px;
        }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox QAbstractItemView {
            background: #2a2a2a; color: #ddd;
            border: 1px solid #555; outline: none;
            selection-background-color: #0060aa;
            selection-color: #ffffff;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # ネイティブポップアップではなく Qt ウィジェットのビューを使わせる
        self.setView(QListView())
        self.setStyleSheet(self.POPUP_STYLE)
        self._popup_open = False
        self._container = None
        self._wayland_fallback_done = False
        # hidePopup() を経由しない選択経路があっても閉じられるようにする
        self.activated.connect(self._on_activated)

    # ── ポップアップのウィンドウを特定する ──
    def _popup_containers(self):
        """このコンボボックスに属するポップアップウィンドウを列挙する"""
        found = []
        view = self.view()
        if view is not None:
            win = view.window()
            if win is not None and win is not self.window():
                found.append(win)
        if self._container is not None and self._container not in found:
            found.append(self._container)
        # QComboBoxPrivateContainer は自分自身を親に持つトップレベルなので、
        # 取りこぼしがあってもここで拾える
        app = QApplication.instance()
        if app is not None:
            for widget in app.topLevelWidgets():
                if widget.parent() is self and widget not in found:
                    found.append(widget)
        return found

    def _force_close_popup(self):
        if self._popup_open:
            return              # 開き直された直後なら触らない
        for container in self._popup_containers():
            if container.isVisible():
                container.hide()

    def _on_activated(self, _index):
        self._popup_open = False
        self._force_close_popup()

    def _apply_wayland_fallback(self):
        """Wayland で動いている場合、コンテナを Popup から Tool へ切り替える

        Wayland では Popup ウィンドウの grab に失敗して開閉そのものが
        成立しないため、Tool ウィンドウへ落として最低限使えるようにする。
        フラグ変更は最初のポップアップを開く「前」に済ませる必要がある
        （開いた後だと初回の表示が失われる）。
        """
        if self._wayland_fallback_done:
            return
        app = QApplication.instance()
        if app is None or app.platformName() != "wayland":
            self._wayland_fallback_done = True    # xcb 等では何もしない
            return

        containers = self._popup_containers()
        if not containers:
            return                                # コンテナ未生成。次回に回す
        for container in containers:
            if container.windowFlags() & Qt.WindowType.Popup:
                container.setWindowFlags(
                    Qt.WindowType.Tool
                    | Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.NoDropShadowWindowHint
                )
        self._wayland_fallback_done = True

    # ── 開閉 ──
    def showPopup(self):
        self._apply_wayland_fallback()
        self._popup_open = True
        super().showPopup()
        containers = self._popup_containers()
        self._container = containers[0] if containers else None

    def hidePopup(self):
        self._popup_open = False
        super().hidePopup()
        self._force_close_popup()
        # 直後にブロッキング処理へ入ってもゴーストが残らないよう、
        # イベントループが 1 周した時点でもう一度閉じる
        QTimer.singleShot(0, self._force_close_popup)


class FrameViewport(QGraphicsView):
    """
    フレーム表示用カスタムビューポート
    - ズーム / パン
    - クリックでポイントプロンプト
    - ブラシ / 消しゴムでマスク編集
    """
    point_clicked = pyqtSignal(float, float, int)   # x, y, label(1=pos, 0=neg)
    brush_painted = pyqtSignal(float, float, float)  # x, y, radius
    eraser_painted = pyqtSignal(float, float, float)  # x, y, radius
    stroke_finished = pyqtSignal()                   # ブラシ/消しゴムのドラッグ終了

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = None
        self.overlay_item = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._tool_mode = ToolMode.SELECT_PERSON
        self._brush_size = 20
        self._is_painting = False

        # ビュー設定
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setMinimumSize(640, 360)

    def set_frame(self, pixmap: QPixmap, overlay: QPixmap = None):
        """フレームを表示"""
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        if overlay:
            self.overlay_item = self.scene.addPixmap(overlay)
            self.overlay_item.setOpacity(0.4)
        self.setSceneRect(QRectF(pixmap.rect()))
        if self._zoom == 1.0:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_tool(self, mode: ToolMode):
        self._tool_mode = mode
        if mode == ToolMode.PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode in (ToolMode.ADD_MASK, ToolMode.REMOVE_MASK):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == ToolMode.NEGATIVE_POINT:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_brush_size(self, size):
        self._brush_size = size

    def wheelEvent(self, event: QWheelEvent):
        """ズーム"""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom *= factor
            self.scale(factor, factor)
        else:
            self._zoom /= factor
            self.scale(1 / factor, 1 / factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton or \
           self._tool_mode == ToolMode.PAN:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if self._tool_mode == ToolMode.SELECT_PERSON:
            if event.button() == Qt.MouseButton.LeftButton:
                self.point_clicked.emit(scene_pos.x(), scene_pos.y(), 1)
                event.accept()
                return
        elif self._tool_mode == ToolMode.NEGATIVE_POINT:
            if event.button() == Qt.MouseButton.LeftButton:
                self.point_clicked.emit(scene_pos.x(), scene_pos.y(), 0)
                event.accept()
                return
        elif self._tool_mode == ToolMode.ADD_MASK:
            self._is_painting = True
            self.brush_painted.emit(scene_pos.x(), scene_pos.y(), self._brush_size)
            event.accept()
            return
        elif self._tool_mode == ToolMode.REMOVE_MASK:
            self._is_painting = True
            self.eraser_painted.emit(scene_pos.x(), scene_pos.y(), self._brush_size)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        if self._is_painting and self._tool_mode == ToolMode.ADD_MASK:
            self.brush_painted.emit(scene_pos.x(), scene_pos.y(), self._brush_size)
        elif self._is_painting and self._tool_mode == ToolMode.REMOVE_MASK:
            self.eraser_painted.emit(scene_pos.x(), scene_pos.y(), self._brush_size)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            if self._tool_mode == ToolMode.PAN:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.set_tool(self._tool_mode)
            return

        if self._is_painting:
            self._is_painting = False
            # 1 ストローク = Undo 1 件にまとめるため、離した時点を通知する
            self.stroke_finished.emit()
        super().mouseReleaseEvent(event)

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 1.0
        if self.sceneRect():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ══════════════════════════════════════════════════════════════════════════════
# タイムラインウィジェット
# ══════════════════════════════════════════════════════════════════════════════

class TimelineWidget(QWidget):
    """フレーム選択タイムライン"""
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.current_frame = 0
        self.keyframes = set()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # 再生コントロール
        self.btn_start = QPushButton("⏮")
        self.btn_prev = QPushButton("◀")
        self.btn_play = QPushButton("▶")
        self.btn_next = QPushButton("▶")
        self.btn_end = QPushButton("⏭")

        for btn in [self.btn_start, self.btn_prev, self.btn_play,
                     self.btn_next, self.btn_end]:
            btn.setFixedSize(36, 28)
            btn.setStyleSheet("""
                QPushButton {
                    background: #3a3a3a; border: 1px solid #555;
                    border-radius: 4px; color: #ddd; font-size: 12px;
                }
                QPushButton:hover { background: #4a4a4a; }
                QPushButton:pressed { background: #2a2a2a; }
            """)

        # タイムラインスライダー
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444; height: 8px;
                background: #2a2a2a; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4; border: 1px solid #0060aa;
                width: 14px; margin: -4px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #0060aa; border-radius: 4px;
            }
        """)

        # フレーム番号表示
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setStyleSheet("color: #aaa; font-size: 11px; min-width: 80px;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # フレームジャンプ
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setStyleSheet("""
            QSpinBox {
                background: #2a2a2a; border: 1px solid #555;
                border-radius: 3px; color: #ddd; padding: 2px 4px;
            }
        """)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_end)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_spin)

        # シグナル接続
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.frame_spin.valueChanged.connect(self._on_spin_changed)
        self.btn_start.clicked.connect(lambda: self.go_to_frame(0))
        self.btn_end.clicked.connect(lambda: self.go_to_frame(self.total_frames - 1))
        self.btn_prev.clicked.connect(lambda: self.go_to_frame(self.current_frame - 1))
        self.btn_next.clicked.connect(lambda: self.go_to_frame(self.current_frame + 1))

        # 再生タイマー
        self._playing = False
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._play_tick)
        self.btn_play.clicked.connect(self._toggle_play)

    def setup(self, total_frames, fps=30.0):
        self.total_frames = total_frames
        self._fps = fps
        self.slider.setMaximum(max(0, total_frames - 1))
        self.frame_spin.setMaximum(max(0, total_frames - 1))
        self.frame_label.setText(f"0 / {total_frames - 1}")

    def go_to_frame(self, idx):
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(idx)
        self.frame_spin.blockSignals(False)
        self.frame_label.setText(f"{idx} / {self.total_frames - 1}")
        self.frame_changed.emit(idx)

    def _on_slider_changed(self, val):
        self.go_to_frame(val)

    def _on_spin_changed(self, val):
        self.go_to_frame(val)

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self._playing = True
            self._play_timer.start(int(1000 / self._fps))
            self.btn_play.setText("⏸")

    def _play_tick(self):
        if self.current_frame >= self.total_frames - 1:
            self._toggle_play()
            return
        self.go_to_frame(self.current_frame + 1)


# ══════════════════════════════════════════════════════════════════════════════
# メインウィンドウ
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """メインアプリケーションウィンドウ"""

    PERSON_COLORS = [
        QColor(220, 50, 50),    # 濃いめの赤
        QColor(255, 100, 50),   # オレンジ
        QColor(50, 220, 100),   # 緑
        QColor(255, 50, 200),   # ピンク
        QColor(200, 200, 50),   # 黄
    ]

    def __init__(self):
        super().__init__()

        self.state = ProjectState()
        self.sam2 = SAM2Engine()
        self.video = VideoHandler()
        self.all_masks = {}         # {track_id: {frame_idx: mask}}
        self.current_track_id = 0
        self._frames_dir = None

        # 変更履歴（30 操作まで遡れる）
        self._undo_stack = UndoStack()
        self._stroke_open = False   # ブラシ/消しゴムのドラッグ中かどうか
        self._points_dirty = False  # ポイントの変更が他フレームへ未反映か
        self._prop_done = 0         # 伝播の進捗カウンタ
        # ぼかし・合成の GPU 処理（CUDA が無ければ自動で CPU に落ちる）
        self._blender = GpuBlender()
        self._blender.warmup()

        self._setup_ui()
        self._setup_shortcuts()
        self._apply_stylesheet()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._autosave_dir = os.path.join(base_dir, "autosave")
        # 動画を開くときは input/、書き出すときは output/ を初期表示にする
        self._input_dir = os.path.join(base_dir, "input")
        self._output_dir = os.path.join(base_dir, "output")
        for d in (self._autosave_dir, self._input_dir, self._output_dir):
            os.makedirs(d, exist_ok=True)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._auto_save)
        self._autosave_timer.start(300_000)

    # ── UI構築 ─────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("Video Blur Studio — SAM2 セグメンテーション")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 950)

        # ─── メニューバー ───
        menubar = self.menuBar()

        file_menu = menubar.addMenu("ファイル(&F)")
        act_open = file_menu.addAction("動画を開く(&O)")
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_video)

        act_export = file_menu.addAction("エクスポート(&E)")
        act_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_export.triggered.connect(self._export_video)

        act_export_split = file_menu.addAction("分割エクスポート (30秒区切り)(&D)")
        act_export_split.triggered.connect(self._export_split_video)

        act_split_process = file_menu.addAction("分割して処理 (長尺動画向け)(&P)")
        act_split_process.triggered.connect(self._split_and_process_video)

        file_menu.addSeparator()
        act_save_proj = file_menu.addAction("プロジェクト保存(&S)")
        act_save_proj.setShortcut(QKeySequence("Ctrl+S"))
        act_save_proj.triggered.connect(self._save_project)

        act_load_proj = file_menu.addAction("プロジェクト読込(&L)")
        act_load_proj.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_load_proj.triggered.connect(self._load_project)

        edit_menu = menubar.addMenu("編集(&E)")
        self.act_undo = edit_menu.addAction("元に戻す(&U)")
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self._undo)
        self.act_undo.setEnabled(False)

        self.act_redo = edit_menu.addAction("やり直す(&R)")
        self.act_redo.setShortcuts(
            [QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
        )
        self.act_redo.triggered.connect(self._redo)
        self.act_redo.setEnabled(False)

        edit_menu.addSeparator()
        act_free_mem = edit_menu.addAction("メモリを解放(&M)")
        act_free_mem.setToolTip(
            "SAM2 の作業状態とフレームキャッシュを解放します。\n"
            "マスクとポイントは残ります。"
        )
        act_free_mem.triggered.connect(self._free_memory_now)

        model_menu = menubar.addMenu("モデル(&M)")
        act_load_model = model_menu.addAction("SAM2 モデルをロード")
        act_load_model.triggered.connect(self._load_model_dialog)

        help_menu = menubar.addMenu("ヘルプ(&H)")
        act_refine = help_menu.addAction("小物がマスクから外れるとき(&R)")
        act_refine.triggered.connect(self._show_refine_help)

        act_about = help_menu.addAction("バージョン情報")
        act_about.triggered.connect(self._show_about)

        # ─── ツールバー ───
        toolbar = QToolBar("ツール")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("""
            QToolBar {
                background: #2d2d2d; border-bottom: 1px solid #444;
                padding: 4px; spacing: 6px;
            }
            QToolButton {
                background: #3a3a3a; border: 1px solid #555;
                border-radius: 4px; padding: 6px 10px;
                color: #ddd; font-size: 12px;
            }
            QToolButton:hover { background: #4a4a4a; }
            QToolButton:checked { background: #0078d4; border-color: #0060aa; }
        """)
        self.addToolBar(toolbar)

        self.tool_buttons = {}
        tools = [
            ("人物選択 (V)", ToolMode.SELECT_PERSON),
            ("除外ポイント (X)", ToolMode.NEGATIVE_POINT),
            ("ブラシ追加 (B)", ToolMode.ADD_MASK),
            ("消しゴム (E)", ToolMode.REMOVE_MASK),
            ("パン (H)", ToolMode.PAN),
        ]

        for name, mode in tools:
            btn = QToolButton()
            btn.setText(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self._set_tool(m))
            toolbar.addWidget(btn)
            self.tool_buttons[mode] = btn

        self.tool_buttons[ToolMode.SELECT_PERSON].setChecked(True)

        toolbar.addSeparator()

        # ブラシサイズ
        toolbar.addWidget(QLabel(" ブラシ: "))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(5, 100)
        self.brush_slider.setValue(20)
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.valueChanged.connect(
            lambda v: self.viewport.set_brush_size(v)
        )
        toolbar.addWidget(self.brush_slider)

        toolbar.addSeparator()

        # 伝播ボタン（全フレーム）
        propagate_style = """
            QPushButton {
                background: #0078d4; border: none; border-radius: 4px;
                color: white; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background: #1a8ae8; }
            QPushButton:disabled { background: #555; color: #999; }
        """
        self.btn_propagate = QPushButton("▶ 全フレーム伝播")
        self.btn_propagate.setStyleSheet(propagate_style)
        self.btn_propagate.clicked.connect(self._propagate_masks)
        toolbar.addWidget(self.btn_propagate)

        # 伝播ボタン（現在のフレームから）
        self.btn_propagate_from = QPushButton("▶| ここから伝播")
        self.btn_propagate_from.setStyleSheet(propagate_style.replace("#0078d4", "#0a8a4a").replace("#1a8ae8", "#12a85e"))
        self.btn_propagate_from.clicked.connect(self._propagate_from_current)
        toolbar.addWidget(self.btn_propagate_from)

        # 追加したポイントを動画全体へ反映し直す（前方＋後方）
        self.btn_repropagate = QPushButton("⟲ ポイントを反映")
        self.btn_repropagate.setToolTip(
            "現在のポイントを起点に、前方と後方の両方へ伝播し直します。\n"
            "小物が抜けたフレームでポイントを 1 つ足してから押すと、\n"
            "その修正が全フレームに反映されます。"
        )
        self.btn_repropagate.setStyleSheet(
            propagate_style.replace("#0078d4", "#7a3fa0").replace("#1a8ae8", "#9a5ac0"))
        self.btn_repropagate.clicked.connect(self._repropagate_from_points)
        toolbar.addWidget(self.btn_repropagate)

        # 伝播ボタン（1フレームだけ）
        self.btn_propagate_one = QPushButton("▶ 1フレーム")
        self.btn_propagate_one.setStyleSheet(propagate_style.replace("#0078d4", "#7a5800").replace("#1a8ae8", "#9a7010"))
        self.btn_propagate_one.clicked.connect(self._propagate_one_frame)
        toolbar.addWidget(self.btn_propagate_one)

        # ─── 中央レイアウト ───
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # スプリッター（ビューポート + サイドパネル）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ビューポート
        self.viewport = FrameViewport()
        self.viewport.point_clicked.connect(self._on_point_clicked)
        self.viewport.brush_painted.connect(self._on_brush_paint)
        self.viewport.eraser_painted.connect(self._on_eraser_paint)
        self.viewport.stroke_finished.connect(self._on_stroke_finished)
        splitter.addWidget(self.viewport)

        # ─── 右サイドパネル ───
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)
        side_panel.setFixedWidth(300)
        side_panel.setStyleSheet("background: #252525;")

        # 人物リスト
        person_group = QGroupBox("トラッキング対象")
        person_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; color: #ccc;
                border: 1px solid #444; border-radius: 4px;
                margin-top: 12px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px; padding: 0 4px;
            }
        """)
        person_layout = QVBoxLayout(person_group)

        self.person_list = QListWidget()
        self.person_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e; border: 1px solid #444;
                border-radius: 3px; color: #ddd;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background: #0060aa; }
        """)
        self.person_list.currentRowChanged.connect(self._on_person_selected)
        person_layout.addWidget(self.person_list)

        btn_row = QHBoxLayout()
        self.btn_add_person = QPushButton("＋ 追加")
        self.btn_remove_person = QPushButton("－ 削除")
        for btn in [self.btn_add_person, self.btn_remove_person]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #3a3a3a; border: 1px solid #555;
                    border-radius: 3px; color: #ddd; padding: 4px 8px;
                }
                QPushButton:hover { background: #4a4a4a; }
            """)
        self.btn_add_person.clicked.connect(self._add_person)
        self.btn_remove_person.clicked.connect(self._remove_person)
        btn_row.addWidget(self.btn_add_person)
        btn_row.addWidget(self.btn_remove_person)
        person_layout.addLayout(btn_row)

        side_layout.addWidget(person_group)

        # ぼかし設定
        blur_group = QGroupBox("ぼかし設定")
        blur_group.setStyleSheet(person_group.styleSheet())
        blur_layout = QVBoxLayout(blur_group)

        spinbox_style = """
            QSpinBox {
                background: #2a2a2a; border: 1px solid #555;
                border-radius: 3px; color: #ddd; padding: 2px 4px;
                min-width: 50px;
            }
        """

        # ぼかし強度
        blur_layout.addWidget(QLabel("ぼかし強度:"))
        blur_row = QHBoxLayout()
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 100)
        self.blur_slider.setValue(25)
        self.blur_slider.setStyleSheet(self._slider_style())
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 100)
        self.blur_spin.setValue(25)
        self.blur_spin.setStyleSheet(spinbox_style)
        self.blur_slider.valueChanged.connect(self._on_blur_slider_changed)
        self.blur_spin.valueChanged.connect(self._on_blur_spin_changed)
        blur_row.addWidget(self.blur_slider, stretch=1)
        blur_row.addWidget(self.blur_spin)
        blur_layout.addLayout(blur_row)

        # ぼかしタイプ
        blur_layout.addWidget(QLabel("ぼかしタイプ:"))
        self.blur_type_combo = SafeComboBox()
        self.blur_type_combo.addItems(["ガウシアン", "ボックス", "モーション"])
        self.blur_type_combo.currentIndexChanged.connect(self._on_blur_type_changed)
        blur_layout.addWidget(self.blur_type_combo)

        # エッジフェザリング
        blur_layout.addWidget(QLabel("エッジフェザー:"))
        feather_row = QHBoxLayout()
        self.feather_slider = QSlider(Qt.Orientation.Horizontal)
        self.feather_slider.setRange(0, 50)
        self.feather_slider.setValue(5)
        self.feather_slider.setStyleSheet(self._slider_style())
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 50)
        self.feather_spin.setValue(5)
        self.feather_spin.setStyleSheet(spinbox_style)
        self.feather_slider.valueChanged.connect(self._on_feather_slider_changed)
        self.feather_spin.valueChanged.connect(self._on_feather_spin_changed)
        feather_row.addWidget(self.feather_slider, stretch=1)
        feather_row.addWidget(self.feather_spin)
        blur_layout.addLayout(feather_row)

        # プレビューチェック
        self.chk_preview = QCheckBox("リアルタイムプレビュー")
        self.chk_preview.setChecked(True)
        self.chk_preview.setStyleSheet("color: #ccc;")
        self.chk_preview.stateChanged.connect(self._refresh_display)
        blur_layout.addWidget(self.chk_preview)

        # マスクオーバーレイ表示
        self.chk_overlay = QCheckBox("マスクオーバーレイ表示")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setStyleSheet("color: #ccc;")
        self.chk_overlay.stateChanged.connect(self._refresh_display)
        blur_layout.addWidget(self.chk_overlay)

        # ブラックアウトプレビュー（エクスポートには影響なし）
        self.chk_blackout = QCheckBox("ぼかし部分をブラックアウト表示")
        self.chk_blackout.setChecked(False)
        self.chk_blackout.setStyleSheet("color: #ccc;")
        self.chk_blackout.stateChanged.connect(self._refresh_display)
        blur_layout.addWidget(self.chk_blackout)

        side_layout.addWidget(blur_group)

        # モデル設定
        model_group = QGroupBox("モデル設定")
        model_group.setStyleSheet(person_group.styleSheet())
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("モデルサイズ:"))
        self.model_combo = SafeComboBox()
        self.model_combo.addItems(["large", "base_plus", "small", "tiny"])
        model_layout.addWidget(self.model_combo)

        self.model_status = QLabel("未ロード")
        self.model_status.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        model_layout.addWidget(self.model_status)

        self.btn_load_model = QPushButton("モデルをロード")
        self.btn_load_model.setStyleSheet("""
            QPushButton {
                background: #2d7d46; border: none; border-radius: 4px;
                color: white; padding: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #36945a; }
        """)
        self.btn_load_model.clicked.connect(self._load_model)
        model_layout.addWidget(self.btn_load_model)

        side_layout.addWidget(model_group)
        side_layout.addStretch()

        splitter.addWidget(side_panel)
        splitter.setSizes([1300, 300])

        main_layout.addWidget(splitter, stretch=1)

        # ─── タイムライン ───
        self.timeline = TimelineWidget()
        self.timeline.setStyleSheet("background: #2a2a2a; border-top: 1px solid #444;")
        self.timeline.frame_changed.connect(self._on_frame_changed)
        main_layout.addWidget(self.timeline)

        # ─── プログレスバー ───
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none; background: #2a2a2a; height: 3px;
            }
            QProgressBar::chunk {
                background: #0078d4;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # ─── ステータスバー ───
        self.statusBar().showMessage("準備完了 — 動画を開いてください")
        self.statusBar().setStyleSheet(
            "background: #252525; color: #999; border-top: 1px solid #444;"
        )

    def _setup_shortcuts(self):
        """キーボードショートカット"""
        shortcuts = {
            "V": lambda: self._set_tool(ToolMode.SELECT_PERSON),
            "X": lambda: self._set_tool(ToolMode.NEGATIVE_POINT),
            "B": lambda: self._set_tool(ToolMode.ADD_MASK),
            "E": lambda: self._set_tool(ToolMode.REMOVE_MASK),
            "H": lambda: self._set_tool(ToolMode.PAN),
            "F": lambda: self.viewport.reset_zoom(),
            "Space": lambda: self.timeline._toggle_play(),
            "Left": lambda: self.timeline.go_to_frame(self.state.current_frame - 1),
            "Right": lambda: self.timeline.go_to_frame(self.state.current_frame + 1),
            "Shift+Left": lambda: self.timeline.go_to_frame(self.state.current_frame - 10),
            "Shift+Right": lambda: self.timeline.go_to_frame(self.state.current_frame + 10),
            "Ctrl+S": lambda: self._save_current_state(),
        }

        for key, func in shortcuts.items():
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(func)
            self.addAction(action)

    def _slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #444; height: 6px;
                background: #2a2a2a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4; border: 1px solid #0060aa;
                width: 12px; margin: -4px 0; border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #0060aa; border-radius: 3px;
            }
        """

    def _apply_stylesheet(self):
        """グローバルスタイルシート"""
        self.setStyleSheet("""
            QMainWindow { background: #1e1e1e; }
            QMenuBar {
                background: #2d2d2d; color: #ddd;
                border-bottom: 1px solid #444;
            }
            QMenuBar::item:selected { background: #3d3d3d; }
            QMenu {
                background: #2d2d2d; color: #ddd; border: 1px solid #444;
            }
            QMenu::item:selected { background: #0060aa; }
            QLabel { color: #ccc; font-size: 12px; }
            QSplitter::handle { background: #444; width: 2px; }
        """)

    # ── 変更履歴（Undo / Redo）─────────────────────────────────────────────

    def _track_by_id(self, track_id):
        for track in self.state.person_tracks:
            if track.track_id == track_id:
                return track
        return None

    def _capture(self, label, mask_slots=(), edit_frames=(), roster=False):
        """指定スロットの現在値だけを抜き取ったスナップショットを作る。

        mask_slots : [(track_id, frame_idx), ...] マスクとポイントを対象にする
        edit_frames: [frame_idx, ...]             手動編集を対象にする
        roster     : True で人物リスト全体（ポイント・マスク込み）を対象にする
        値が無いスロットは None で記録し、復元時は「削除」として扱う。
        """
        snap = {
            "label": label,
            "masks": {},
            "points": {},
            "edits": {},
            "roster": None,
            "current_frame": self.state.current_frame,
            "nbytes": 0,
        }
        nbytes = 0

        if roster:
            snap["roster"], nbytes = self._capture_roster()
        else:
            for tid, frame_idx in mask_slots:
                blob = _pack_mask(self.all_masks.get(tid, {}).get(frame_idx))
                snap["masks"][(tid, frame_idx)] = blob
                if blob is not None:
                    nbytes += blob[2].nbytes
                track = self._track_by_id(tid)
                pts = track.points.get(frame_idx) if track is not None else None
                snap["points"][(tid, frame_idx)] = (
                    [tuple(p) for p in pts] if pts is not None else None
                )

        for frame_idx in edit_frames:
            blob = _pack_mask(self.state.manual_edits.get(frame_idx))
            snap["edits"][frame_idx] = blob
            if blob is not None:
                nbytes += blob[2].nbytes

        snap["nbytes"] = nbytes
        return snap

    def _capture_roster(self):
        """人物リストを丸ごと控える（人物の追加・削除の Undo 用）"""
        entries = []
        nbytes = 0
        for track in self.state.person_tracks:
            masks = {}
            for frame_idx, m in self.all_masks.get(track.track_id, {}).items():
                blob = _pack_mask(m)
                masks[frame_idx] = blob
                if blob is not None:
                    nbytes += blob[2].nbytes
            entries.append({
                "track_id": track.track_id,
                "name": track.name,
                "color": (track.color.red(), track.color.green(),
                          track.color.blue()),
                "is_visible": track.is_visible,
                "is_locked": track.is_locked,
                "points": {f: [tuple(p) for p in v]
                           for f, v in track.points.items()},
                "masks": masks,
            })
        return entries, nbytes

    def _restore_roster(self, entries):
        self.state.person_tracks.clear()
        self.all_masks.clear()
        for e in entries:
            track = PersonTrack(
                track_id=e["track_id"],
                name=e["name"],
                color=QColor(*e["color"]),
                is_visible=e["is_visible"],
                is_locked=e["is_locked"],
            )
            track.points = {f: [tuple(p) for p in v]
                            for f, v in e["points"].items()}
            # track.masks と all_masks は同じ配列オブジェクトを共有させる
            # （元の実装と同じで、実体は二重に持たない）
            masks = {f: m for f, m in
                     ((f, _unpack_mask(b)) for f, b in e["masks"].items())
                     if m is not None}
            track.masks = dict(masks)
            self.all_masks[track.track_id] = dict(masks)
            self.state.person_tracks.append(track)
        self._rebuild_person_list()

    def _rebuild_person_list(self, select_row=None):
        """人物リストのウィジェットを state から作り直す"""
        self.person_list.blockSignals(True)
        self.person_list.clear()
        for track in self.state.person_tracks:
            item = QListWidgetItem(f"● {track.name}")
            item.setForeground(QBrush(track.color))
            self.person_list.addItem(item)
        self.person_list.blockSignals(False)
        if self.state.person_tracks:
            row = 0 if select_row is None else select_row
            row = max(0, min(row, len(self.state.person_tracks) - 1))
            self.person_list.setCurrentRow(row)
            self.current_track_id = self.state.person_tracks[row].track_id

    def _push_undo(self, label, mask_slots=(), edit_frames=(), roster=False):
        """操作の「直前」に呼び、変更前の状態を履歴へ積む"""
        self._commit_undo(
            self._capture(label, mask_slots, edit_frames, roster)
        )

    def _commit_undo(self, snap):
        """_capture 済みのスナップショットを履歴へ積む"""
        self._undo_stack.push(snap)
        self._update_history_actions()

    def _begin_stroke(self, label, mask_slots=(), edit_frames=()):
        """ブラシ/消しゴムのドラッグ 1 回につき履歴を 1 件だけ積む"""
        if self._stroke_open:
            return
        self._push_undo(label, mask_slots=mask_slots, edit_frames=edit_frames)
        self._stroke_open = True

    def _on_stroke_finished(self):
        self._stroke_open = False

    def _apply_snapshot(self, snap):
        """スナップショットを適用し、適用前の状態を同じ形で返す

        戻り値をそのまま反対側のスタックへ積むことで Undo / Redo が対になる。
        """
        inverse = self._capture(
            snap["label"],
            mask_slots=list(snap["masks"].keys()),
            edit_frames=list(snap["edits"].keys()),
            roster=snap["roster"] is not None,
        )

        if snap["roster"] is not None:
            self._restore_roster(snap["roster"])

        for (tid, frame_idx), blob in snap["masks"].items():
            mask = _unpack_mask(blob)
            frames = self.all_masks.setdefault(tid, {})
            track = self._track_by_id(tid)
            if mask is None:
                frames.pop(frame_idx, None)
                if track is not None:
                    track.masks.pop(frame_idx, None)
            else:
                frames[frame_idx] = mask
                if track is not None:
                    track.masks[frame_idx] = mask

        for (tid, frame_idx), pts in snap["points"].items():
            track = self._track_by_id(tid)
            if track is None:
                continue
            if pts is None:
                track.points.pop(frame_idx, None)
            else:
                track.points[frame_idx] = [tuple(p) for p in pts]

        for frame_idx, blob in snap["edits"].items():
            mask = _unpack_mask(blob)
            if mask is None:
                self.state.manual_edits.pop(frame_idx, None)
            else:
                self.state.manual_edits[frame_idx] = mask

        return inverse

    def _undo(self):
        snap = self._undo_stack.pop_undo()
        if snap is None:
            self.statusBar().showMessage("これ以上元に戻せません")
            return
        self._undo_stack.push_redo(self._apply_snapshot(snap))
        self._after_history_change(snap, f"元に戻す: {snap['label']}")

    def _redo(self):
        snap = self._undo_stack.pop_redo()
        if snap is None:
            self.statusBar().showMessage("やり直せる操作がありません")
            return
        self._undo_stack.push_undo_only(self._apply_snapshot(snap))
        self._after_history_change(snap, f"やり直し: {snap['label']}")

    def _after_history_change(self, snap, message):
        # 操作したフレームへ戻して、何が変わったか見えるようにする
        target = snap.get("current_frame", self.state.current_frame)
        if target != self.state.current_frame and \
                0 <= target < max(1, self.state.total_frames):
            self.timeline.go_to_frame(target)   # ここで _refresh_display も走る
        else:
            self._refresh_display()
        self._update_history_actions()
        self.statusBar().showMessage(
            f"{message}　（あと {self._undo_stack.depth()} 件戻せます）"
        )

    def _update_history_actions(self):
        can_undo = self._undo_stack.can_undo()
        can_redo = self._undo_stack.can_redo()
        self.act_undo.setEnabled(can_undo)
        self.act_redo.setEnabled(can_redo)
        self.act_undo.setText(
            f"元に戻す: {self._undo_stack.undo_label()}(&U)"
            if can_undo else "元に戻す(&U)"
        )
        self.act_redo.setText(
            f"やり直す: {self._undo_stack.redo_label()}(&R)"
            if can_redo else "やり直す(&R)"
        )

    def _release_memory(self, label=None):
        """使い終わった中間バッファを能動的に返す

        伝播やエクスポートの直後は torch のキャッシュアロケータが VRAM を
        握ったままになり、CPU 側も大きな一時配列で断片化したまま残るため、
        処理の区切りで明示的に解放しておく。解放できた MB 数を返す。
        """
        before = _rss_mb()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        # gc だけでは glibc がヒープを抱えたままで RSS が下がらない
        _malloc_trim()
        freed = before - _rss_mb()
        if label and freed >= 1.0:
            logger.info(f"{label}: {freed:.0f} MB 解放 (RSS {_rss_mb():.0f} MB)")
        return freed

    def _drop_finished_threads(self):
        """終了済みのワーカースレッドへの参照を手放す

        QThread オブジェクトはシグナル接続や内部バッファを抱えたまま
        属性に残り続けるので、走っていないものは明示的に捨てる。
        """
        for attr in ("_export_thread", "_prop_thread", "_extract_thread"):
            thread = getattr(self, attr, None)
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    continue
            except RuntimeError:
                pass            # C++ 側が既に消えている
            setattr(self, attr, None)

    def _release_video_resources(self, label="メモリ解放"):
        """再生成できる重いデータを手放す（マスクなどの作業内容は残す）

        いちばん大きいのは SAM2 の inference_state で、動画フレームを
        1 枚あたり約 12MB の float32 で CPU に抱えている。
        次の動画を読む前にここを通さないと 2 本ぶんが同時に載る。
        """
        before = _rss_mb()
        self.sam2.release_state()
        self.video.clear_cache()
        self._drop_finished_threads()
        self._release_memory()
        freed = before - _rss_mb()
        if freed >= 1.0:
            logger.info(f"{label}: {freed:.0f} MB 解放 (RSS {_rss_mb():.0f} MB)")
        return freed

    def _clear_project_data(self):
        """動画に紐づく作業データ（マスク・ポイント・履歴）を捨てる"""
        for track in self.state.person_tracks:
            track.masks.clear()
            track.points.clear()
        self.state.person_tracks.clear()
        for frames in self.all_masks.values():
            frames.clear()
        self.all_masks.clear()
        self.state.manual_edits.clear()
        self.current_track_id = 0
        self._reset_history()

    def _free_memory_now(self):
        """メニューから手動でメモリを解放する

        作業内容（マスク・ポイント）は残し、作り直せるものだけ捨てる。
        SAM2 の状態は次の伝播時に自動で再構築される。
        """
        before = _rss_mb()
        freed = self._release_video_resources(label="手動メモリ解放")
        msg = (f"メモリを解放しました: {freed:.0f} MB  "
               f"(使用量 {before:.0f} MB → {_rss_mb():.0f} MB)")
        self.statusBar().showMessage(msg)
        QMessageBox.information(
            self, "メモリ解放",
            f"{msg}\n\n"
            f"マスクとポイントはそのまま残っています。\n"
            f"SAM2 の状態は次の伝播時に自動で作り直されます。"
        )

    def _reset_history(self):
        """動画/プロジェクトを読み直したときは履歴を破棄する"""
        self._undo_stack.clear()
        self._stroke_open = False
        self._points_dirty = False
        self._update_repropagate_hint()
        self._update_history_actions()

    # ── アクション ─────────────────────────────────────────────────────────

    def _default_input_path(self):
        """「開く」系ダイアログの初期表示先"""
        os.makedirs(self._input_dir, exist_ok=True)
        return self._input_dir

    def _default_output_path(self, suffix="_2", ext=None):
        """「保存」系ダイアログの初期表示先（output/ + 元ファイル名ベース）

        出力先は常に output/ を開き、そこからユーザーが任意のフォルダへ
        移動して保存できるようにする。
        """
        os.makedirs(self._output_dir, exist_ok=True)
        if self.state.video_path:
            src = Path(self.state.video_path)
            name = f"{src.stem}{suffix}{ext or src.suffix}"
        else:
            name = f"output{suffix}{ext or '.mp4'}"
        return os.path.join(self._output_dir, name)

    def _set_tool(self, mode: ToolMode):
        for m, btn in self.tool_buttons.items():
            btn.setChecked(m == mode)
        self.viewport.set_tool(mode)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画ファイルを開く", self._default_input_path(),
            "動画ファイル (*.mp4 *.avi *.mov *.mkv *.webm);;すべて (*)"
        )
        if not path:
            return

        # 新しい動画を開く前に、前の動画ぶんの重いデータを先に手放す。
        # 開いてから解放すると inference_state が一時的に 2 本ぶん載る。
        # ここで捨てるのは作り直せるものだけなので、
        # 万一この後の読み込みに失敗しても作業内容は失われない。
        self._release_video_resources(label="前の動画の解放")

        info = self.video.open_video(path)
        if info is None:
            QMessageBox.critical(self, "エラー", "動画ファイルを開けませんでした。")
            return

        self.state.video_path = path
        self.state.fps = info["fps"]
        self.state.width = info["width"]            # 作業解像度（プロキシ）
        self.state.height = info["height"]
        self.state.source_width = info["source_width"]
        self.state.source_height = info["source_height"]
        self.state.proxy_scale = info["proxy_scale"]
        self.state.total_frames = info["total_frames"]
        self.state.current_frame = 0
        self._clear_project_data()
        self._release_memory(label="前のプロジェクトデータの解放")

        self.timeline.setup(info["total_frames"], info["fps"])
        self.person_list.clear()

        # 最初のフレームを表示
        self._on_frame_changed(0)

        # 4. ウィンドウタイトルにファイル名を表示
        self.setWindowTitle(
            f"Video Blur Studio — {Path(path).name}"
        )

        if self.state.proxy_scale < 1.0:
            size_desc = (
                f"{info['source_width']}x{info['source_height']} → "
                f"{info['width']}x{info['height']} で編集"
            )
            self.statusBar().showMessage(
                f"読込完了: {Path(path).name}  ({size_desc}, "
                f"{info['fps']:.1f}fps, {info['total_frames']}フレーム)"
            )
            QMessageBox.information(
                self, "プロキシモード",
                f"{info['source_width']}x{info['source_height']} の動画を読み込みました。\n\n"
                f"メモリと処理時間を抑えるため、編集は "
                f"{info['width']}x{info['height']} に縮小して行います。\n"
                f"エクスポートは元の "
                f"{info['source_width']}x{info['source_height']} に戻して書き出されます。"
            )
        else:
            self.statusBar().showMessage(
                f"読込完了: {Path(path).name}  "
                f"({info['width']}x{info['height']}, "
                f"{info['fps']:.1f}fps, "
                f"{info['total_frames']}フレーム)"
            )

        # 同じ動画の自動保存があれば復帰を提案する
        # （承諾された場合は _load_project_file 側でフレーム準備まで完結する）
        if self._offer_auto_saved(path):
            return

        # SAM2 用フレームの準備（フォルダ決定 → 抽出 → inference_state 初期化）
        self._frames_dir = self._frames_dir_for(path)
        frames_ready = self._prepare_frames_for_sam2(info)

        if frames_ready:
            self.statusBar().showMessage("フレーム準備完了")

    def _frames_dir_for(self, path):
        """SAM2 用フレームの置き場所を決める

        SAM2 はパスに非 ASCII があると読み込みに失敗するため、TEMP 配下に
        ハッシュ名のフォルダを作る。プロキシ解像度が変わると書き出す画像も
        変わるので、解像度もハッシュに混ぜてキャッシュを分ける。
        """
        import hashlib, tempfile
        key = f"{path}|{self.state.width}x{self.state.height}"
        path_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), "vbs_frames", path_hash)

    def _prepare_frames_for_sam2(self, info):
        """
        SAM2 用のフレームを準備し、inference_state を初期化する。
        1. フレームがない、または不足している場合抽出
        2. SAM2 モデルがロード済みか確認
        3. inference_state を初期化
        """
        # フレームディレクトリ作成
        self._frames_dir = self._frames_dir_for(self.state.video_path)
        os.makedirs(self._frames_dir, exist_ok=True)

        # フレーム抽出が必要か確認
        missing_frames = self._count_missing_frames(self._frames_dir, info["total_frames"])

        if missing_frames > 0:
            # 抽出が必要
            reply = QMessageBox.question(
                self, "フレーム抽出",
                f"{missing_frames} フレームが必要です。\n抽出しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # スレッドで抽出し、完了まで実際に待つ
                self._extract_frames()
                self._wait_for_thread(
                    self._extract_thread,
                    f"フレーム抽出中... ({missing_frames} フレーム)"
                )

        # SAM2 初期化
        extracted = self._count_existing_frames(
            self._frames_dir, info["total_frames"]
        )

        if not self.sam2.model_loaded:
            # モデル未ロードは異常ではない（ロード時に改めて init_video される）
            self.statusBar().showMessage(
                f"フレーム準備済み ({extracted}/{info['total_frames']})　"
                f"— 「モデルをロード」を実行してください"
            )
            return False

        if extracted < info["total_frames"]:
            QMessageBox.warning(
                self, "警告",
                f"フレームが不足しています "
                f"({extracted}/{info['total_frames']})。\n"
                f"抽出をキャンセルした場合は、動画を開き直してください。"
            )
            return False

        if self.sam2.init_video(self._frames_dir):
            self.statusBar().showMessage("SAM2 準備完了")
            return True

        QMessageBox.warning(
            self, "警告",
            "SAM2 の初期化に失敗しました。\n"
            "動画が長い場合はメモリ不足の可能性があります。"
        )
        return False

    def _count_existing_frames(self, frames_dir, total_frames):
        """指定ディレクトリの既存フレーム数をカウント"""
        if not os.path.exists(frames_dir):
            return 0
        try:
            files = os.listdir(frames_dir)
            # .jpg, .jpeg ファイルのみカウント
            frame_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg')))
            return min(frame_count, total_frames)
        except Exception:
            return 0

    def _count_missing_frames(self, frames_dir, total_frames):
        """指定ディレクトリのフレーム不足数をカウント"""
        existing = self._count_existing_frames(frames_dir, total_frames)
        return total_frames - existing

    def _split_and_process_video(self, segment_duration=30):
        """長尺動画を区間ごとに処理し、最後に 1 本へ結合する

        SAM2 の inference_state は読み込んだフレームを丸ごと抱えるため、
        長い動画をそのまま init_video するとメモリが足りなくなる。
        そこで segment_duration 秒ずつ読み込み、区間ごとに
        「ポイント指定 → 伝播 → その区間を書き出し」を繰り返す。

        マスクとポイントはグローバルなフレーム番号で保持し、
        SAM2 へ渡す直前に SAM2Engine.frame_offset で区間内の番号へ変換する。
        """
        if not self.state.video_path:
            QMessageBox.warning(self, "エラー", "動画が読み込まれていません。")
            return
        if not self.sam2.model_loaded:
            QMessageBox.warning(
                self, "エラー",
                "SAM2モデルがロードされていません。\n"
                "先に「モデルをロード」を実行してください。"
            )
            return

        fps = self.state.fps or 30.0
        total = self.state.total_frames
        segment_frames = max(1, int(segment_duration * fps))
        num_segments = (total + segment_frames - 1) // segment_frames

        if num_segments <= 1:
            QMessageBox.information(
                self, "分割不要",
                f"この動画は {total / fps:.1f} 秒で、"
                f"{segment_duration} 秒以内に収まっています。\n"
                f"通常どおり伝播とエクスポートを実行してください。"
            )
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "分割処理の出力先を選択", self._default_output_path(),
            "MP4 (*.mp4);;AVI (*.avi);;すべて (*)"
        )
        if not output_path:
            return

        reply = QMessageBox.question(
            self, "分割して処理",
            f"{num_segments} 区間 × 最大 {segment_duration} 秒に分けて処理します。\n\n"
            f"各区間で人物のポイントを指定していただきます。\n"
            f"区間ごとに伝播と書き出しを行い、最後に結合します。\n\n"
            f"開始しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="vbs_split_proc_")

        # 書き出しは元解像度。マスクは作業解像度で組み立てて拡大する。
        out_w = self.state.source_width or self.state.width
        out_h = self.state.source_height or self.state.height
        work_w, work_h = self.state.width, self.state.height
        scale = self.state.proxy_scale or 1.0
        strength = max(1, int(round(self.state.blur_strength / scale)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 区間をまたいでフレーム番号は昇順なので、元解像度は 1 本で順次読みできる
        reader = self.video.open_source_reader()
        if reader is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            QMessageBox.critical(self, "エラー", "元動画を開けませんでした。")
            return

        saved_frames_dir = self._frames_dir
        saved_title = self.windowTitle()
        part_files = []
        blend_buf = None
        cancelled = False

        self.progress_bar.setVisible(True)

        try:
            for seg_idx in range(num_segments):
                start = seg_idx * segment_frames
                end = min(start + segment_frames, total)
                seg_title = (
                    f"区間 {seg_idx + 1}/{num_segments}"
                    f"（フレーム {start}-{end - 1} / {(end - start) / fps:.1f}秒）"
                )
                self.setWindowTitle(f"Video Blur Studio — {seg_title}")

                # ── 1. この区間のフレームだけを抽出 ──
                seg_frames_dir = os.path.join(tmp_dir, f"frames_{seg_idx:03d}")
                self.progress_bar.setRange(0, end - start)
                self.statusBar().showMessage(f"{seg_title}: フレーム抽出中...")

                def _on_extract(i, n, _title=seg_title):
                    self.progress_bar.setValue(i)
                    QApplication.processEvents()

                self.video.extract_frames_to_dir(
                    seg_frames_dir, callback=_on_extract,
                    start_frame=start, end_frame=end
                )

                # ── 2. この区間で SAM2 を初期化（グローバル番号との対応を付ける）──
                self._frames_dir = seg_frames_dir
                if not self.sam2.init_video(seg_frames_dir, frame_offset=start):
                    QMessageBox.critical(
                        self, "エラー", f"{seg_title}: SAM2 の初期化に失敗しました。"
                    )
                    cancelled = True
                    break

                # ── 3. ポイント指定をユーザーに任せる（ダイアログはモードレス）──
                self.timeline.go_to_frame(start)
                self._refresh_display()

                box = QMessageBox(self)
                box.setWindowTitle(seg_title)
                box.setText(
                    f"{seg_title}\n\n"
                    f"この区間の人物をビューポートでクリックしてポイントを指定してください。\n"
                    f"（このダイアログを開いたまま操作できます）\n\n"
                    f"「OK」  : 伝播してこの区間を書き出す\n"
                    f"「Skip」: ぼかしのみ（人物を保護せずに書き出す）\n"
                    f"「Abort」: 分割処理を中止する"
                )
                box.setStandardButtons(
                    QMessageBox.StandardButton.Ok
                    | QMessageBox.StandardButton.Ignore
                    | QMessageBox.StandardButton.Abort
                )
                box.button(QMessageBox.StandardButton.Ignore).setText("Skip")
                answer = self._wait_for_dialog(box)

                if answer == QMessageBox.StandardButton.Abort or \
                        answer == QMessageBox.StandardButton.Cancel:
                    cancelled = True
                    break

                # ── 4. 伝播（この区間のみ）──
                if answer == QMessageBox.StandardButton.Ok:
                    has_prompts = any(
                        any(start <= f < end for f in track.points)
                        for track in self.state.person_tracks
                    )
                    if has_prompts:
                        self._push_undo(
                            f"分割処理 区間{seg_idx + 1}",
                            mask_slots=[
                                (t.track_id, f)
                                for t in self.state.person_tracks
                                for f in range(start, end)
                            ],
                        )
                        self.statusBar().showMessage(f"{seg_title}: 伝播準備中...")
                        QApplication.processEvents()
                        if self._reinit_and_reapply_points():
                            self.progress_bar.setRange(start, end)
                            prop = PropagationThread(
                                self.sam2, end, start_frame=start
                            )
                            prop.progress.connect(self._on_propagation_progress)
                            prop.time_info.connect(
                                lambda msg, t=seg_title:
                                self.statusBar().showMessage(f"{t}: {msg}")
                            )
                            prop.start()
                            self._wait_for_thread(prop, f"{seg_title}: マスク伝播中...")
                        else:
                            QMessageBox.warning(
                                self, "警告",
                                f"{seg_title}: SAM2 の再初期化に失敗しました。\n"
                                f"この区間はぼかしのみで書き出します。"
                            )
                    else:
                        QMessageBox.information(
                            self, seg_title,
                            "この区間にポイントが指定されていません。\n"
                            "ぼかしのみで書き出します。"
                        )

                # ── 5. この区間を書き出す ──
                part_path = os.path.join(tmp_dir, f"part_{seg_idx:04d}.mp4")
                out = cv2.VideoWriter(part_path, fourcc, fps, (out_w, out_h))
                if not out.isOpened():
                    QMessageBox.critical(
                        self, "エラー", f"{seg_title}: 出力ファイルを作成できませんでした。"
                    )
                    cancelled = True
                    break

                self.progress_bar.setRange(start, end)
                self.statusBar().showMessage(f"{seg_title}: 書き出し中...")

                written = 0
                for i in range(start, end):
                    ret, frame = reader.read()
                    if not ret:
                        break

                    # 拡大は blend_blur 側（GPU が使えれば GPU 上）で行う
                    mask_float = build_mask_float(
                        self.state, self.all_masks, i, work_w, work_h
                    )

                    if blend_buf is None or blend_buf.shape != frame.shape:
                        blend_buf = np.empty(frame.shape, dtype=np.float32)
                    out.write(blend_blur(
                        frame, mask_float, self.state.blur_type,
                        strength, blend_buf, self._blender
                    ))
                    written += 1
                    self.progress_bar.setValue(i)
                    QApplication.processEvents()

                out.release()
                part_files.append(part_path)
                self.statusBar().showMessage(
                    f"{seg_title}: 完了（{written} フレーム）"
                )

                # 次の区間へ移る前に、この区間の inference_state を解放する
                self.sam2.inference_state = None
                self._release_memory()

            # ── 6. 結合 ──
            if cancelled:
                QMessageBox.information(
                    self, "中止",
                    f"分割処理を中止しました。\n"
                    f"{len(part_files)} 区間まで処理済みでしたが、結合は行いません。"
                )
            elif part_files:
                self.statusBar().showMessage("区間を結合中...")
                self.progress_bar.setRange(0, len(part_files))
                QApplication.processEvents()

                final_out = cv2.VideoWriter(
                    output_path, fourcc, fps, (out_w, out_h)
                )
                for idx, part_path in enumerate(part_files):
                    cap = cv2.VideoCapture(part_path)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        final_out.write(frame)
                    cap.release()
                    self.progress_bar.setValue(idx + 1)
                    QApplication.processEvents()
                final_out.release()

                QMessageBox.information(
                    self, "完了",
                    f"分割処理が完了しました:\n\n{output_path}\n\n"
                    f"{len(part_files)} 区間を結合"
                )
                self.statusBar().showMessage(f"分割処理完了: {output_path}")

        except Exception as e:
            logger.error(f"分割処理エラー: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "エラー", f"分割処理に失敗: {e}")
        finally:
            reader.release()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self.progress_bar.setVisible(False)
            self.setWindowTitle(saved_title)

            # 区間用の inference_state を捨て、動画全体の状態へ戻す
            self.sam2.inference_state = None
            self.sam2.frame_offset = 0
            self.sam2.frame_count = 0
            self._frames_dir = saved_frames_dir
            self._release_memory()

            if self._frames_dir and os.path.exists(self._frames_dir) and \
                    self._count_missing_frames(
                        self._frames_dir, self.state.total_frames) <= 0:
                self.sam2.init_video(self._frames_dir)
            self._refresh_display()

    def _wait_for_thread(self, thread, message="処理中..."):
        """GUI を応答させたままバックグラウンドスレッドの完了を待つ

        元は processEvents() を 100 回空回りさせるだけで、コメントの
        「約 10 秒待機」に反して一瞬で抜けていた（sleep が無いため）。
        その結果、抽出途中のフォルダで init_video を呼んでしまっていた。
        ここでは wait() で実際に眠りながら、合間にイベントを処理する。
        """
        if thread is None:
            return True
        if message:
            self.statusBar().showMessage(message)
        while thread.isRunning():
            QApplication.processEvents()
            thread.wait(50)     # 50ms 眠る（直前にイベントは捌いてある）
        # 完了シグナル（finished_signal）を受け取りきってから戻る
        QApplication.processEvents()
        return True

    def _wait_for_dialog(self, box):
        """モードレスダイアログの応答を待つ（メインウィンドウは操作可能なまま）

        分割処理では、ダイアログを出したままビューポートでポイントを
        打ってもらう必要があるため、exec() によるモーダル待ちは使えない。
        """
        clicked = {}
        box.buttonClicked.connect(
            lambda btn: clicked.setdefault("value", box.standardButton(btn))
        )
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.show()
        while box.isVisible():
            QApplication.processEvents()
            QThread.msleep(20)
        QApplication.processEvents()
        box.deleteLater()
        # × で閉じられた場合はボタン未クリックなので中止扱いにする
        return clicked.get("value", QMessageBox.StandardButton.Cancel)

    def _extract_frames(self):
        """フレームを抽出"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.state.total_frames)

        self._extract_thread = FrameExtractionThread(self.video, self._frames_dir)
        self._extract_thread.progress.connect(
            lambda i, t: self.progress_bar.setValue(i)
        )
        self._extract_thread.finished_signal.connect(self._on_frames_extracted)
        self._extract_thread.start()
        self.statusBar().showMessage("フレーム抽出中...")

    def _on_frames_extracted(self, success, result):
        self.progress_bar.setVisible(False)
        if success:
            self.statusBar().showMessage(f"フレーム抽出完了: {result}")
            # SAM2のinference_stateを初期化
            if self.sam2.model_loaded:
                if self.sam2.init_video(result):
                    self.statusBar().showMessage("SAM2 ビデオ初期化完了 — 人物を選択してください")
                    # 読み込み済みのポイントがあれば再登録する
                    if any(track.points for track in self.state.person_tracks):
                        self._reapply_saved_points()
                else:
                    self.statusBar().showMessage("SAM2 ビデオ初期化に失敗")
        else:
            QMessageBox.warning(self, "エラー", f"フレーム抽出に失敗: {result}")

    def _load_model(self):
        model_size = self.model_combo.currentText()
        # ロード中はイベントループが止まるので、開いたままのドロップダウンが
        # 画面に残らないよう先に閉じて描画を反映させる
        self.model_combo.hidePopup()
        self.model_status.setText("ロード中...")
        self.model_status.setStyleSheet("color: #ffa500; font-size: 11px;")
        QApplication.processEvents()

        # 別サイズへ入れ替える場合、前のモデルの VRAM を先に返す
        self._release_memory()

        if self.sam2.load_model(model_size):
            self.model_status.setText(f"✓ {model_size} ロード完了 ({self.sam2.device})")
            self.model_status.setStyleSheet("color: #50c878; font-size: 11px;")

            # フレームが既に抽出されていれば初期化
            if self._frames_dir and os.path.exists(self._frames_dir):
                if self.sam2.init_video(self._frames_dir):
                    self.statusBar().showMessage("SAM2 準備完了")
                    # プロジェクト読込済みのポイントがあれば再登録
                    if any(track.points for track in self.state.person_tracks):
                        self._reapply_saved_points()
        else:
            self.model_status.setText("✗ ロード失敗")
            self.model_status.setStyleSheet("color: #ff6b6b; font-size: 11px;")
            QMessageBox.warning(
                self, "モデルロード失敗",
                "SAM2モデルのロードに失敗しました。\n\n"
                "以下を確認してください:\n"
                "1. sam2パッケージがインストールされているか\n"
                "   pip install sam2\n"
                "2. チェックポイントファイルがあるか\n"
                "   ./checkpoints/sam2.1_hiera_large.pt\n"
                "3. CUDAが利用可能か"
            )

    def _load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "SAM2 チェックポイントを選択",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
            "PyTorch チェックポイント (*.pt *.pth);;すべて (*)"
        )
        if path:
            model_size = self.model_combo.currentText()
            self.model_combo.hidePopup()
            self.model_status.setText("ロード中...")
            QApplication.processEvents()
            self._release_memory()
            if self.sam2.load_model(model_size, checkpoint_path=path):
                self.model_status.setText(f"✓ ロード完了 ({self.sam2.device})")
                self.model_status.setStyleSheet("color: #50c878; font-size: 11px;")
            else:
                self.model_status.setText("✗ ロード失敗")
                self.model_status.setStyleSheet("color: #ff6b6b; font-size: 11px;")

    def _next_track_id(self):
        """未使用の track_id を返す

        リストの長さを ID にすると、削除して追加し直したときに
        既存トラックと衝突し、all_masks[id] = {} でそのトラックの
        マスクを丸ごと消してしまう。使用中の ID を避けて採番する。
        """
        used = {t.track_id for t in self.state.person_tracks}
        used |= set(self.all_masks.keys())
        track_id = 0
        while track_id in used:
            track_id += 1
        return track_id

    def _add_person(self):
        """人物トラックを追加"""
        self._push_undo("人物の追加", roster=True)
        track_id = self._next_track_id()
        color = self.PERSON_COLORS[track_id % len(self.PERSON_COLORS)]
        track = PersonTrack(
            track_id=track_id,
            name=f"人物 {track_id + 1}",
            color=color,
        )
        self.state.person_tracks.append(track)
        self.all_masks[track_id] = {}

        row = len(self.state.person_tracks) - 1
        item = QListWidgetItem(f"● {track.name}")
        item.setForeground(QBrush(color))
        self.person_list.addItem(item)
        self.person_list.setCurrentRow(row)
        self.current_track_id = track_id
        self.statusBar().showMessage(
            f"{track.name} を追加しました — ビューポートで対象をクリックしてください"
        )

    def _remove_person(self):
        row = self.person_list.currentRow()
        if row < 0:
            return
        self._push_undo("人物の削除", roster=True)
        track = self.state.person_tracks[row]
        self.state.person_tracks.pop(row)
        self.all_masks.pop(track.track_id, None)
        self.person_list.takeItem(row)

        # SAM2 側にも残っていると伝播で追跡され続けるので落としておく
        # （伝播前には必ず再構築されるが、状態を合わせておく）
        if self.sam2.inference_state is not None:
            try:
                self.sam2.predictor.remove_object(
                    self.sam2.inference_state, track.track_id, strict=False
                )
            except Exception as e:
                logger.warning(f"SAM2 からのオブジェクト削除に失敗: {e}")

        # 選択行を詰めて current_track_id を追従させる
        if self.state.person_tracks:
            new_row = min(row, len(self.state.person_tracks) - 1)
            self.person_list.setCurrentRow(new_row)
            self.current_track_id = self.state.person_tracks[new_row].track_id
        else:
            self.current_track_id = 0
        self._refresh_display()

    def _on_person_selected(self, row):
        if row >= 0 and row < len(self.state.person_tracks):
            self.current_track_id = self.state.person_tracks[row].track_id

    def _on_point_clicked(self, x, y, label):
        """ビューポートでポイントがクリックされた"""
        if not self.state.person_tracks:
            QMessageBox.information(
                self, "ヒント",
                "まず「＋ 追加」ボタンで人物トラックを追加してください。"
            )
            return

        if not self.sam2.model_loaded:
            QMessageBox.information(
                self, "ヒント",
                "まずSAM2モデルをロードしてください。"
            )
            return

        frame_idx = self.state.current_frame
        row = self.person_list.currentRow()
        if row < 0:
            self.statusBar().showMessage("人物トラックが選択されていません")
            return
        track = self.state.person_tracks[row]

        # 変更前の状態を控えておき、セグメンテーションが成功したときだけ履歴へ積む
        snap = self._capture(
            "ポイント追加" if label == 1 else "ネガティブポイント追加",
            mask_slots=[(track.track_id, frame_idx)],
        )

        # ポイントを保存
        if frame_idx not in track.points:
            track.points[frame_idx] = []
        track.points[frame_idx].append((x, y, label))

        # 全ポイントを集める
        all_pts = [(p[0], p[1]) for p in track.points[frame_idx]]
        all_labels = [p[2] for p in track.points[frame_idx]]

        # 伝播後に inference_state が解放されている場合は、
        # 動画全体を再ロードせず image_predictor で当該フレームのみ更新する
        # (全フレーム再 init_video は長尺動画で OOM を招くため回避)
        if self.sam2.inference_state is None:
            frame_bgr = self.video.get_frame(frame_idx)
            if frame_bgr is None:
                track.points[frame_idx].pop()
                self.statusBar().showMessage("フレーム取得に失敗しました")
                return
            # 既存の伝播マスクをヒントとして渡し、そこへ追加/削除する形で再生成
            prev = self.all_masks.get(track.track_id, {}).get(frame_idx)
            mask = self.sam2.segment_single_frame(
                frame_bgr, points=all_pts, labels=all_labels, prev_mask=prev
            )
        else:
            mask = self.sam2.add_points(
                frame_idx, track.track_id, all_pts, all_labels
            )

        if mask is None:
            # 追加した履歴を戻す
            track.points[frame_idx].pop()
            self.statusBar().showMessage(
                "ポイント追加に失敗しました（SAM2 状態を確認してください）"
            )
            return

        if mask is not None:
            self._commit_undo(snap)
            track.masks[frame_idx] = mask
            self.all_masks[track.track_id][frame_idx] = mask

            # このポイントが反映されるのは今のフレームだけ。
            # 他フレームに伝播済みのマスクがあるなら再伝播が要ることを伝える。
            self._update_points_dirty()

            self._refresh_display()
            label_str = "ポジティブ" if label == 1 else "ネガティブ"
            msg = (f"ポイント追加: {track.name} @ フレーム {frame_idx}  "
                   f"座標: ({x:.0f}, {y:.0f})  [{label_str}]")
            if self._points_dirty:
                msg += "　→ 全フレームに反映するには「⟲ ポイントを反映」を押してください"
            self.statusBar().showMessage(msg)
            # CLIで使う座標をコンソールにも出力
            logger.info(msg)
            logger.info(f"  → CLI用: --points \"{x:.0f},{y:.0f}\"")

    def _edit_store_size(self):
        """手動編集マスクの保持解像度

        SAM2 マスクと同じく MASK_STORE_SCALE で縮小して持つ。
        FHD なら 1 フレームあたり 2.0MB → 0.5MB。
        """
        return (max(1, int(self.state.width * MASK_STORE_SCALE)),
                max(1, int(self.state.height * MASK_STORE_SCALE)))

    def _on_brush_paint(self, x, y, radius):
        """ブラシでマスクを手動追加"""
        if self.state.width <= 0 or self.state.height <= 0:
            return
        frame_idx = self.state.current_frame
        self._begin_stroke("ブラシ", edit_frames=[frame_idx])

        w, h = self._edit_store_size()
        edit = self.state.manual_edits.get(frame_idx)
        if edit is None:
            edit = np.zeros((h, w), dtype=np.uint8)
        elif edit.shape[:2] != (h, w):
            # 旧プロジェクトなど別解像度で保存されていた場合は揃える
            edit = _to_size(edit, w, h)
        self.state.manual_edits[frame_idx] = edit

        sx = w / self.state.width
        cv2.circle(
            edit,
            (int(x * sx), int(y * h / self.state.height)),
            max(1, int(radius * sx)), 1, -1
        )
        self._refresh_display()

    def _on_eraser_paint(self, x, y, radius):
        """消しゴムでマスクを手動削除"""
        if self.state.width <= 0 or self.state.height <= 0:
            return
        frame_idx = self.state.current_frame
        self._begin_stroke(
            "消しゴム",
            mask_slots=[(t.track_id, frame_idx)
                        for t in self.state.person_tracks],
            edit_frames=[frame_idx],
        )

        # 全トラックのマスクから消す
        for track in self.state.person_tracks:
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue
            # cv2.circle は bool 配列に描けないので uint8 に起こして描き、
            # 結果を書き戻す。元実装は astype() が返す一時コピーに描いていたため
            # 消去がどこにも反映されていなかった。
            work = mask.astype(np.uint8)
            mh, mw = work.shape[:2]
            sx = mw / self.state.width
            cv2.circle(
                work,
                (int(x * sx), int(y * mh / self.state.height)),
                max(1, int(radius * sx)), 0, -1
            )
            erased = work.astype(bool)
            track.masks[frame_idx] = erased
            self.all_masks.setdefault(track.track_id, {})[frame_idx] = erased

        # 手動編集からも消す
        edit = self.state.manual_edits.get(frame_idx)
        if edit is not None:
            eh, ew = edit.shape[:2]
            sx = ew / self.state.width
            cv2.circle(
                edit,
                (int(x * sx), int(y * eh / self.state.height)),
                max(1, int(radius * sx)), 0, -1
            )

        self._refresh_display()

    def _reinit_and_reapply_points(self):
        """伝播前に inference_state をクリーンに再構築してポイントを再登録する"""
        if not self._frames_dir or not os.path.exists(self._frames_dir):
            return False
        # 区間だけを読み込んでいる場合、再初期化しても同じ区間を指すようにする
        offset = self.sam2.frame_offset
        self.sam2.reset_state()
        if not self.sam2.init_video(self._frames_dir, frame_offset=offset):
            return False
        for track in self.state.person_tracks:
            for frame_idx, pts_list in track.points.items():
                if not pts_list:
                    continue
                pts = [(p[0], p[1]) for p in pts_list]
                lbls = [p[2] for p in pts_list]
                self.sam2.add_points(frame_idx, track.track_id, pts, lbls)
        return True

    # ── 伝播の UI 状態 ─────────────────────────────────────────────────────

    PROPAGATE_STYLE = """
        QPushButton {
            background: #0078d4; border: none; border-radius: 4px;
            color: white; padding: 6px 12px; font-weight: bold;
        }
        QPushButton:hover { background: #1a8ae8; }
        QPushButton:disabled { background: #555; color: #999; }
    """

    CANCEL_STYLE = """
        QPushButton {
            background: #d43030; border: none; border-radius: 4px;
            color: white; padding: 6px 12px; font-weight: bold;
        }
        QPushButton:hover { background: #e84545; }
    """

    def _propagation_buttons(self):
        """(ボタン, 既定ラベル, 通常色, ホバー色, 押したときの動作)"""
        return (
            (self.btn_propagate, "▶ 全フレーム伝播",
             "#0078d4", "#1a8ae8", self._propagate_masks),
            (self.btn_propagate_from, "▶| ここから伝播",
             "#0a8a4a", "#12a85e", self._propagate_from_current),
            (self.btn_repropagate, "⟲ ポイントを反映",
             "#7a3fa0", "#9a5ac0", self._repropagate_from_points),
            (self.btn_propagate_one, "▶ 1フレーム",
             "#7a5800", "#9a7010", self._propagate_one_frame),
        )

    def _button_style(self, base, hover):
        return self.PROPAGATE_STYLE.replace("#0078d4", base).replace("#1a8ae8", hover)

    def _restore_propagation_buttons(self):
        for btn, label, base, hover, slot in self._propagation_buttons():
            btn.setText(label)
            btn.setStyleSheet(self._button_style(base, hover))
            btn.setEnabled(True)
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(slot)
        self._update_repropagate_hint()

    def _begin_propagation_ui(self, cancel_button):
        """伝播中は対象ボタンをキャンセルに変え、他は無効化する"""
        for btn, _label, _base, _hover, _slot in self._propagation_buttons():
            if btn is cancel_button:
                btn.setText("■ キャンセル")
                btn.setStyleSheet(self.CANCEL_STYLE)
                btn.setEnabled(True)
                try:
                    btn.clicked.disconnect()
                except TypeError:
                    pass
                btn.clicked.connect(self._cancel_propagation)
            else:
                btn.setEnabled(False)

    def _update_repropagate_hint(self):
        """ポイントの変更が未反映なら「ポイントを反映」ボタンを目立たせる"""
        if self._points_dirty:
            self.btn_repropagate.setText("⟲ ポイントを反映（未反映）")
            self.btn_repropagate.setStyleSheet(
                self._button_style("#d07a10", "#e89a30"))
        else:
            self.btn_repropagate.setText("⟲ ポイントを反映")
            self.btn_repropagate.setStyleSheet(
                self._button_style("#7a3fa0", "#9a5ac0"))

    def _update_points_dirty(self):
        """ポイントの変更が他フレームへ未反映かを判定して UI に出す

        伝播済みの他フレームがあるのにポイントを足した場合、
        そのフレームのマスクしか更新されない。再伝播が必要なことを示す。
        """
        frame_idx = self.state.current_frame
        self._points_dirty = any(
            any(f != frame_idx for f in frames)
            for frames in self.all_masks.values()
        )
        self._update_repropagate_hint()

    def _start_propagation(self, cancel_button, thread, expected, message):
        """伝播スレッドを開始し、進捗表示とボタン状態を整える"""
        self._prop_done = 0
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, expected))
        self.progress_bar.setValue(0)
        self._begin_propagation_ui(cancel_button)
        self.statusBar().showMessage(message)

        self._prop_thread = thread
        thread.progress.connect(self._on_propagation_progress)
        thread.time_info.connect(
            lambda msg: self.statusBar().showMessage(f"マスク伝播中... {msg}")
        )
        thread.finished_signal.connect(self._on_propagation_done)
        thread.start()

    def _anchor_frame_for_repropagation(self):
        """再伝播の起点にするフレームを選ぶ

        現在のフレームにポイントがあればそこを使う（直前に打った修正点を
        起点にしたいため）。無ければ最も近いポイントのあるフレーム。
        """
        prompted = sorted({
            f for track in self.state.person_tracks
            for f, pts in track.points.items() if pts
        })
        if not prompted:
            return None
        current = self.state.current_frame
        if current in prompted:
            return current
        return min(prompted, key=lambda f: abs(f - current))

    def _repropagate_from_points(self):
        """追加したポイントを動画全体へ反映し直す

        起点フレームから前方へ伝播したあと、逆方向にも伝播して
        起点より手前のフレームも更新する。小物が抜けたフレームで
        ポイントを 1 つ足してからこれを押せば、
        フレームごとにブラシで塗り直さなくても全フレームへ届く。
        """
        if not self.sam2.model_loaded:
            QMessageBox.warning(self, "エラー", "SAM2モデルがロードされていません。")
            return
        if not self.state.person_tracks:
            QMessageBox.warning(self, "エラー", "人物トラックがありません。")
            return

        anchor = self._anchor_frame_for_repropagation()
        if anchor is None:
            QMessageBox.warning(
                self, "エラー",
                "ポイントプロンプトがありません。\n"
                "含めたい対象をクリックしてポイントを追加してください。"
            )
            return

        if not self._check_ready_to_propagate(self.state.total_frames):
            return

        self.statusBar().showMessage("伝播準備中... (inference_state 再構築)")
        QApplication.processEvents()
        if not self._reinit_and_reapply_points():
            QMessageBox.warning(self, "エラー", "SAM2 の再初期化に失敗しました。")
            return

        total = self.state.total_frames
        self._push_undo("ポイントを反映して再伝播",
                        mask_slots=self._propagation_slots(0))

        # 起点から前方へ、続いて後方へ。2 パスで動画全体をカバーする
        passes = [(anchor, False)]
        expected = max(1, total - anchor)
        if anchor > 0:
            passes.append((anchor, True))
            expected += anchor

        self._start_propagation(
            self.btn_repropagate,
            PropagationThread(self.sam2, total, start_frame=0,
                              passes=passes, expected_frames=expected),
            expected,
            f"ポイントを反映中... (フレーム {anchor} を起点に前方・後方へ)"
        )

    def _check_ready_to_propagate(self, num_frames):
        """伝播を始めてよいか確認する（CUDA の状態とメモリ残量）"""
        if self.sam2.cuda_broken:
            self._warn_cuda_broken()
            return False

        need = self.sam2.estimate_state_mb(num_frames)
        avail = _available_mb()
        if avail > 0 and need > avail * 0.8:
            reply = QMessageBox.warning(
                self, "メモリが不足するおそれがあります",
                f"この伝播には約 {need / 1024:.1f} GB のメモリが必要と見込まれます。\n"
                f"現在の空きは {avail / 1024:.1f} GB です。\n\n"
                f"このまま実行するとメモリを使い切って落ちる可能性があります。\n"
                f"「編集 → メモリを解放」を試すか、\n"
                f"「ファイル → 分割して処理」で区間ごとに処理してください。\n\n"
                f"それでも実行しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
        return True

    def _warn_cuda_broken(self):
        """CUDA が壊れたことを一度だけ知らせる"""
        if getattr(self, "_cuda_warned", False):
            return
        self._cuda_warned = True
        QMessageBox.critical(
            self, "GPU が使用できなくなりました",
            "CUDA が回復不能な状態になりました。\n"
            "（多くはメモリ不足が原因です）\n\n"
            "これ以降 GPU を使う処理は実行できません。\n"
            "作業内容を保存してから、アプリを再起動してください。\n\n"
            "「ファイル → プロジェクト保存」で現在のマスクを保存できます。"
        )
        self.statusBar().showMessage(
            "GPU が使用できません — 保存してアプリを再起動してください"
        )

    def _propagation_slots(self, start_frame):
        """伝播で上書きされうる (track_id, frame_idx) を列挙する。

        未使用スロットは None として記録されるだけなのでコストは無視できる。
        既存マスクがある場合もビットパックして持つため、
        FHD・900 フレーム・1 人でおよそ 58MB。UndoStack 側の
        MAX_BYTES を超えた分は古い履歴から自動的に捨てられる。
        """
        return [
            (track.track_id, f)
            for track in self.state.person_tracks
            for f in range(start_frame, self.state.total_frames)
        ]

    def _propagate_masks(self):
        """SAM2でマスクを全フレームに伝播"""
        if not self.sam2.model_loaded:
            QMessageBox.warning(self, "エラー", "SAM2モデルがロードされていません。")
            return

        if not self.state.person_tracks:
            QMessageBox.warning(self, "エラー", "人物トラックがありません。")
            return

        has_prompts = any(
            bool(track.points) for track in self.state.person_tracks
        )
        if not has_prompts:
            QMessageBox.warning(
                self, "エラー",
                "ポイントプロンプトがありません。\n"
                "人物をクリックしてからマスク伝播を実行してください。"
            )
            return

        if not self._check_ready_to_propagate(self.state.total_frames):
            return

        self.statusBar().showMessage("伝播準備中... (inference_state 再構築)")
        QApplication.processEvents()
        if not self._reinit_and_reapply_points():
            QMessageBox.warning(self, "エラー", "SAM2 の再初期化に失敗しました。")
            return

        self._push_undo("全フレーム伝播", mask_slots=self._propagation_slots(0))

        self._start_propagation(
            self.btn_propagate,
            PropagationThread(self.sam2, self.state.total_frames),
            self.state.total_frames,
            "マスク伝播中... (キャンセルするにはボタンを押してください)"
        )

    def _cancel_propagation(self):
        """伝播をキャンセル"""
        if hasattr(self, '_prop_thread') and self._prop_thread.isRunning():
            self._prop_thread.cancel()
            self.statusBar().showMessage("キャンセル中... 現在のフレーム処理完了後に停止します")

    def _propagate_from_current(self):
        """現在のフレームから先のみマスク伝播"""
        if not self.sam2.model_loaded:
            QMessageBox.warning(self, "エラー", "SAM2モデルがロードされていません。")
            return

        if not self.state.person_tracks:
            QMessageBox.warning(self, "エラー", "人物トラックがありません。")
            return

        has_prompts = any(
            bool(track.points) for track in self.state.person_tracks
        )
        if not has_prompts:
            QMessageBox.warning(
                self, "エラー",
                "ポイントプロンプトがありません。\n"
                "人物をクリックしてからマスク伝播を実行してください。"
            )
            return

        start_frame = self.state.current_frame
        if not self._check_ready_to_propagate(self.state.total_frames - start_frame):
            return

        self.statusBar().showMessage("伝播準備中... (inference_state 再構築)")
        QApplication.processEvents()
        if not self._reinit_and_reapply_points():
            QMessageBox.warning(self, "エラー", "SAM2 の再初期化に失敗しました。")
            return

        self._push_undo("ここから伝播",
                        mask_slots=self._propagation_slots(start_frame))

        # 開始フレームにプロンプトがあれば、そこを推論の起点にできる。
        # 手前のフレームを推論し直さずに済むぶん速い。
        # プロンプトが無いフレームを起点にすると追跡の根拠が無くなるため、
        # その場合は従来どおり最初のプロンプトから流して手前を捨てる。
        has_prompt_here = any(
            track.points.get(start_frame) for track in self.state.person_tracks
        )
        anchor = start_frame if has_prompt_here else None
        expected = max(1, self.state.total_frames - start_frame)

        self._start_propagation(
            self.btn_propagate_from,
            PropagationThread(
                self.sam2, self.state.total_frames, start_frame=start_frame,
                passes=[(anchor, False)], expected_frames=expected
            ),
            expected,
            f"フレーム {start_frame} から伝播中..."
        )

    def _propagate_one_frame(self):
        """現在のフレームのマスクを次の1フレームに適用"""
        if not self.sam2.model_loaded:
            QMessageBox.warning(self, "エラー", "SAM2モデルがロードされていません。")
            return

        if not self.state.person_tracks:
            QMessageBox.warning(self, "エラー", "人物トラックがありません。")
            return

        current = self.state.current_frame
        next_frame = current + 1
        if next_frame >= self.state.total_frames:
            self.statusBar().showMessage("最終フレームです")
            return

        self._push_undo(
            "1フレーム伝播",
            mask_slots=[(t.track_id, next_frame)
                        for t in self.state.person_tracks],
        )

        self.statusBar().showMessage(f"フレーム {current} → {next_frame} に伝播中...")
        QApplication.processEvents()

        try:
            applied = False
            for track in self.state.person_tracks:
                tid = track.track_id

                # 現在のフレームにマスクがあるか確認
                current_mask = None
                if tid in self.all_masks and current in self.all_masks[tid]:
                    current_mask = self.all_masks[tid][current]
                elif current in track.masks:
                    current_mask = track.masks[current]

                if current_mask is None:
                    continue

                # 現在のマスクをプロンプトとして次フレームに登録
                import torch as _torch
                mask_tensor = _torch.from_numpy(
                    current_mask.astype(np.float32)
                ).to(self.sam2.device)

                try:
                    # add_new_mask で前フレームのマスクを次フレームのプロンプトとして使用
                    _, out_obj_ids, out_mask_logits = self.sam2.predictor.add_new_mask(
                        inference_state=self.sam2.inference_state,
                        frame_idx=self.sam2.to_local_frame(next_frame),
                        obj_id=tid,
                        mask=mask_tensor,
                    )
                    logits = self.sam2._select_obj_mask(
                        tid, out_obj_ids, out_mask_logits)
                    if logits is None:
                        continue
                    new_mask = (logits > 0.0).cpu().numpy().squeeze()
                    new_mask = _downscale_mask_for_storage(new_mask)
                except (AttributeError, TypeError):
                    # add_new_mask が無い場合、現在のポイントを次フレームにも適用
                    if current in track.points and track.points[current]:
                        pts = [(p[0], p[1]) for p in track.points[current]]
                        lbls = [p[2] for p in track.points[current]]
                        new_mask = self.sam2.add_points(
                            next_frame, tid, pts, lbls
                        )
                    else:
                        # ポイントもマスクAPIも無ければ、マスクをそのままコピー
                        new_mask = current_mask.copy()

                if new_mask is not None:
                    if tid not in self.all_masks:
                        self.all_masks[tid] = {}
                    self.all_masks[tid][next_frame] = new_mask
                    track.masks[next_frame] = new_mask
                    applied = True

            if applied:
                self.timeline.go_to_frame(next_frame)
                self._refresh_display()
                self.statusBar().showMessage(
                    f"1フレーム伝播完了: フレーム {current} → {next_frame}"
                )
            else:
                self.statusBar().showMessage(
                    "現在のフレームにマスクがありません。先に人物を選択してください。"
                )
        except Exception as e:
            logger.error(f"1フレーム伝播エラー: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"1フレーム伝播エラー: {e}")

    def _on_propagation_progress(self, frame_idx, expected, masks):
        # 逆方向のパスではフレーム番号が減っていくため、
        # バーは処理済み件数で進める
        self._prop_done += 1
        self.progress_bar.setValue(self._prop_done)
        for obj_id, mask in masks.items():
            if obj_id not in self.all_masks:
                self.all_masks[obj_id] = {}
            self.all_masks[obj_id][frame_idx] = mask

            # トラックにも保存
            for track in self.state.person_tracks:
                if track.track_id == obj_id:
                    track.masks[frame_idx] = mask
                    break

    def _on_propagation_done(self, success):
        self.progress_bar.setVisible(False)

        if success:
            # 伝播が通ったのでポイントは全フレームへ反映済み
            self._points_dirty = False
        self._restore_propagation_buttons()

        if self.sam2.cuda_broken:
            self._warn_cuda_broken()

        # 処理済みフレーム数を集計
        processed_frames = 0
        for tid in self.all_masks:
            processed_frames = max(processed_frames, len(self.all_masks[tid]))

        if success:
            self.statusBar().showMessage(
                f"マスク伝播完了 — {processed_frames}フレーム処理済み"
            )
        else:
            self.statusBar().showMessage(
                f"マスク伝播中断 — {processed_frames}フレームまで処理済み"
            )

        # 伝播中に確保された中間バッファを解放する
        QTimer.singleShot(0, self._drop_finished_threads)
        self._release_memory(label="伝播後の解放")
        self._refresh_display()

    def _on_frame_changed(self, frame_idx):
        """フレームが変更された時"""
        self.state.current_frame = frame_idx
        self._refresh_display()

    def _on_blur_slider_changed(self, value):
        self.state.blur_strength = value
        self.blur_spin.blockSignals(True)
        self.blur_spin.setValue(value)
        self.blur_spin.blockSignals(False)
        if self.chk_preview.isChecked():
            self._refresh_display()

    def _on_blur_spin_changed(self, value):
        self.state.blur_strength = value
        self.blur_slider.blockSignals(True)
        self.blur_slider.setValue(value)
        self.blur_slider.blockSignals(False)
        if self.chk_preview.isChecked():
            self._refresh_display()

    def _on_blur_type_changed(self, index):
        types = ["gaussian", "box", "motion"]
        self.state.blur_type = types[index]
        if self.chk_preview.isChecked():
            self._refresh_display()

    def _on_feather_slider_changed(self, value):
        self.state.edge_feather = value
        self.feather_spin.blockSignals(True)
        self.feather_spin.setValue(value)
        self.feather_spin.blockSignals(False)
        if self.chk_preview.isChecked():
            self._refresh_display()

    def _on_feather_spin_changed(self, value):
        self.state.edge_feather = value
        self.feather_slider.blockSignals(True)
        self.feather_slider.setValue(value)
        self.feather_slider.blockSignals(False)
        if self.chk_preview.isChecked():
            self._refresh_display()

    def _refresh_display(self, *args):
        """現在のフレームを再描画"""
        frame = self.video.get_frame(self.state.current_frame)
        if frame is None:
            return

        h, w = frame.shape[:2]

        # 合成マスク作成（保持解像度が違っても _to_size で表示解像度へ揃える）
        frame_idx = self.state.current_frame
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for track in self.state.person_tracks:
            m = self.all_masks.get(track.track_id, {}).get(frame_idx)
            if m is not None:
                combined_mask = np.maximum(combined_mask, _to_size(m, w, h))

        # 手動編集
        edit = self.state.manual_edits.get(frame_idx)
        if edit is not None:
            combined_mask = np.maximum(combined_mask, _to_size(edit, w, h))

        # プレビュー用にぼかし適用 or ブラックアウト
        if combined_mask.any():
            if self.chk_blackout.isChecked():
                # ブラックアウト表示（人物以外を黒に）
                display_frame = frame.copy()
                mask_3ch = np.stack([combined_mask] * 3, axis=-1).astype(np.float32)
                if mask_3ch.max() > 0:
                    mask_3ch = mask_3ch / mask_3ch.max()
                display_frame = (display_frame * mask_3ch).astype(np.uint8)
            elif self.chk_preview.isChecked():
                display_frame = self._apply_blur_preview(frame, combined_mask)
            else:
                display_frame = frame.copy()
        else:
            display_frame = frame.copy()

        # QPixmapに変換
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            display_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)

        # マスクオーバーレイ
        overlay = None
        if self.chk_overlay.isChecked() and combined_mask.any():
            overlay_img = np.zeros((h, w, 4), dtype=np.uint8)

            for track in self.state.person_tracks:
                m = self.all_masks.get(track.track_id, {}).get(frame_idx)
                if m is not None:
                    color = track.color
                    overlay_img[_to_size(m, w, h) > 0] = [
                        color.red(), color.green(), color.blue(), 120
                    ]

            # ポイントを描画
            for track in self.state.person_tracks:
                if self.state.current_frame in track.points:
                    for px, py, label in track.points[self.state.current_frame]:
                        c = (0, 255, 0, 255) if label == 1 else (255, 0, 0, 255)
                        cv2.circle(overlay_img, (int(px), int(py)), 6, c, -1)
                        cv2.circle(overlay_img, (int(px), int(py)), 7,
                                   (255, 255, 255, 255), 2)

            qimg_overlay = QImage(
                overlay_img.data, w, h, w * 4, QImage.Format.Format_RGBA8888
            )
            overlay = QPixmap.fromImage(qimg_overlay)

        self.viewport.set_frame(pixmap, overlay)

    def _apply_blur_preview(self, frame, person_mask):
        """プレビュー用のぼかし適用（person_mask は frame と同サイズの 0/1）"""
        mask_float = person_mask.astype(np.float32)
        if self.state.edge_feather > 0:
            fk = self.state.edge_feather * 2 + 1
            mask_float = cv2.GaussianBlur(mask_float, (fk, fk), 0)

        peak = float(mask_float.max())
        if peak > 0:
            mask_float /= peak

        return blend_blur(frame, mask_float, self.state.blur_type,
                          self.state.blur_strength, blender=self._blender)

    def _export_video(self):
        """ぼかし適用済みビデオをエクスポート"""
        if not self.state.video_path:
            QMessageBox.warning(self, "エラー", "動画が読み込まれていません。")
            return

        # 既定は output/ 配下の「元ファイル名_2」
        path, _ = QFileDialog.getSaveFileName(
            self, "エクスポート先を選択", self._default_output_path(),
            "MP4 (*.mp4);;AVI (*.avi);;すべて (*)"
        )
        if not path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.state.total_frames)

        self._export_thread = ExportThread(
            self.video, self.state, self.all_masks, path, blender=self._blender
        )
        self._export_thread.progress.connect(
            lambda i, t: self.progress_bar.setValue(i)
        )
        self._export_thread.finished_signal.connect(self._on_export_done)
        self._export_thread.start()
        self.statusBar().showMessage("エクスポート中...")

    def _on_export_done(self, success, result):
        self.progress_bar.setVisible(False)
        # スレッド自身のシグナル処理中に破棄すると危ないので次のループで片付ける
        QTimer.singleShot(0, self._cleanup_after_export)
        if success:
            QMessageBox.information(
                self, "完了",
                f"エクスポートが完了しました:\n{result}"
            )
            self.statusBar().showMessage(f"エクスポート完了: {result}")
        else:
            QMessageBox.critical(self, "エラー", f"エクスポートに失敗: {result}")

    def _cleanup_after_export(self):
        """エクスポート後の後始末（スレッドと一時バッファの解放）"""
        self._drop_finished_threads()
        freed = self._release_memory(label="エクスポート後の解放")
        if freed >= 1.0:
            self.statusBar().showMessage(
                f"{self.statusBar().currentMessage()}　"
                f"（メモリ {freed:.0f} MB 解放）"
            )

    def _export_split_video(self):
        """30秒ごとに分割してエクスポート、最後に結合"""
        if not self.state.video_path:
            QMessageBox.warning(self, "エラー", "動画が読み込まれていません。")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "分割エクスポート先を選択", self._default_output_path(),
            "MP4 (*.mp4);;AVI (*.avi);;すべて (*)"
        )
        if not output_path:
            return

        segment_sec = 30
        fps = self.state.fps
        total = self.state.total_frames
        segment_frames = int(segment_sec * fps)
        num_segments = (total + segment_frames - 1) // segment_frames

        self.statusBar().showMessage(
            f"分割エクスポート: {num_segments}セグメント × {segment_sec}秒"
        )

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="vbs_split_")
        part_files = []

        # 書き出しは元解像度。マスクは作業解像度のまま組み立てて拡大する。
        out_w = self.state.source_width or self.state.width
        out_h = self.state.source_height or self.state.height
        work_w, work_h = self.state.width, self.state.height
        scale = self.state.proxy_scale or 1.0
        strength = max(1, int(round(self.state.blur_strength / scale)))

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)

        # セグメントをまたいでフレーム番号は昇順なので、
        # 元解像度は 1 本のキャプチャで順次読みできる
        reader = self.video.open_source_reader()
        if reader is None:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "エラー", "元動画を開けませんでした。")
            return
        blend_buf = None

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            for seg_idx in range(num_segments):
                start = seg_idx * segment_frames
                end = min(start + segment_frames, total)

                part_path = os.path.join(tmp_dir, f"part_{seg_idx:04d}.mp4")
                part_files.append(part_path)

                out = cv2.VideoWriter(part_path, fourcc, fps, (out_w, out_h))

                for i in range(start, end):
                    ret, frame = reader.read()
                    if not ret:
                        break

                    # 拡大は blend_blur 側（GPU が使えれば GPU 上）で行う
                    mask_float = build_mask_float(
                        self.state, self.all_masks, i, work_w, work_h
                    )

                    if blend_buf is None or blend_buf.shape != frame.shape:
                        blend_buf = np.empty(frame.shape, dtype=np.float32)
                    out.write(blend_blur(
                        frame, mask_float, self.state.blur_type,
                        strength, blend_buf, self._blender
                    ))
                    self.progress_bar.setValue(i)
                    QApplication.processEvents()

                out.release()
                self.statusBar().showMessage(
                    f"セグメント {seg_idx + 1}/{num_segments} 完了"
                )

            # 結合
            self.statusBar().showMessage("セグメントを結合中...")
            QApplication.processEvents()

            # OpenCVで結合
            final_out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            for part_path in part_files:
                cap = cv2.VideoCapture(part_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    final_out.write(frame)
                cap.release()
            final_out.release()

            self.progress_bar.setVisible(False)
            QMessageBox.information(
                self, "完了",
                f"分割エクスポートが完了しました:\n{output_path}\n\n"
                f"{num_segments}セグメントを結合"
            )
            self.statusBar().showMessage(f"エクスポート完了: {output_path}")

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "エラー", f"分割エクスポートに失敗: {e}")
            import traceback
            traceback.print_exc()
        finally:
            reader.release()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._release_memory()

    def _save_project(self):
        """プロジェクト状態を保存"""
        path, _ = QFileDialog.getSaveFileName(
            self, "プロジェクトを保存",
            self._default_output_path(suffix="", ext=".vbsproj"),
            "VBS プロジェクト (*.vbsproj);;すべて (*)"
        )
        if not path:
            return

        try:
            data = {
                "video_path": self.state.video_path,
                "blur_strength": self.state.blur_strength,
                "blur_type": self.state.blur_type,
                "edge_feather": self.state.edge_feather,
                "width": self.state.width,
                "height": self.state.height,
                "source_width": self.state.source_width,
                "source_height": self.state.source_height,
                "proxy_scale": self.state.proxy_scale,
                "tracks": [],
            }
            for track in self.state.person_tracks:
                t = {
                    "track_id": track.track_id,
                    "name": track.name,
                    "color": [track.color.red(), track.color.green(),
                              track.color.blue()],
                    "points": {
                        str(k): v for k, v in track.points.items()
                    },
                }
                data["tracks"].append(t)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # マスク本体を .npz サイドカーに保存（bool化で軽量化）
            mask_arrays = {}
            for tid, frames in self.all_masks.items():
                for f_idx, m in frames.items():
                    if m is None:
                        continue
                    mask_arrays[f"track_{tid}_frame_{f_idx}"] = m.astype(bool)
            for f_idx, m in self.state.manual_edits.items():
                if m is None:
                    continue
                mask_arrays[f"manual_frame_{f_idx}"] = m.astype(bool)
            np.savez_compressed(path + ".npz", **mask_arrays)

            self.statusBar().showMessage(f"プロジェクト保存完了: {path}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"保存に失敗: {e}")

    def _load_project(self):
        """プロジェクト状態を読み込み"""
        path, _ = QFileDialog.getOpenFileName(
            self, "プロジェクトを読み込む", self._output_dir,
            "VBS プロジェクト (*.vbsproj);;すべて (*)"
        )
        if not path:
            return
        self._load_project_file(path)

    def _load_project_file(self, path):
        """.vbsproj（+ .npz サイドカー）から状態を復元する

        通常のプロジェクト読込と自動保存からの復帰で共通に使う。
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 動画を開く（前の動画ぶんを先に手放してから）
            if data.get("video_path"):
                video_path = data["video_path"]
                self._release_video_resources(label="前の動画の解放")
                info = self.video.open_video(video_path)
                if info:
                    self.state.video_path = video_path
                    self.state.fps = info["fps"]
                    self.state.width = info["width"]
                    self.state.height = info["height"]
                    self.state.source_width = info["source_width"]
                    self.state.source_height = info["source_height"]
                    self.state.proxy_scale = info["proxy_scale"]
                    self.state.total_frames = info["total_frames"]
                    self.timeline.setup(info["total_frames"], info["fps"])
                    self.setWindowTitle(f"Video Blur Studio — {Path(video_path).name}")
                else:
                    QMessageBox.warning(
                        self, "警告",
                        f"動画ファイルが見つかりません:\n{video_path}"
                    )

            self.state.blur_strength = data.get("blur_strength", 25)
            self.state.blur_type = data.get("blur_type", "gaussian")
            self.state.edge_feather = data.get("edge_feather", 5)

            self.blur_slider.setValue(self.state.blur_strength)
            self.blur_spin.setValue(self.state.blur_strength)
            self.feather_slider.setValue(self.state.edge_feather)
            self.feather_spin.setValue(self.state.edge_feather)

            # トラック復元
            self.state.person_tracks.clear()
            self.person_list.clear()
            self.all_masks.clear()

            for t in data.get("tracks", []):
                color = QColor(*t["color"])
                track = PersonTrack(
                    track_id=t["track_id"],
                    name=t["name"],
                    color=color,
                    points={int(k): v for k, v in t.get("points", {}).items()},
                )
                self.state.person_tracks.append(track)
                self.all_masks[track.track_id] = {}

                item = QListWidgetItem(f"● {track.name}")
                item.setForeground(QBrush(color))
                self.person_list.addItem(item)

            # マスク本体を .npz サイドカーから復元
            self.state.manual_edits.clear()
            npz_path = path + ".npz"
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path) as npz:
                        for key in npz.files:
                            arr = npz[key]
                            if key.startswith("track_"):
                                # track_{tid}_frame_{f_idx}
                                _, tid_s, _, f_s = key.split("_")
                                tid, f_idx = int(tid_s), int(f_s)
                                if tid not in self.all_masks:
                                    self.all_masks[tid] = {}
                                self.all_masks[tid][f_idx] = arr
                                for tr in self.state.person_tracks:
                                    if tr.track_id == tid:
                                        tr.masks[f_idx] = arr
                                        break
                            elif key.startswith("manual_"):
                                f_idx = int(key.split("_")[2])
                                self.state.manual_edits[f_idx] = arr.astype(np.uint8)
                    logger.info(f"マスクを復元: {npz_path}")
                except Exception as e:
                    logger.warning(f"マスク復元失敗: {e}")

            self._on_frame_changed(0)

            # 読み込んだ内容は履歴の連続性が無いので破棄する
            self._reset_history()

            # フレーム抽出 + SAM2初期化 + ポイント再登録
            if self.state.video_path:
                self._frames_dir = self._frames_dir_for(self.state.video_path)
                missing = self._count_missing_frames(
                    self._frames_dir, self.state.total_frames
                )
                if missing > 0:
                    # 抽出は VideoCapture を共有するので、必ず完了まで待つ。
                    # 待たずに戻ると、後続処理が cap を release した瞬間に
                    # デコードスレッドが落ちる。
                    self._extract_frames()
                    self._wait_for_thread(
                        self._extract_thread,
                        f"フレーム抽出中... ({missing} フレーム)"
                    )
                if self.sam2.model_loaded and \
                        self._count_missing_frames(
                            self._frames_dir, self.state.total_frames) <= 0:
                    if self.sam2.init_video(self._frames_dir):
                        self._reapply_saved_points()

            self.statusBar().showMessage(f"プロジェクト読込完了: {path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"読込に失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

 

    AUTOSAVE_MAX_AGE_DAYS = 7      # これより古い自動保存は提示しない

    def _find_autosave_for(self, video_path):
        """指定の動画に対応する最新の自動保存ファイルを返す（無ければ None）"""
        import glob, datetime

        if not video_path:
            return None

        stem = Path(video_path).stem
        pattern = os.path.join(self._autosave_dir, f"{stem}_*.vbsproj")
        candidates = sorted(glob.glob(pattern), reverse=True)
        if not candidates:
            return None

        max_age = self.AUTOSAVE_MAX_AGE_DAYS * 86400
        now = datetime.datetime.now().timestamp()

        for candidate in candidates:
            try:
                if now - os.path.getmtime(candidate) > max_age:
                    break          # 降順なので、これ以降は全て古い
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                logger.warning(f"自動保存の読み取りに失敗: {candidate} ({e})")
                continue

            # ファイル名の前方一致だけだと別の動画を拾いうるので中身で確認する
            if os.path.normpath(data.get("video_path", "")) != \
                    os.path.normpath(video_path):
                continue
            return candidate, data

        return None

    def _offer_auto_saved(self, video_path):
        """動画を開いた直後に、対応する自動保存があれば復帰を提案する

        受け入れられた場合は _load_project_file で復元して True を返す。
        （元の実装は呼び出し元が無いうえ、str に .stem を呼んでおり、
          復元してもポイントだけでマスクは戻らなかった）
        """
        import datetime

        found = self._find_autosave_for(video_path)
        if not found:
            return False
        latest_file, data = found

        saved_at = os.path.getmtime(latest_file)
        age = datetime.datetime.now().timestamp() - saved_at
        if age < 3600:
            age_desc = f"{int(age // 60)} 分前"
        elif age < 86400:
            age_desc = f"{int(age // 3600)} 時間前"
        else:
            age_desc = f"{int(age // 86400)} 日前"

        track_count = len(data.get("tracks", []))
        reply = QMessageBox.question(
            self, "自動保存を読み込みますか",
            f"この動画の作業中データが見つかりました:\n\n"
            f"  {Path(latest_file).name}\n"
            f"  保存日時: {data.get('timestamp', '不明')}（{age_desc}）\n"
            f"  人物トラック: {track_count} 件\n\n"
            f"読み込みますか？\n"
            f"（「いいえ」を選ぶと、この動画を新規に開きます）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        if self._load_project_file(latest_file):
            self.statusBar().showMessage(
                f"自動保存から復帰: {Path(latest_file).name}"
            )
            logger.info(f"自動保存から復帰: {latest_file}")
            return True
        return False

    def _reapply_saved_points(self):
        """保存されたポイントをSAM2に再登録してマスクを復元"""
        if not self.sam2.model_loaded or self.sam2.inference_state is None:
            logger.warning("SAM2が初期化されていないため、ポイント再登録をスキップ")
            return

        logger.info("保存されたポイントを再登録中...")
        for track in self.state.person_tracks:
            for frame_idx, pts_list in track.points.items():
                all_pts = [(p[0], p[1]) for p in pts_list]
                all_labels = [p[2] for p in pts_list]

                mask = self.sam2.add_points(
                    frame_idx, track.track_id, all_pts, all_labels
                )
                if mask is not None:
                    track.masks[frame_idx] = mask
                    self.all_masks[track.track_id][frame_idx] = mask
                    logger.info(
                        f"  {track.name}: フレーム {frame_idx} に "
                        f"{len(all_pts)} ポイント再登録"
                    )

        self._refresh_display()
        logger.info("ポイント再登録完了")

    def _show_refine_help(self):
        """小物が抜けたときの直し方を案内する"""
        QMessageBox.information(
            self, "小物がマスクから外れるとき",
            "バッグなどの小物が伝播後にマスクから外れる場合の手順です。\n\n"
            "1. 小物が外れているフレームへ移動する\n"
            "2. ツールを「人物選択」(V) にして、小物の上をクリックする\n"
            "   → そのフレームのマスクに小物が加わります\n"
            "3. ツールバーの「⟲ ポイントを反映」を押す\n"
            "   → 打ったポイントを起点に前方・後方の両方へ伝播し直し、\n"
            "     全フレームに小物が含まれるようになります\n\n"
            "余計な部分まで含まれてしまう場合は、\n"
            "「除外ポイント」(X) でその部分をクリックしてから\n"
            "同じく「⟲ ポイントを反映」を押してください。\n\n"
            "ブラシ (B) は 1 フレームだけを直す用途です。\n"
            "複数フレームに効かせたいときはポイントを使ってください。"
        )

    def _show_about(self):
        QMessageBox.about(
            self, "Video Blur Studio",
            "<h2>Video Blur Studio</h2>"
            "<p>SAM2ベースの人物セグメンテーション &amp; 背景ぼかしツール</p>"
            "<p>ロトブラシ3.0同等の精度を目指した本格的なビデオ編集ツール</p>"
            "<hr>"
            "<p><b>主な機能:</b></p>"
            "<ul>"
            "<li>SAM2 (Segment Anything Model 2) による高精度セグメンテーション</li>"
            "<li>クリック1つで人物を選択 → 全フレーム自動追跡</li>"
            "<li>手動でマスク修正（ブラシ / 消しゴム）</li>"
            "<li>ぼかし強度・タイプ・エッジフェザリングの細かい調整</li>"
            "<li>リアルタイムプレビュー</li>"
            "</ul>"
            "<hr>"
            "<p>Powered by Meta SAM 2.1 + PyQt6</p>"
        )

    def _save_current_state(self):
        """現在までの作業状況をプロジェクト保存ファイルに保存（Ctrl+S / 手動）"""
        self._do_save(self._autosave_dir)

    def _auto_save(self):
        """2分毎のタイマーから呼ばれる自動保存"""
        if not self.state.video_path:
            return
        self._do_save(self._autosave_dir)

    def _do_save(self, save_dir):
        """save_dir に .vbsproj + .npz を書き出す"""
        import datetime

        os.makedirs(save_dir, exist_ok=True)

        if self.state.video_path:
            video_name = Path(self.state.video_path).stem
        else:
            video_name = "unknown"

        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{video_name}_{timestamp_str}.vbsproj"
        filepath = os.path.join(save_dir, filename)

        try:
            data = {
                "version": 1,
                "video_path": self.state.video_path,
                "timestamp": timestamp.isoformat(),
                "current_frame": self.state.current_frame,
                "blur_strength": self.state.blur_strength,
                "blur_type": self.state.blur_type,
                "edge_feather": self.state.edge_feather,
                "total_frames": self.state.total_frames,
                "fps": self.state.fps,
                "width": self.state.width,
                "height": self.state.height,
                "source_width": self.state.source_width,
                "source_height": self.state.source_height,
                "proxy_scale": self.state.proxy_scale,
                "tracks": [],
            }

            for track in self.state.person_tracks:
                t = {
                    "track_id": track.track_id,
                    "name": track.name,
                    "color": [track.color.red(), track.color.green(), track.color.blue()],
                    "points": {str(k): v for k, v in track.points.items()},
                }
                data["tracks"].append(t)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            mask_arrays = {}
            for tid, frames in self.all_masks.items():
                for f_idx, m in frames.items():
                    if m is None:
                        continue
                    mask_arrays[f"track_{tid}_frame_{f_idx}"] = m.astype(bool)
            for f_idx, m in self.state.manual_edits.items():
                if m is None:
                    continue
                mask_arrays[f"manual_frame_{f_idx}"] = m.astype(bool)
            if mask_arrays:
                np.savez_compressed(filepath + ".npz", **mask_arrays)

            self.statusBar().showMessage(f"自動保存：{filename}")
            logger.info(f"自動保存：{filepath}")

        except Exception as e:
            logger.error(f"自動保存に失敗：{e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        """ウィンドウ閉じ時の処理"""
        self._autosave_timer.stop()
        if self._blender is not None:
            self._blender.close()

        if self.state.auto_save_enabled and self.state.video_path:
            self._do_save(self._autosave_dir)

        cuda_broken = self.sam2.cuda_broken
        if not cuda_broken:
            self.sam2.unload_model()
        self._clear_project_data()
        self._drop_finished_threads()
        self.video.close()

        if cuda_broken:
            # CUDA が壊れていると、残った GPU テンソルのデストラクタが
            # 例外を投げて terminate され、保存内容ごと異常終了しかねない。
            # 保存は済ませてあるので、後始末を待たずにそのまま抜ける。
            logger.warning("CUDA 異常のため、後始末を行わず終了します")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        def finish_app():
            QApplication.quit()

        QTimer.singleShot(3000, finish_app)
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ══════════════════════════════════════════════════════════════════════════════

def _is_wsl():
    """WSL (WSLg) 上で動いているか"""
    if os.path.exists("/mnt/wslg"):
        return True
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def select_qt_platform():
    """WSLg では Qt のプラットフォームプラグインに xcb を選ぶ

    WSLg の Wayland 上では Qt6 が grabbing popup を作れず
    （qt.qpa.wayland: Failed to create grabbing popup）、
    QComboBox のドロップダウンが閉じずにメインウィンドウの裏へ
    残り続ける。同じ環境でも XWayland 経由の xcb では
    ポップアップが正しく開閉するため、そちらを既定にする。

    QApplication を作る前に呼ぶこと。
    QT_QPA_PLATFORM が明示されている場合はユーザーの指定を尊重する。
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if not _is_wsl():
        return
    if not os.environ.get("WAYLAND_DISPLAY"):
        return                      # そもそも Wayland ではない
    if not os.environ.get("DISPLAY"):
        return                      # X ディスプレイが無いなら切り替えられない
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    logger.info(
        "WSLg を検出したため QT_QPA_PLATFORM=xcb を使用します"
        "（Wayland ではドロップダウンが閉じない問題があるため）"
    )


def main():
    select_qt_platform()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    logger.info(f"Qt プラットフォーム: {app.platformName()}")

    # ダークパレット
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Text, QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(210, 210, 210))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 120, 212))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 96, 170))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()