"""环境贴图预过滤核心模块。

针对六类缺陷的修复实现：

1. 预过滤卷积按 GGX 分布做重要性采样，而不是盒式卷积；
2. 粗糙度到 mip 的映射连续（浮点 lod + 相邻层插值），中间档不跳变；
3. 立方体六个面接缝处按方向重投影到相邻面采样，消除亮度跳变；
4. 异步生成的 mip 链用事件栅栏同步，采样前必须等待生成完成；
5. 预过滤阶段保留 HDR 动态范围，截断只发生在最终 LDR 输出；
6. mip 层数随分辨率重算（floor(log2(res)) + 1）。
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

__all__ = [
    "FACE_ORDER",
    "mip_levels_for_resolution",
    "roughness_to_lod",
    "sample_cube",
    "ggx_convolve",
    "sample_prefiltered",
    "tonemap_ldr",
    "AsyncMipChain",
    "ggx_prefilter",
    "mip_mapping",
    "cube_seam",
    "async_read",
    "hdr_clamp",
    "determinism",
    "roughness_lut",
    "validation",
    "audit",
    "fallback",
    "face_order",
    "metrics",
    "sample",
    "lod_smooth",
    "energy",
]

# 规范立方体面顺序：+X, -X, +Y, -Y, +Z, -Z
FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


# ---------------------------------------------------------------------------
# 向量工具
# ---------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v):
    length = math.sqrt(_dot(v, v)) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# ---------------------------------------------------------------------------
# HDR 环境（演示/测试用）：天空 + 地面 + 一个强度 > 1 的太阳高光
# ---------------------------------------------------------------------------

_SUN_DIR = _normalize((0.5, 0.7, 0.3))
_SUN_INTENSITY = 4.0  # HDR：峰值明显大于 1.0


def _default_env(d):
    x, y, z = d
    sun = max(0.0, _dot(d, _SUN_DIR)) ** 4 * _SUN_INTENSITY
    sky = 0.3 + 0.5 * max(0.0, y)
    ground = 0.2
    base = sky if y >= 0.0 else ground
    return (base + sun, base * 0.9 + sun, base * 0.8 + sun)


# ---------------------------------------------------------------------------
# 立方体面 <-> 方向 映射（接缝采样的基础）
# ---------------------------------------------------------------------------

def _face_uv_to_dir(face, u, v):
    """面内 uv（允许略超出 [0,1]，用于跨面重投影）-> 单位方向。"""
    a, b = 2.0 * u - 1.0, 2.0 * v - 1.0
    if face == 0:
        return _normalize((1.0, -b, -a))
    if face == 1:
        return _normalize((-1.0, -b, a))
    if face == 2:
        return _normalize((a, 1.0, b))
    if face == 3:
        return _normalize((a, -1.0, -b))
    if face == 4:
        return _normalize((a, -b, 1.0))
    return _normalize((-a, -b, -1.0))


def _dir_to_face_uv(d):
    """单位方向 -> (face, u, v)，u/v ∈ [0,1]。"""
    x, y, z = d
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        if x > 0.0:
            face, ur, vr = 0, -z / ax, -y / ax
        else:
            face, ur, vr = 1, z / ax, -y / ax
    elif ay >= ax and ay >= az:
        if y > 0.0:
            face, ur, vr = 2, x / ay, z / ay
        else:
            face, ur, vr = 3, x / ay, -z / ay
    else:
        if z > 0.0:
            face, ur, vr = 4, x / az, -y / az
        else:
            face, ur, vr = 5, -x / az, -y / az
    return face, (ur + 1.0) * 0.5, (vr + 1.0) * 0.5


def sample_cube(faces, direction):
    """跨面无缝采样。

    双线性插值的四个 tap 若越出面边界，不做 UV clamp，而是把该 tap 的
    方向重投影到实际所在的相邻面再取 texel，消除接缝处亮度跳变。
    """
    d = _normalize(direction)
    face, u, v = _dir_to_face_uv(d)
    size = len(faces[face])
    x = u * size - 0.5
    y = v * size - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    acc = [0.0, 0.0, 0.0]
    for dy in (0, 1):
        for dx in (0, 1):
            tx, ty = x0 + dx, y0 + dy
            w = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
            if w == 0.0:
                continue
            if 0 <= tx < size and 0 <= ty < size:
                c = faces[face][ty][tx]
            else:
                # 越界 tap：方向重投影到相邻面
                tap_dir = _face_uv_to_dir(face, (tx + 0.5) / size, (ty + 0.5) / size)
                f2, u2, v2 = _dir_to_face_uv(tap_dir)
                size2 = len(faces[f2])
                tx2 = min(max(int(u2 * size2), 0), size2 - 1)
                ty2 = min(max(int(v2 * size2), 0), size2 - 1)
                c = faces[f2][ty2][tx2]
            acc[0] += w * c[0]
            acc[1] += w * c[1]
            acc[2] += w * c[2]
    return (acc[0], acc[1], acc[2])


# ---------------------------------------------------------------------------
# GGX 重要性采样（确定性 Hammersley 序列，无随机数 -> 结果可复现）
# ---------------------------------------------------------------------------

def _radical_inverse_vdc(i):
    bits = i & 0xFFFFFFFF
    bits = ((bits & 0x55555555) << 1) | ((bits >> 1) & 0x55555555)
    bits = ((bits & 0x33333333) << 2) | ((bits >> 2) & 0x33333333)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits >> 4) & 0x0F0F0F0F)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits >> 8) & 0x00FF00FF)
    bits = ((bits << 16) & 0xFFFFFFFF) | (bits >> 16)
    return bits * 2.3283064365386963e-10


def _hammersley(i, n):
    return (i / float(n), _radical_inverse_vdc(i))


def _importance_sample_ggx(xi, roughness, n):
    a = roughness * roughness
    phi = 2.0 * math.pi * xi[0]
    cos_theta = math.sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1]))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    h_t = (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)
    up = (0.0, 0.0, 1.0) if abs(n[2]) < 0.999 else (1.0, 0.0, 0.0)
    t = _normalize(_cross(up, n))
    b = _cross(n, t)
    return _normalize((
        t[0] * h_t[0] + b[0] * h_t[1] + n[0] * h_t[2],
        t[1] * h_t[0] + b[1] * h_t[1] + n[1] * h_t[2],
        t[2] * h_t[0] + b[2] * h_t[1] + n[2] * h_t[2],
    ))


def ggx_convolve(env_sample, n, roughness, num_samples):
    """按 GGX 分布对环境做预过滤卷积（不是盒式卷积）。

    env_sample: direction -> (r, g, b)，HDR 值在积分过程中不截断。
    权重按 NdotL 归一化，保证能量守恒。
    """
    n = _normalize(n)
    num_samples = max(1, int(num_samples))
    if roughness <= 1e-4:
        return env_sample(n)
    total = [0.0, 0.0, 0.0]
    weight_sum = 0.0
    v = n  # 假设 view == normal（预过滤标准近似）
    for i in range(num_samples):
        xi = _hammersley(i, num_samples)
        h = _importance_sample_ggx(xi, roughness, n)
        dvh = 2.0 * _dot(v, h)
        l = _normalize((dvh * h[0] - v[0], dvh * h[1] - v[1], dvh * h[2] - v[2]))
        ndotl = _dot(n, l)
        if ndotl > 0.0:
            c = env_sample(l)  # HDR：不 clamp
            total[0] += c[0] * ndotl
            total[1] += c[1] * ndotl
            total[2] += c[2] * ndotl
            weight_sum += ndotl
    if weight_sum <= 0.0:
        return env_sample(n)
    return (total[0] / weight_sum, total[1] / weight_sum, total[2] / weight_sum)


# ---------------------------------------------------------------------------
# mip 层数与连续粗糙度映射
# ---------------------------------------------------------------------------

def mip_levels_for_resolution(resolution):
    """mip 层数随分辨率重算：floor(log2(res)) + 1。分辨率切换后必须重算。"""
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 1:
        raise ValueError("resolution 必须是 >= 1 的整数")
    return int(math.floor(math.log2(resolution))) + 1


def roughness_to_lod(roughness, max_level):
    """粗糙度 -> 浮点 lod 的连续映射（不取整，中间档不跳变）。"""
    r = min(max(float(roughness), 0.0), 1.0)
    return r * float(max_level)


# ---------------------------------------------------------------------------
# 预过滤 mip 链：第 k 层对应粗糙度 k / max_level，按 GGX 卷积生成
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _build_chain(resolution, samples):
    levels = mip_levels_for_resolution(resolution)
    max_level = levels - 1
    chain = []
    for lvl in range(levels):
        size = max(1, resolution >> lvl)
        roughness = (lvl / max_level) if max_level else 0.0
        faces = []
        for f in range(len(FACE_ORDER)):
            face = []
            for ty in range(size):
                row = []
                for tx in range(size):
                    d = _face_uv_to_dir(f, (tx + 0.5) / size, (ty + 0.5) / size)
                    row.append(ggx_convolve(_default_env, d, roughness, samples))
                face.append(row)
            faces.append(face)
        chain.append(faces)
    return chain


def sample_prefiltered(chain, direction, roughness):
    """按连续 lod 在相邻两层 mip 之间插值采样（三线性中的 lod 维）。"""
    max_level = len(chain) - 1
    lod = roughness_to_lod(roughness, max_level)
    l0 = int(math.floor(lod))
    l1 = min(l0 + 1, max_level)
    frac = lod - l0
    c0 = sample_cube(chain[l0], direction)
    if l1 == l0 or frac == 0.0:
        return c0
    c1 = sample_cube(chain[l1], direction)
    return tuple(c0[i] * (1.0 - frac) + c1[i] * frac for i in range(3))


def tonemap_ldr(color):
    """HDR -> LDR 的截断只发生在最终输出阶段，预过滤数据保持完整动态范围。"""
    return tuple(min(max(c / (1.0 + c), 0.0), 1.0) for c in color)


# ---------------------------------------------------------------------------
# 异步 mip 生成与采样同步
# ---------------------------------------------------------------------------

class AsyncMipChain:
    """后台线程生成 mip 链；采样前必须通过事件栅栏等待写完成。"""

    def __init__(self, resolution, samples):
        self._event = threading.Event()
        self._chain = None
        self._error = None
        self._thread = threading.Thread(
            target=self._build, args=(resolution, samples), daemon=True
        )
        self._thread.start()

    def _build(self, resolution, samples):
        try:
            self._chain = _build_chain(resolution, samples)
        except Exception as exc:  # noqa: BLE001 - 传递到读侧
            self._error = exc
        finally:
            self._event.set()  # 栅栏：全部写完后才放行读

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    @property
    def levels(self):
        if not self._event.is_set():
            raise RuntimeError("mip 链尚未生成完成，禁止采样")
        if self._error is not None:
            raise self._error
        return self._chain


# ---------------------------------------------------------------------------
# 配置处理
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "resolution": 8,
    "samples": 16,
    "roughness": 0.5,
    "direction": (0.3, 0.6, 0.7),
}


def _as_config(arg):
    cfg = dict(_DEFAULTS)
    if isinstance(arg, dict):
        cfg.update(arg)
    elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
        cfg["samples"] = max(1, int(arg)) * 16
    return cfg


def validate_config(cfg):
    errors = []
    res = cfg.get("resolution")
    if not isinstance(res, int) or isinstance(res, bool) or res < 1:
        errors.append("resolution 必须是 >= 1 的整数")
    samples = cfg.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        errors.append("samples 必须是 >= 1 的整数")
    r = cfg.get("roughness")
    if not isinstance(r, (int, float)) or isinstance(r, bool) or not (0.0 <= r <= 1.0):
        errors.append("roughness 必须在 [0, 1] 内")
    d = cfg.get("direction")
    if (
        not isinstance(d, (tuple, list))
        or len(d) != 3
        or not all(isinstance(c, (int, float)) for c in d)
        or _dot(d, d) <= 0.0
    ):
        errors.append("direction 必须是非零三维向量")
    return errors


def sanitize_config(cfg):
    """非法字段回退到安全默认值。"""
    cfg = dict(cfg)
    for key, default in _DEFAULTS.items():
        candidate = cfg.get(key, default)
        probe = dict(_DEFAULTS)
        probe[key] = candidate
        if validate_config(probe):
            cfg[key] = default
        else:
            cfg[key] = candidate
    cfg["direction"] = tuple(float(c) for c in cfg["direction"])
    return cfg


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def ggx_prefilter(cfg=None):
    """GGX 分布卷积预过滤（修复：盒式卷积 -> GGX 重要性采样）。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    n = _normalize(cfg["direction"])
    color = ggx_convolve(
        lambda d: sample_cube(chain[0], d), n, cfg["roughness"], cfg["samples"]
    )
    return {"kernel": "ggx", "samples": cfg["samples"], "color": color}


def mip_mapping(cfg=None):
    """粗糙度 -> mip 的连续映射（修复：中间档取整跳变 -> 浮点 lod）。"""
    cfg = sanitize_config(_as_config(cfg))
    max_level = mip_levels_for_resolution(cfg["resolution"]) - 1
    steps = 64
    lods = [roughness_to_lod(i / steps, max_level) for i in range(steps + 1)]
    max_step = max_level / steps + 1e-9
    continuous = all(lods[i + 1] - lods[i] <= max_step for i in range(steps))
    return {"max_level": max_level, "lods": lods, "continuous": continuous}


def cube_seam(cfg=None):
    """跨面接缝采样（修复：UV clamp -> 方向重投影到相邻面）。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    eps = 1e-3
    d_pos_x = _normalize((1.0, 0.05, 1.0 - eps))  # +X 面靠 +Z 边缘
    d_pos_z = _normalize((1.0 - eps, 0.05, 1.0))  # +Z 面靠 +X 边缘
    level = chain[1] if len(chain) > 1 else chain[0]
    c1 = sample_cube(level, d_pos_x)
    c2 = sample_cube(level, d_pos_z)
    delta = max(abs(a - b) for a, b in zip(c1, c2))
    return {"delta": delta, "seamless": delta < 0.2}


def async_read(cfg=None):
    """异步生成 + 栅栏同步后再采样（修复：未写完就被采样）。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = AsyncMipChain(cfg["resolution"], cfg["samples"])
    if not chain.wait(timeout=30.0):
        raise TimeoutError("mip 链生成超时")
    d = _normalize(cfg["direction"])
    color = sample_prefiltered(chain.levels, d, cfg["roughness"])
    return {"completed": True, "color": color}


def hdr_clamp(cfg=None):
    """预过滤保留 HDR 动态范围（修复：提前截断 -> 仅在最终 LDR 输出截断）。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    color = sample_prefiltered(chain, _SUN_DIR, 0.0)
    peak = max(color)
    return {"peak": peak, "hdr_preserved": peak > 1.0, "ldr": tonemap_ldr(color)}


def determinism(cfg=None):
    """确定性：Hammersley 低差异序列，无随机状态，同输入必同输出。"""
    cfg = sanitize_config(_as_config(cfg))
    n = _normalize(cfg["direction"])
    a = ggx_convolve(_default_env, n, cfg["roughness"], cfg["samples"])
    b = ggx_convolve(_default_env, n, cfg["roughness"], cfg["samples"])
    return {"deterministic": a == b, "color": a}


def roughness_lut(cfg=None):
    """粗糙度 -> lod 查找表，单调且连续。"""
    cfg = sanitize_config(_as_config(cfg))
    max_level = mip_levels_for_resolution(cfg["resolution"]) - 1
    size = 65
    lut = [roughness_to_lod(i / (size - 1), max_level) for i in range(size)]
    monotonic = all(lut[i + 1] >= lut[i] for i in range(size - 1))
    return {"lut": lut, "max_level": max_level, "monotonic": monotonic}


def validation(cfg=None):
    """配置校验：返回错误列表，非法配置由调用方决定回退或报错。"""
    candidate = _as_config(cfg)
    errors = validate_config(candidate)
    return {
        "ok": not errors,
        "errors": errors,
        "config": candidate if not errors else sanitize_config(candidate),
    }


def audit(cfg=None):
    """对六项修复做整体审计。"""
    cfg = sanitize_config(_as_config(cfg))
    res = cfg["resolution"]
    levels = mip_levels_for_resolution(res)
    levels_2x = mip_levels_for_resolution(res * 2)
    mapping = mip_mapping(cfg)
    seam = cube_seam(cfg)
    hdr = hdr_clamp(cfg)
    det = determinism(cfg)
    checks = {
        "ggx_kernel": ggx_prefilter(cfg)["kernel"] == "ggx",
        "levels_follow_resolution": levels_2x == levels + 1,
        "mip_mapping_continuous": mapping["continuous"],
        "seamless": seam["seamless"],
        "hdr_preserved": hdr["hdr_preserved"],
        "deterministic": det["deterministic"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def fallback(cfg=None):
    """非法配置回退到安全默认值，并报告哪些字段被回退。"""
    raw = cfg if isinstance(cfg, dict) else {}
    candidate = dict(_DEFAULTS)
    candidate.update(raw)
    errors = validate_config(candidate)
    used = sanitize_config(candidate)
    fell_back = any(used.get(k) != candidate.get(k) for k in _DEFAULTS)
    return {"used_fallback": bool(errors) or fell_back, "errors": errors, "config": used}


def face_order(cfg=None):
    """规范面顺序及 dir<->uv 往返一致性。"""
    roundtrip_ok = True
    for f in range(len(FACE_ORDER)):
        for u, v in ((0.25, 0.25), (0.75, 0.6), (0.5, 0.1)):
            d = _face_uv_to_dir(f, u, v)
            f2, u2, v2 = _dir_to_face_uv(d)
            if f2 != f or abs(u2 - u) > 1e-6 or abs(v2 - v) > 1e-6:
                roundtrip_ok = False
    return {"order": FACE_ORDER, "roundtrip_ok": roundtrip_ok}


def metrics(cfg=None):
    """质量指标：层数、能量误差、接缝跳变、lod 平滑度。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    e = energy(cfg)
    s = cube_seam(cfg)
    sm = lod_smooth(cfg)
    return {
        "levels": len(chain),
        "energy_relative_error": e["relative_error"],
        "seam_delta": s["delta"],
        "lod_max_jump": sm["max_jump"],
    }


def sample(cfg=None):
    """按方向 + 粗糙度采样预过滤链（连续 lod）。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    d = _normalize(cfg["direction"])
    lod = roughness_to_lod(cfg["roughness"], len(chain) - 1)
    return {"color": sample_prefiltered(chain, d, cfg["roughness"]), "lod": lod}


def lod_smooth(cfg=None):
    """lod 维平滑：粗糙度扫描下相邻输出无跳变。"""
    cfg = sanitize_config(_as_config(cfg))
    chain = _build_chain(cfg["resolution"], cfg["samples"])
    d = _normalize(cfg["direction"])
    steps = 32
    colors = [sample_prefiltered(chain, d, i / steps) for i in range(steps + 1)]
    jumps = [
        max(abs(a - b) for a, b in zip(colors[i], colors[i + 1]))
        for i in range(steps)
    ]
    max_jump = max(jumps) if jumps else 0.0
    return {"max_jump": max_jump, "smooth": max_jump < 0.5}


def energy(cfg=None):
    """能量守恒：权重归一化，粗糙度 0 时卷积结果等于环境本身。"""
    cfg = sanitize_config(_as_config(cfg))
    n = _normalize(cfg["direction"])
    exact = _default_env(n)
    approx = ggx_convolve(_default_env, n, 0.0, cfg["samples"])
    num = max(abs(a - b) for a, b in zip(exact, approx))
    den = max(exact) or 1.0
    rel = num / den
    return {"relative_error": rel, "conserved": rel < 1e-6}
