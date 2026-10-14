"""
Lab 2 - Post-training quantization from first principles (NumPy only).

Implements the affine (asymmetric) INT8 scheme used by TFLite / ONNX Runtime:
        real = scale * (q - zero_point)
and shows: per-tensor vs per-channel, the effect of outliers, calibration by
percentile clipping, and how error accumulates through a layer.
"""
import numpy as np
rng = np.random.default_rng(0)

# ---------- core affine quantizer ----------
def qparams(x_min, x_max, n_bits=8, symmetric=False):
    qmin, qmax = (-(2**(n_bits-1)), 2**(n_bits-1) - 1)          # int8: -128..127
    x_min, x_max = min(x_min, 0.0), max(x_max, 0.0)             # range must include 0
    if symmetric:
        m = max(abs(x_min), abs(x_max))
        scale = m / qmax if m > 0 else 1.0
        return scale, 0
    scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else 1.0
    zp = int(np.clip(round(qmin - x_min / scale), qmin, qmax))
    return scale, zp

def quantize(x, scale, zp, n_bits=8):
    qmin, qmax = (-(2**(n_bits-1)), 2**(n_bits-1) - 1)
    return np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.int32)

def dequantize(q, scale, zp):
    return scale * (q.astype(np.float32) - zp)

def sqnr_db(x, x_hat):
    """Signal-to-quantization-noise ratio in dB (higher = better)."""
    return 10 * np.log10(np.sum(x**2) / (np.sum((x - x_hat)**2) + 1e-30))

# ---------- Experiment 1: bit-width sweep ----------
print("Exp 1: SQNR vs bit-width (Gaussian weights)")
w = rng.normal(0, 0.05, size=100_000).astype(np.float32)
for b in [8, 6, 4, 2]:
    s, z = qparams(w.min(), w.max(), b)
    w_hat = dequantize(quantize(w, s, z, b), s, z)
    ideal = 6.02 * b + 1.76
    print(f"  {b}-bit: SQNR = {sqnr_db(w, w_hat):6.2f} dB   [ideal {ideal:5.1f} dB]")

print("  DISCUSS: measured SQNR sits well below the ideal 6.02b+1.76 dB. Why?")
print("  (Hint: the ideal assumes a signal that uses the FULL range; a Gaussian's")
print("   min/max are set by rare tail values, so most weights use few levels.)")

# ---------- Experiment 2: outliers wreck per-tensor scale ----------
print("\nExp 2: one outlier vs percentile calibration")
w_out = w.copy(); w_out[0] = 5.0                                  # a single large weight
s, z = qparams(w_out.min(), w_out.max())
print(f"  min/max range   : SQNR = {sqnr_db(w_out, dequantize(quantize(w_out,s,z),s,z)):6.2f} dB")
lo, hi = np.percentile(w_out, [0.01, 99.99])
s, z = qparams(lo, hi)
w_clip = dequantize(quantize(w_out, s, z), s, z)
mask = np.arange(len(w_out)) != 0
print(f"  99.99% clipping : SQNR on non-outliers = {sqnr_db(w_out[mask], w_clip[mask]):6.2f} dB "
      f"(outlier clipped 5.0 -> {w_clip[0]:.3f})")

# ---------- Experiment 3: per-tensor vs per-channel ----------
print("\nExp 3: per-tensor vs per-channel (conv weights with unequal channel ranges)")
W = rng.normal(0, 1, size=(16, 3 * 3 * 32)).astype(np.float32)
W *= np.logspace(-2, 0, 16)[:, None]                              # channels differ 100x in scale
s, z = qparams(W.min(), W.max(), symmetric=True)
Wt = dequantize(quantize(W, s, z), s, z)
Wc = np.empty_like(W)
for c in range(16):
    s, z = qparams(W[c].min(), W[c].max(), symmetric=True)
    Wc[c] = dequantize(quantize(W[c], s, z), s, z)
print(f"  per-tensor  SQNR = {sqnr_db(W, Wt):6.2f} dB")
print(f"  per-channel SQNR = {sqnr_db(W, Wc):6.2f} dB")

# ---------- Experiment 4: integer-only matmul ----------
print("\nExp 4: INT8 matmul with INT32 accumulator vs FP32")
X = rng.normal(0, 1, size=(64, 256)).astype(np.float32)
Wm = rng.normal(0, 0.1, size=(256, 128)).astype(np.float32)
sx, zx = qparams(X.min(), X.max())
sw, zw = qparams(Wm.min(), Wm.max(), symmetric=True)
Xq, Wq = quantize(X, sx, zx), quantize(Wm, sw, zw)
acc = (Xq - zx) @ (Wq - zw)                                       # int32 accumulate
Y_int = acc * (sx * sw)                                           # requantize scale
Y_fp = X @ Wm
print(f"  output SQNR (INT8 vs FP32) = {sqnr_db(Y_fp, Y_int):6.2f} dB")
print(f"  max |error| = {np.abs(Y_fp - Y_int).max():.4f}  (output std = {Y_fp.std():.3f})")
print(f"  memory: FP32 weights {Wm.nbytes/1e3:.1f} kB -> INT8 {Wq.astype(np.int8).nbytes/1e3:.1f} kB (4x)")
