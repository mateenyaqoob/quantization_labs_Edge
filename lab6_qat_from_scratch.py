"""
Lab 6 - Quantization-Aware Training (QAT) from scratch, NumPy only.

Builds on Lab 2 (PTQ mechanics). Implements:
  * the "fake-quant" node: quantize-dequantize in the forward pass, with a FIXED clip
    range (set once from the pretrained fp32 weights) -- this is standard practice
    (Jacob et al., CVPR 2018) and is NOT the same as recomputing max|W| every step.
  * the Straight-Through Estimator (STE) for the backward pass: gradient passes
    through unchanged inside the clip range, and is zeroed outside it.
  * a head-to-head: FP32 baseline vs PTQ vs QAT across decreasing bit-widths.

ENGINEERING NOTE (why the fixed clip range matters -- found during preparation):
  An earlier version of this lab recomputed the clip range from the CURRENT max|W|
  on every forward pass. At 2 bits (only 3 representable levels: -m, 0, +m) this
  creates a moving target: SGD pushes weights larger, which pushes m larger, which
  re-quantizes everything again, and training never settles -- QAT ended up WORSE
  than PTQ, which should never happen. Holding the clip range fixed (as real QAT
  does; see Jacob et al.) fixes this. This is a good illustration of why every
  "my method didn't work" result needs to be checked against a sane baseline
  before you believe it is a property of the METHOD rather than of your CODE.

Dataset: sklearn digits (8x8 images, 10 classes), small by design so every laptop
reproduces this in well under a minute. The qualitative pattern (QAT closes the
accuracy gap at low bit-width, and the gap grows as bits decrease) is the lesson;
exact percentages are not meant to generalize to ImageNet-scale models.
"""
import numpy as np, copy, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Ytr = np.eye(10)[ytr]

def fake_quant(W, bits, clip):
    """Forward: symmetric uniform quantization to a FIXED range [-clip, clip].
       Backward (STE): identity gradient inside the range, zero outside (clipped)."""
    qmax = 2 ** (bits - 1) - 1
    scale = clip / qmax
    Wc = np.clip(W, -clip, clip)
    Wq = np.round(Wc / scale) * scale
    ste_mask = (np.abs(W) <= clip).astype(np.float32)
    return Wq, ste_mask

def relu(x): return np.maximum(x, 0)
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

class MLP:
    """2-layer MLP (64 -> hidden -> 10). Clip ranges are set ONCE at construction
       (from the initial weight scale) and held fixed through training -- this is what
       keeps low-bit QAT stable. Pass quant_w=False to run in plain FP32."""
    def __init__(self, hidden=32, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.3, (64, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 0.3, (hidden, 10)); self.b2 = np.zeros(10)
        self.clip1 = np.abs(self.W1).max()
        self.clip2 = np.abs(self.W2).max()

    def set_clips_from(self, other):
        """For PTQ: borrow the clip range from an already-trained fp32 model's weights."""
        self.clip1 = np.abs(other.W1).max()
        self.clip2 = np.abs(other.W2).max()

    def forward(self, X, bits=None, quant_w=False, training=True):
        if quant_w:
            W1, m1 = fake_quant(self.W1, bits, self.clip1)
            W2, m2 = fake_quant(self.W2, bits, self.clip2)
        else:
            W1, m1, W2, m2 = self.W1, None, self.W2, None
        z1 = X @ W1 + self.b1; a1 = relu(z1)
        z2 = a1 @ W2 + self.b2
        return z2, dict(X=X, z1=z1, a1=a1, W1=W1, W2=W2, m1=m1, m2=m2)

    def backward(self, cache, dZ2):
        dW2 = cache["a1"].T @ dZ2; db2 = dZ2.sum(0)
        if cache["m2"] is not None: dW2 = dW2 * cache["m2"]
        da1 = dZ2 @ cache["W2"].T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["X"].T @ dz1; db1 = dz1.sum(0)
        if cache["m1"] is not None: dW1 = dW1 * cache["m1"]
        return dW1, db1, dW2, db2

    def step(self, X, Y, bits, quant_w, lr):
        z, cache = self.forward(X, bits, quant_w, training=True)
        p = softmax(z); n = len(X)
        dZ2 = (p - Y) / n
        dW1, db1, dW2, db2 = self.backward(cache, dZ2)
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2

    def accuracy(self, X, y, bits=None, quant_w=False):
        z, _ = self.forward(X, bits, quant_w, training=False)
        return (z.argmax(1) == y).mean()

def train(model, epochs, bits=None, quant_w=False, lr=0.5, batch=64, seed=0):
    r = np.random.default_rng(seed); n = len(Xtr)
    for _ in range(epochs):
        idx = r.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            model.step(Xtr[b], Ytr[b], bits, quant_w, lr)
    return model

if __name__ == "__main__":
    print("Step 1: train an FP32 baseline (shared starting point for PTQ)")
    fp32 = MLP(hidden=32, seed=0)
    train(fp32, epochs=60)
    acc_fp32 = fp32.accuracy(Xte, yte)
    print(f"  FP32 baseline accuracy: {acc_fp32:.4f}\n")

    print(f"{'bits':>5} | {'PTQ':>8} | {'QAT':>8} | {'QAT - PTQ':>10}")
    results = {"bits": [], "ptq": [], "qat": []}
    for bits in [8, 6, 4, 3, 2]:
        # PTQ: quantize the FROZEN fp32 weights with no further training
        ptq_model = copy.deepcopy(fp32)
        ptq_model.set_clips_from(fp32)
        acc_ptq = ptq_model.accuracy(Xte, yte, bits=bits, quant_w=True)

        # QAT: fresh init (same seed as fp32 for a fair comparison), fake-quant DURING training
        qat_model = MLP(hidden=32, seed=0)
        train(qat_model, epochs=60, bits=bits, quant_w=True)
        acc_qat = qat_model.accuracy(Xte, yte, bits=bits, quant_w=True)

        results["bits"].append(bits); results["ptq"].append(float(acc_ptq)); results["qat"].append(float(acc_qat))
        print(f"{bits:5d} | {acc_ptq:8.4f} | {acc_qat:8.4f} | {acc_qat-acc_ptq:+10.4f}")

    print(f"\nFP32 reference: {acc_fp32:.4f}")
    print("\nDISCUSS: the PTQ/QAT gap is near zero at 8-6 bits and large at 2-3 bits. Why?")
    print("DISCUSS: QAT needs a full retraining run; PTQ needs seconds. When is QAT worth the cost?")
    print("DISCUSS: what would happen if the clip range were recomputed from max|W| every step")
    print("         instead of held fixed? (Try it: this was a real bug found while building this lab.)")

    import json
    with open("qat_results.json", "w") as f: json.dump(results, f)
