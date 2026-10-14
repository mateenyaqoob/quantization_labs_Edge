"""
Lab 7 - Pruning in depth: criteria, iterative schedules, and the Lottery Ticket Hypothesis.

Builds on Lab 3 (unstructured vs structured, SQNR proxy). Here we train real classifiers
with a from-scratch NumPy MLP, prune with different CRITERIA and SCHEDULES, fine-tune, and
measure actual test accuracy -- plus a small Lottery Ticket Hypothesis experiment with a
proper control.

ENGINEERING NOTE (found during preparation): the first version of the iterative-pruning
schedule computed each round's threshold as a quantile of the CURRENTLY-ALIVE weights using
the ROUND's target sparsity directly. That is wrong: it compounds, so five rounds aiming for
(45%, 70%, 83%, 91%, 95%) actually produced sparsities of (45%, 83%, 97%, 99.7%, 100%) -- total
collapse. The fix: convert each round's TOTAL target sparsity into a "fraction of the still-alive
weights to remove this round" before taking the quantile. Verified below: actual sparsity now
tracks the target exactly at every round. This is a good example of why pruning code needs a
sparsity sanity-check printed at every step, not just a final accuracy number.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Ytr = np.eye(10)[ytr]

def relu(x): return np.maximum(x, 0)
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

class MLP:
    def __init__(self, hidden=64, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.3, (64, hidden)).astype(np.float32); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = r.normal(0, 0.3, (hidden, 10)).astype(np.float32); self.b2 = np.zeros(10, np.float32)
        self.mask1 = np.ones_like(self.W1); self.mask2 = np.ones_like(self.W2)

    def forward(self, X):
        z1 = X @ (self.W1 * self.mask1) + self.b1; a1 = relu(z1)
        z2 = a1 @ (self.W2 * self.mask2) + self.b2
        return z2, (X, z1, a1)

    def train(self, epochs, lr=0.5, batch=64, seed=0):
        r = np.random.default_rng(seed); n = len(Xtr)
        for _ in range(epochs):
            idx = r.permutation(n)
            for i in range(0, n, batch):
                b = idx[i:i+batch]; Xb, Yb = Xtr[b], Ytr[b]
                z2, (Xc, z1, a1) = self.forward(Xb)
                p = softmax(z2); m = len(Xb)
                dZ2 = (p - Yb) / m
                dW2 = (a1.T @ dZ2) * self.mask2; db2 = dZ2.sum(0)
                da1 = dZ2 @ (self.W2 * self.mask2).T
                dz1 = da1 * (z1 > 0)
                dW1 = (Xc.T @ dz1) * self.mask1; db1 = dz1.sum(0)
                self.W1 -= lr * dW1; self.b1 -= lr * db1
                self.W2 -= lr * dW2; self.b2 -= lr * db2
                self.W1 *= self.mask1; self.W2 *= self.mask2   # keep pruned weights exactly zero

    def accuracy(self, X, y):
        z, _ = self.forward(X); return (z.argmax(1) == y).mean()

    def sparsity(self):
        total = self.mask1.size + self.mask2.size
        zeros = (self.mask1 == 0).sum() + (self.mask2 == 0).sum()
        return zeros / total

def prune_magnitude(model, target_total_sparsity):
    """Set the mask so the OVERALL (total) sparsity equals target_total_sparsity, regardless of
    how much is already pruned. Threshold is a quantile of the currently-alive weights only."""
    cur = model.sparsity()
    if target_total_sparsity <= cur:
        return
    frac_of_alive_to_remove = (target_total_sparsity - cur) / (1 - cur)
    alive_w = np.concatenate([np.abs(model.W1[model.mask1 == 1]), np.abs(model.W2[model.mask2 == 1])])
    thr = np.quantile(alive_w, frac_of_alive_to_remove)
    model.mask1 = (np.abs(model.W1) > thr).astype(np.float32)
    model.mask2 = (np.abs(model.W2) > thr).astype(np.float32)
    model.W1 *= model.mask1; model.W2 *= model.mask2

def prune_random(model, target_total_sparsity, seed=0):
    r = np.random.default_rng(seed)
    # operate on the combined parameter pool so the overall sparsity is exact
    shapes = [model.mask1.shape, model.mask2.shape]
    flat_mask = np.concatenate([model.mask1.ravel(), model.mask2.ravel()])
    n_total = flat_mask.size
    n_target_zero = int(round(target_total_sparsity * n_total))
    alive_idx = np.flatnonzero(flat_mask == 1)
    n_drop = max(0, n_target_zero - (n_total - len(alive_idx)))
    drop = r.choice(alive_idx, size=min(n_drop, len(alive_idx)), replace=False)
    flat_mask[drop] = 0
    s1 = shapes[0][0] * shapes[0][1]
    model.mask1 = flat_mask[:s1].reshape(shapes[0]); model.mask2 = flat_mask[s1:].reshape(shapes[1])
    model.W1 *= model.mask1; model.W2 *= model.mask2

def one_shot(criterion, target_sparsity, finetune_epochs=20, seed=0):
    m = MLP(hidden=64, seed=seed); m.train(epochs=60, seed=seed)
    acc_before = m.accuracy(Xte, yte)
    criterion(m, target_sparsity)
    acc_pruned = m.accuracy(Xte, yte)
    m.train(epochs=finetune_epochs, seed=seed + 100)
    acc_finetuned = m.accuracy(Xte, yte)
    return acc_before, acc_pruned, acc_finetuned, m.sparsity()

def iterative(target_sparsity, steps=5, epochs_per_step=15, seed=0):
    m = MLP(hidden=64, seed=seed); m.train(epochs=60, seed=seed)
    sparsities = 1 - (1 - target_sparsity) ** (np.arange(1, steps + 1) / steps)   # geometric schedule
    log = []
    for s in sparsities:
        prune_magnitude(m, s)
        m.train(epochs=epochs_per_step, seed=seed + int(s * 1000))
        log.append((float(s), float(m.sparsity()), float(m.accuracy(Xte, yte))))
    return m.accuracy(Xte, yte), m.sparsity(), log

if __name__ == "__main__":
    print("=== Experiment 1: magnitude vs random pruning, one-shot, with fine-tuning ===")
    print(f"{'sparsity':>9} | {'criterion':>9} | {'before':>7} | {'pruned':>7} | {'fine-tuned':>10}")
    for sp in [0.5, 0.7, 0.9, 0.95]:
        for name, crit in [("magnitude", prune_magnitude), ("random", prune_random)]:
            accs = [one_shot(crit, sp, seed=s) for s in range(3)]
            b = np.mean([a[0] for a in accs]); pr = np.mean([a[1] for a in accs]); f = np.mean([a[2] for a in accs])
            print(f"{sp:9.0%} | {name:>9} | {b:7.4f} | {pr:7.4f} | {f:10.4f}")
        print()

    print("=== Experiment 2: one-shot vs iterative pruning at 95% sparsity, SAME total fine-tune budget ===")
    oneshot_accs = [one_shot(prune_magnitude, 0.95, finetune_epochs=75, seed=s)[2] for s in range(3)]
    print(f"  One-shot   (prune once to 95%, then 75 fine-tune epochs): acc = {np.mean(oneshot_accs):.4f}")
    iter_accs = [iterative(0.95, steps=5, epochs_per_step=15, seed=s)[0] for s in range(3)]
    print(f"  Iterative  (5 rounds x 15 fine-tune epochs = 75 total):   acc = {np.mean(iter_accs):.4f}")
    _, _, example_log = iterative(0.95, steps=5, epochs_per_step=15, seed=0)
    print("  Iterative schedule detail (seed 0): target_sp, actual_sp, acc_after_finetune")
    for t, a, acc in example_log: print(f"    target={t:.3f}  actual={a:.3f}  acc={acc:.4f}")

    print("\n=== Experiment 3: Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019) ===")
    print("Claim: a pruned MASK, re-trained from the ORIGINAL initialization, can match the dense")
    print("network. Control: same mask, but a FRESH random re-init (tests whether it's the specific")
    print("initial values that matter, or just that the sub-network is prunable at all).")
    seed = 0
    dense = MLP(hidden=64, seed=seed); init_W1, init_W2 = dense.W1.copy(), dense.W2.copy()
    dense.train(epochs=60, seed=seed)
    acc_dense = dense.accuracy(Xte, yte)
    prune_magnitude(dense, 0.9)
    mask1, mask2 = dense.mask1.copy(), dense.mask2.copy()

    lottery = MLP(hidden=64, seed=seed)
    lottery.W1, lottery.W2 = init_W1.copy(), init_W2.copy()
    lottery.mask1, lottery.mask2 = mask1, mask2
    lottery.W1 *= lottery.mask1; lottery.W2 *= lottery.mask2
    lottery.train(epochs=60, seed=seed)

    reinit = MLP(hidden=64, seed=seed + 999)
    reinit.mask1, reinit.mask2 = mask1, mask2
    reinit.W1 *= reinit.mask1; reinit.W2 *= reinit.mask2
    reinit.train(epochs=60, seed=seed + 999)

    print(f"  Dense baseline (0% sparsity):                   acc = {acc_dense:.4f}")
    print(f"  Winning ticket (mask + ORIGINAL init, 90% sp.): acc = {lottery.accuracy(Xte, yte):.4f}")
    print(f"  Same mask + RANDOM re-init (control), 90% sp.:  acc = {reinit.accuracy(Xte, yte):.4f}")
    print("\nDISCUSS: run Experiment 3 over several seeds. Is the winning-ticket advantage over the")
    print("         random-reinit control consistent, or within noise at this model/dataset scale?")
