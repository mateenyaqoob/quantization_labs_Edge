"""
Lab 3 - Pruning, and why sparsity != speed. Then: benchmark like an engineer.

Part A: magnitude pruning in NumPy; accuracy proxy vs sparsity; the
        unstructured-vs-structured distinction and its hardware consequence.
Part B: latency benchmarking done RIGHT (warm-up, many runs, percentiles),
        using ONNX Runtime on a model we build ourselves.
"""
import time, numpy as np
rng = np.random.default_rng(1)

# ================= Part A: magnitude pruning =================
def sqnr_db(x, y): return 10*np.log10(np.sum(x**2)/(np.sum((x-y)**2)+1e-30))

def prune_unstructured(W, sparsity):
    thr = np.quantile(np.abs(W), sparsity)
    return np.where(np.abs(W) >= thr, W, 0.0)

def prune_structured(W, sparsity):
    """Remove whole OUTPUT rows (= neurons/filters) with smallest L2 norm."""
    n_drop = int(sparsity * W.shape[0])
    order = np.argsort(np.linalg.norm(W, axis=1))
    keep = np.sort(order[n_drop:])
    return W[keep], keep

print("Part A: pruning a dense layer  (256 -> 512, heavy-tailed weights)")
W = rng.standard_t(df=3, size=(512, 256)).astype(np.float32) * 0.05
X = rng.normal(size=(128, 256)).astype(np.float32)
Y = X @ W.T
print(f"{'sparsity':>9} | {'unstructured SQNR':>18} | {'structured SQNR':>16} | {'struct. real MAC cut':>20}")
for sp in [0.3, 0.5, 0.7, 0.9]:
    Wu = prune_unstructured(W, sp)
    Yu = X @ Wu.T
    Ws, keep = prune_structured(W, sp)
    Ys = np.zeros_like(Y); Ys[:, keep] = X @ Ws.T
    print(f"{sp:9.0%} | {sqnr_db(Y,Yu):15.2f} dB | {sqnr_db(Y,Ys):13.2f} dB | {1/(1-sp):18.2f}x")
print("  -> unstructured wins on accuracy, but its zeros sit at irregular positions.")
print("     A dense kernel still multiplies them. Structured pruning shrinks the matrix.\n")

# ================= Part B: benchmarking done right =================
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def make_mlp_onnx(path, d_in=512, d_hidden=1024, d_out=10, seed=0):
    r = np.random.default_rng(seed)
    W1 = numpy_helper.from_array(r.normal(0,.05,(d_in,d_hidden)).astype(np.float32),"W1")
    b1 = numpy_helper.from_array(np.zeros(d_hidden,np.float32),"b1")
    W2 = numpy_helper.from_array(r.normal(0,.05,(d_hidden,d_hidden)).astype(np.float32),"W2")
    b2 = numpy_helper.from_array(np.zeros(d_hidden,np.float32),"b2")
    W3 = numpy_helper.from_array(r.normal(0,.05,(d_hidden,d_out)).astype(np.float32),"W3")
    b3 = numpy_helper.from_array(np.zeros(d_out,np.float32),"b3")
    nodes = [
        helper.make_node("Gemm",["x","W1","b1"],["h1"]), helper.make_node("Relu",["h1"],["a1"]),
        helper.make_node("Gemm",["a1","W2","b2"],["h2"]), helper.make_node("Relu",["h2"],["a2"]),
        helper.make_node("Gemm",["a2","W3","b3"],["y"]),
    ]
    g = helper.make_graph(nodes,"mlp",
        [helper.make_tensor_value_info("x",TensorProto.FLOAT,[None,d_in])],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,[None,d_out])],
        initializer=[W1,b1,W2,b2,W3,b3])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("",13)])
    m.ir_version = 8
    onnx.checker.check_model(m); onnx.save(m, path)

def benchmark(sess, x, warmup=20, runs=300):
    name = sess.get_inputs()[0].name
    for _ in range(warmup): sess.run(None, {name: x})              # warm-up: caches, lazy init
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter(); sess.run(None, {name: x}); t[i] = (time.perf_counter()-t0)*1e3
    return t

make_mlp_onnx("mlp.onnx")
so = ort.SessionOptions(); so.intra_op_num_threads = 1            # single thread ~ small edge core
sess = ort.InferenceSession("mlp.onnx", so, providers=["CPUExecutionProvider"])

print("Part B: latency benchmark (batch=1, 1 thread, 300 runs after 20 warm-up)")
x = rng.normal(size=(1,512)).astype(np.float32)
t = benchmark(sess, x)
print(f"  mean {t.mean():.3f} ms | p50 {np.percentile(t,50):.3f} | p95 {np.percentile(t,95):.3f} "
      f"| p99 {np.percentile(t,99):.3f} | max {t.max():.3f} ms")
print(f"  tail check: p99 is {np.percentile(t,99)/np.percentile(t,50):.2f}x the median (quiet machine)")

print("\n  Cold-start vs warm (first call includes lazy init):")
sess2 = ort.InferenceSession("mlp.onnx", so, providers=["CPUExecutionProvider"])
t0 = time.perf_counter(); sess2.run(None, {"x": x}); cold = (time.perf_counter()-t0)*1e3
print(f"  first call {cold:.3f} ms  vs  warm median {np.percentile(t,50):.3f} ms")

print("\n  Batch-size effect on throughput (why batch=1 is the edge regime):")
for bs in [1, 4, 16, 64]:
    xb = rng.normal(size=(bs,512)).astype(np.float32)
    tb = benchmark(sess, xb, warmup=10, runs=100)
    print(f"    batch {bs:3d}: {np.median(tb):7.3f} ms/call | {np.median(tb)/bs:7.4f} ms/sample")

# ---- Tail latency under contention: why p99 (not mean) is the edge SLO ----
import threading
print("\n  Tail latency under CPU contention (background busy-loop threads):")
stop = False
def burn():
    while not stop: sum(i*i for i in range(2000))
workers = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
for w_ in workers: w_.start()
tc = benchmark(sess, x, warmup=20, runs=300)
stop = True
print(f"    quiet : p50 {np.percentile(t,50):.3f} ms | p99 {np.percentile(t,99):.3f} ms")
print(f"    busy  : p50 {np.percentile(tc,50):.3f} ms | p99 {np.percentile(tc,99):.3f} ms | max {tc.max():.3f} ms")
print("    A real-time deadline must be met by the p99 (or max), not the mean.")
