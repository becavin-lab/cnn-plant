"""Outils de test : faux .pth au format PyTorch (sans torch) et ResNet18 de référence en numpy."""
import io, pickle, sys, types, zipfile, collections, json
import numpy as np

# ----------------------------------------------------------- faux modules torch pour écrire le pickle
torch = types.ModuleType("torch"); tu = types.ModuleType("torch._utils")
def _rebuild_tensor_v2(*a): pass
_rebuild_tensor_v2.__module__ = "torch._utils"; tu._rebuild_tensor_v2 = _rebuild_tensor_v2
class FloatStorage: pass
FloatStorage.__module__ = "torch"; torch.FloatStorage = FloatStorage; torch._utils = tu
sys.modules.setdefault("torch", torch); sys.modules.setdefault("torch._utils", tu)


class _T:  # tenseur à sérialiser
    def __init__(self, arr, key): self.arr, self.key = arr, key
    def __reduce__(self):
        a = self.arr
        stride = tuple(int(s // a.itemsize) for s in a.strides)
        return (_rebuild_tensor_v2, (_S(self.key, a), 0, tuple(a.shape), stride, False, collections.OrderedDict()))


class _S:
    def __init__(self, key, arr): self.key, self.arr = key, arr


def write_fake_pth(sd, path):
    store = {}
    class P(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _S):
                store[obj.key] = obj.arr
                return ("storage", FloatStorage, obj.key, "cpu", obj.arr.size)
            return None
    od = collections.OrderedDict((k, _T(np.ascontiguousarray(v, np.float32), str(i))) for i, (k, v) in enumerate(sd.items()))
    buf = io.BytesIO(); P(buf, protocol=2).dump(od)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        z.writestr("archive/version", "3\n")
        for k, a in store.items(): z.writestr(f"archive/data/{k}", a.tobytes())


def random_resnet18(n_classes=1081, seed=0):
    rng = np.random.default_rng(seed); sd = collections.OrderedDict()
    def conv(name, co, ci, k): sd[name + ".weight"] = rng.normal(0, np.sqrt(2 / (ci * k * k)), (co, ci, k, k))
    def bn(name, c):
        sd[name + ".weight"] = rng.uniform(0.5, 1.5, c); sd[name + ".bias"] = rng.normal(0, .1, c)
        sd[name + ".running_mean"] = rng.normal(0, .1, c); sd[name + ".running_var"] = rng.uniform(0.5, 1.5, c)
    conv("conv1", 64, 3, 7); bn("bn1", 64)
    cin = 64
    for li, ch in enumerate([64, 128, 256, 512], 1):
        for bi in range(2):
            p = f"layer{li}.{bi}"
            conv(p + ".conv1", ch, cin if bi == 0 else ch, 3); bn(p + ".bn1", ch)
            conv(p + ".conv2", ch, ch, 3); bn(p + ".bn2", ch)
            if bi == 0 and (li > 1):
                conv(p + ".downsample.0", ch, cin, 1); bn(p + ".downsample.1", ch)
        cin = ch
    sd["fc.weight"] = rng.normal(0, 0.05, (n_classes, 512)); sd["fc.bias"] = rng.normal(0, 0.01, n_classes)
    return sd


# ----------------------------------------------------------- ResNet18 de référence (sémantique torchvision)
def conv2d(x, w, stride, pad):
    C, H, W = x.shape; co, ci, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1; Wo = (W + 2 * pad - k) // stride + 1
    cols = np.empty((ci * k * k, Ho * Wo), np.float64); r = 0
    for c in range(ci):
        for i in range(k):
            for j in range(k):
                cols[r] = xp[c, i:i + stride * Ho:stride, j:j + stride * Wo:stride].ravel(); r += 1
    return (w.reshape(co, -1) @ cols).reshape(co, Ho, Wo)


def bnf(x, sd, n):
    s = sd[n + ".weight"] / np.sqrt(sd[n + ".running_var"] + 1e-5)
    return x * s[:, None, None] + (sd[n + ".bias"] - sd[n + ".running_mean"] * s)[:, None, None]


def maxpool(x, k=3, s=2, p=1):
    C, H, W = x.shape; xp = np.pad(x, ((0, 0), (p, p), (p, p)), constant_values=-np.inf)
    Ho = (H + 2 * p - k) // s + 1; Wo = (W + 2 * p - k) // s + 1
    out = np.full((C, Ho, Wo), -np.inf)
    for i in range(k):
        for j in range(k):
            out = np.maximum(out, xp[:, i:i + s * Ho:s, j:j + s * Wo:s])
    return out


def ref_forward(sd, x):
    relu = lambda a: np.maximum(a, 0)
    taps = {}
    x = relu(bnf(conv2d(x, sd["conv1.weight"], 2, 3), sd, "bn1")); taps["conv1"] = x
    x = maxpool(x)
    for li, st in enumerate([1, 2, 2, 2], 1):
        for bi in range(2):
            p = f"layer{li}.{bi}"; s = st if bi == 0 else 1
            idt = x
            y = relu(bnf(conv2d(x, sd[p + ".conv1.weight"], s, 1), sd, p + ".bn1"))
            y = bnf(conv2d(y, sd[p + ".conv2.weight"], 1, 1), sd, p + ".bn2")
            if p + ".downsample.0.weight" in sd:
                idt = bnf(conv2d(x, sd[p + ".downsample.0.weight"], s, 0), sd, p + ".downsample.1")
            x = relu(y + idt)
        taps[f"layer{li}"] = x
    f = x.mean(axis=(1, 2))
    return sd["fc.weight"] @ f + sd["fc.bias"], taps


if __name__ == "__main__":
    out = sys.argv[1]
    sd = random_resnet18()
    write_fake_pth(sd, out + "/fake_resnet18.pth")
    names = {str(100000 + i * 7): f"Genus{i % 97} species{i}" for i in range(1081)}
    json.dump(names, open(out + "/fake_species_id_2_name.json", "w"))
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (3, 224, 224))
    logits, taps = ref_forward(sd, x)
    np.save(out + "/test_input.npy", x.astype(np.float32)); np.save(out + "/test_logits.npy", logits)
    print("ref logits[:5]", logits[:5], "argmax", logits.argmax())
