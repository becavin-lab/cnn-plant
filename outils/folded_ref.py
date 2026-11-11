"""Passe avant numpy à partir de model.json + weights.bin (poids fusionnés, float16) -> logits de référence."""
import json, sys, numpy as np
from test_tools import conv2d, maxpool

d = sys.argv[1]
meta = json.load(open(d + "/model.json"))
W = np.fromfile(d + "/weights.bin", dtype=np.float16).astype(np.float64)
T = lambda r: W[r["off"]:r["off"] + int(np.prod(r["shape"]))].reshape(r["shape"])
x = np.load(sys.argv[2]).astype(np.float64)
relu = lambda a: np.maximum(a, 0)
for L in meta["layers"]:
    if L["type"] == "conv":
        x = conv2d(x, T(L["w"]), L["stride"], L["pad"]) + T(L["b"])[:, None, None]
        x = relu(x) if L["relu"] else x
    elif L["type"] == "maxpool":
        x = maxpool(x)
    elif L["type"] == "block":
        y = relu(conv2d(x, T(L["c1"]["w"]), L["stride"], 1) + T(L["c1"]["b"])[:, None, None])
        y = conv2d(y, T(L["c2"]["w"]), 1, 1) + T(L["c2"]["b"])[:, None, None]
        idt = conv2d(x, T(L["down"]["w"]), L["stride"], 0) + T(L["down"]["b"])[:, None, None] if "down" in L else x
        x = relu(y + idt)
    elif L["type"] == "gap":
        x = x.mean(axis=(1, 2))
    elif L["type"] == "fc":
        x = T(L["w"]) @ x + T(L["b"])
np.save(sys.argv[3], x)
ref = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
print("folded argmax", x.argmax(), "| ref argmax", None if ref is None else ref.argmax(),
      "| corr", None if ref is None else np.corrcoef(x, ref)[0, 1])
