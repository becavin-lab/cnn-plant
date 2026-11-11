#!/usr/bin/env python3
"""Convertit un ResNet18 PyTorch (Pl@ntNet-300K) en fichiers pour la webapp, SANS installer torch.

Usage :
    python3 convert.py plantnet_resnet18.pth plantnet300K_species_id_2_name.json ../web/model

Produit dans le dossier de sortie :
    weights.bin   poids en float16 (BatchNorm fusionnée dans les convolutions)
    model.json    description des couches + position des poids dans weights.bin
    species.json  noms latins des 1 081 espèces, dans l'ordre des sorties du réseau
"""
import io, json, pickle, sys, zipfile, collections, os
import numpy as np

# ---------------------------------------------------------------- lecture .pth sans torch
_DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
       "LongStorage": np.int64, "IntStorage": np.int32, "BFloat16Storage": None,
       "ByteStorage": np.uint8, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name): self.name = name


def _rebuild_tensor(storage, offset, size, stride, *args):
    arr, = storage
    if len(size) == 0:
        return arr[offset:offset + 1].reshape(())
    itemsize = arr.dtype.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def _rebuild_parameter(data, *args):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f); self.zf = zf; self.prefix = prefix; self.cache = {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in ("float32", "float16", "int64"):
            return name
        raise pickle.UnpicklingError(f"objet non autorisé dans le fichier : {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _loc, _n = pid
        if key not in self.cache:
            name = stype.name if isinstance(stype, _StorageType) else "FloatStorage"
            dt = _DT.get(name, np.float32)
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            self.cache[key] = np.frombuffer(raw, dtype=dt)
        return (self.cache[key],)


def load_pth(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("/data.pkl")]
    obj = _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    for k in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            obj = obj[k]
    return {k.replace("module.", ""): v for k, v in obj.items()}


# ---------------------------------------------------------------- fusion BN + export
EPS = 1e-5


def fold(sd, conv, bn):
    w = sd[conv + ".weight"].astype(np.float32)
    g, b = sd[bn + ".weight"], sd[bn + ".bias"]
    m, v = sd[bn + ".running_mean"], sd[bn + ".running_var"]
    s = g / np.sqrt(v + EPS)
    return w * s[:, None, None, None], (b - m * s).astype(np.float32)


def resnet18_layers(sd):
    """Liste ordonnée des couches (format compris par cnn.js)."""
    L = []
    w, b = fold(sd, "conv1", "bn1")
    L.append(dict(name="conv1", type="conv", stride=2, pad=3, relu=True, w=w, b=b, tap="conv1"))
    L.append(dict(name="maxpool", type="maxpool", k=3, stride=2, pad=1))
    for li, (ch, st) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], start=1):
        for bi in range(2):
            p = f"layer{li}.{bi}"
            stride = st if bi == 0 else 1
            w1, b1 = fold(sd, p + ".conv1", p + ".bn1")
            w2, b2 = fold(sd, p + ".conv2", p + ".bn2")
            blk = dict(name=p, type="block", stride=stride,
                       c1=dict(w=w1, b=b1), c2=dict(w=w2, b=b2))
            if p + ".downsample.0.weight" in sd:
                wd, bd = fold(sd, p + ".downsample.0", p + ".downsample.1")
                blk["down"] = dict(w=wd, b=bd)
            if bi == 1:
                blk["tap"] = f"layer{li}"
            L.append(blk)
    L.append(dict(name="avgpool", type="gap"))
    L.append(dict(name="fc", type="fc", w=sd["fc.weight"].astype(np.float32), b=sd["fc.bias"].astype(np.float32)))
    return L


def export(sd, outdir):
    os.makedirs(outdir, exist_ok=True)
    layers = resnet18_layers(sd)
    blobs, off = [], 0

    def put(arr):
        nonlocal off
        a = np.ascontiguousarray(arr, dtype=np.float16)
        blobs.append(a.tobytes()); o = off; off += a.size
        return dict(off=o, shape=list(arr.shape))

    spec = []
    for l in layers:
        d = {k: v for k, v in l.items() if not isinstance(v, (np.ndarray, dict))}
        for key in ("w", "b"):
            if key in l: d[key] = put(l[key])
        for sub in ("c1", "c2", "down"):
            if sub in l: d[sub] = dict(w=put(l[sub]["w"]), b=put(l[sub]["b"]))
        spec.append(d)
    # filtres bruts de la 1re couche (pour les afficher), non fusionnés
    raw = sd["conv1.weight"].astype(np.float32)
    meta = dict(arch="resnet18", input=[3, 224, 224], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                n_classes=int(sd["fc.weight"].shape[0]), layers=spec, conv1_raw=put(raw), n_params=int(off))
    with open(os.path.join(outdir, "weights.bin"), "wb") as f:
        for bl in blobs: f.write(bl)
    with open(os.path.join(outdir, "model.json"), "w") as f:
        json.dump(meta, f)
    return meta


def species_list(names_json, n):
    names = json.load(open(names_json, encoding="utf-8"))
    ids = sorted(names.keys())  # ordre torchvision ImageFolder (tri des chaînes)
    assert len(ids) == n, f"{len(ids)} espèces dans le fichier, {n} sorties dans le réseau"
    out = []
    for i in ids:
        nm = names[i].strip()
        out.append(dict(id=i, name=nm, genus=nm.split()[0]))
    return out


if __name__ == "__main__":
    pth, names, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    sd = load_pth(pth)
    print(len(sd), "tenseurs lus ;", "fc:", sd["fc.weight"].shape)
    meta = export(sd, outdir)
    sp = species_list(names, meta["n_classes"])
    json.dump(sp, open(os.path.join(outdir, "species.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("OK :", meta["n_params"], "poids,", len(sp), "espèces ->", outdir)
