import os, csv, pathlib, numpy as np, pickle as pkl, h5py, torch

split_csv = "/data/pathology/projects/ivan/DeepDerma/documents/classifier_splits/mohs_based_on_2024_test/splits_0.csv"
patch_features_dir = pathlib.Path("/data/pa_cpgarchive/archives/skin/internal/images/derived/features_virchow_p2_n0_f3/features_virchow2/")
coordinates_dir = pathlib.Path("/data/pa_cpgarchive/archives/skin/internal/images/derived/features_virchow_p2_n0_f3/coordinates/")
coordinates_suffix = ".npy"
suffixes = [".pt", ".npy", ".npz", ".pkl", ".h5","_patches.h5"]

def collect_basenames(split="train"):
    bases = []
    with open(split_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split in row and row[split]:
                base = os.path.splitext(os.path.basename(row[split].strip()))[0]
                bases.append(base)
    return bases

def find_feature_file(basename):
    for s in suffixes:
        p = patch_features_dir / f"{basename}{s}"
        if p.exists():
            return p
    return None

def load_shape(path):
    ext = path.suffix.lower()
    if ext == ".npy":
        return tuple(np.load(path, allow_pickle=True).shape)
    if ext == ".npz":
        arr = np.load(path)
        return tuple(arr[list(arr.files)[0]].shape)
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pkl.load(f)
        if isinstance(data, dict) and "features" in data:
            data = data["features"]
        return tuple(np.array(data).shape)
    if ext == ".h5":
        with h5py.File(path, "r") as f:
            key = "features" if "features" in f else ("data" if "data" in f else list(f.keys())[0])
            return tuple(f[key][:].shape)
    if ext == ".pt":
        data = torch.load(path, map_location='cpu')
        # Expect tensor or dict with 'features' key
        if isinstance(data, dict):
            if 'features' in data:
                f = data['features']
            else:
                # try to find first tensor
                tensor_keys = [k for k, v in data.items() if isinstance(v, (torch.Tensor, np.ndarray))]
                if tensor_keys:
                    f = data[tensor_keys[0]]
                else:
                    raise ValueError(f"No tensor-like object found in {path}")
        elif isinstance(data, torch.Tensor):
            f = data
        elif isinstance(data, np.ndarray):
            f = torch.from_numpy(data)
        else:
            raise ValueError(f"Unsupported data type in {path}: {type(data)}")
        return tuple(f.shape)
    return None

def load_coords(basename):
    candidates = [
        coordinates_dir / f"{basename}{coordinates_suffix}",
        coordinates_dir / f"{basename}_patches{coordinates_suffix}",
        coordinates_dir / f"{basename}.npy",
        coordinates_dir / f"{basename}.npz",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".npy":
                return np.load(p, allow_pickle=True)
            if p.suffix.lower() == ".npz":
                data = np.load(p)
                return data[data.files[0]]
            if p.suffix.lower() in [".h5", ".hdf5"]:
                with h5py.File(p, "r") as f:
                    key = "coords" if "coords" in f else list(f.keys())[0]
                    return f[key][:]
    return None

bases = collect_basenames("train")
print(f"Train items: {len(bases)}")
for base in bases[:10]:
    feat_path = find_feature_file(base)
    feat_shape = load_shape(feat_path) if feat_path else None
    coords = load_coords(base)
    coord_len = None if coords is None else len(coords)
    print(f"{base}: feat_path={feat_path}, feat_shape={feat_shape}, coords_len={coord_len}")
