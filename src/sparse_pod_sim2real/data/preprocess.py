from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from tqdm import tqdm


def _prefix_length(shape_t: int, prefix_frames: Optional[int | float]) -> int:
    if prefix_frames is None:
        return int(shape_t)
    pval = float(prefix_frames)
    return int(shape_t * pval) if 0.0 < pval <= 1.0 else min(int(shape_t), int(pval))


def _validate_cached_tensor(
    target_file: Path,
    target_res: Tuple[int, int],
    expected_frames: int,
) -> List[int]:
    """Reject stale caches whose shape disagrees with the requested conversion."""
    payload = torch.load(target_file, map_location="cpu", mmap=True, weights_only=True)
    tensor = payload if isinstance(payload, torch.Tensor) else payload.get("tensor")
    if tensor is None:
        raise ValueError(f"Cached file has no tensor payload: {target_file}")
    expected = (int(expected_frames), 2, int(target_res[0]), int(target_res[1]))
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Cached tensor shape mismatch for {target_file}: found {tuple(tensor.shape)}, "
            f"expected {expected}. Re-run preprocessing with --overwrite."
        )
    return list(tensor.shape)


def process_single_arrow(
    arrow_path: Path,
    out_dir: Path,
    target_res: Tuple[int, int] = (64, 128),
    prefix_frames: Optional[int | float] = None,
    overwrite: bool = False,
) -> List[Dict]:
    """Read an Arrow trajectory file, downsample to target_res, and save as float32 .pt."""
    results = []
    with pa.memory_map(str(arrow_path), "r") as src:
        reader = ipc.open_stream(src)
        for batch in reader:
            for row_idx in range(batch.num_rows):
                sim_id = str(batch.column("sim_id")[row_idx].as_py())
                shape_t = int(batch.column("shape_t")[row_idx].as_py())
                shape_h = int(batch.column("shape_h")[row_idx].as_py())
                shape_w = int(batch.column("shape_w")[row_idx].as_py())

                target_file = out_dir / f"{sim_id}.pt"
                if target_file.exists() and not overwrite:
                    cached_shape = _validate_cached_tensor(
                        target_file, target_res, _prefix_length(shape_t, prefix_frames)
                    )
                    results.append({
                        "sim_id": sim_id,
                        "status": "skipped",
                        "path": str(target_file),
                        "shape": cached_shape,
                        "bytes": target_file.stat().st_size,
                    })
                    continue

                u_buf = batch.column("u")[row_idx].as_buffer()
                v_buf = batch.column("v")[row_idx].as_buffer()

                dtype = np.float32 if u_buf.size == shape_t * shape_h * shape_w * 4 else np.float64
                u = np.frombuffer(u_buf, dtype=dtype).reshape(shape_t, shape_h, shape_w)
                v = np.frombuffer(v_buf, dtype=dtype).reshape(shape_t, shape_h, shape_w)

                sh = shape_h // target_res[0]
                sw = shape_w // target_res[1]
                if shape_h % target_res[0] != 0 or shape_w % target_res[1] != 0:
                    raise ValueError(f"Target resolution {target_res} must evenly divide raw shape ({shape_h}, {shape_w})")

                u_down = u[:, ::sh, ::sw].astype(np.float32, copy=False)
                v_down = v[:, ::sh, ::sw].astype(np.float32, copy=False)

                if prefix_frames is not None:
                    max_t = _prefix_length(shape_t, prefix_frames)
                    u_down = u_down[:max_t]
                    v_down = v_down[:max_t]

                stacked = np.stack([u_down, v_down], axis=1)  # (T, 2, H, W)
                tensor = torch.from_numpy(stacked).contiguous()

                out_dir.mkdir(parents=True, exist_ok=True)
                temp_file = target_file.with_name(f".{target_file.name}.tmp.{os.getpid()}")
                try:
                    torch.save(tensor, temp_file)
                    os.replace(temp_file, target_file)
                finally:
                    temp_file.unlink(missing_ok=True)

                results.append({
                    "sim_id": sim_id,
                    "status": "converted",
                    "path": str(target_file),
                    "shape": list(tensor.shape),
                    "bytes": target_file.stat().st_size,
                })
    return results


def process_single_h5(
    h5_path: Path,
    out_dir: Path,
    target_res: Tuple[int, int] = (64, 128),
    prefix_frames: Optional[int | float] = None,
    overwrite: bool = False,
) -> Optional[Dict]:
    """Read an HDF5 trajectory file, downsample to target_res, and save as float32 .pt."""
    # Preserve the original .h5 suffix because the official RealPDEBench split
    # metadata uses it as part of the trajectory identifier.
    sim_id = h5_path.name
    target_file = out_dir / f"{sim_id}.pt"

    with h5py.File(h5_path, "r") as f:
        # Expected keys: 'velocity' or 'u' and 'v'
        if "data" in f:
            data_ds = f["data"]
            if data_ds.shape[-1] == 2:
                shape_t, shape_h, shape_w, _ = data_ds.shape
            elif len(data_ds.shape) == 4 and data_ds.shape[1] == 2:
                shape_t, _, shape_h, shape_w = data_ds.shape
            else:
                raise ValueError(f"Unsupported 'data' shape {data_ds.shape} in {h5_path}")
            if target_file.exists() and not overwrite:
                cached_shape = _validate_cached_tensor(
                    target_file, target_res, _prefix_length(shape_t, prefix_frames)
                )
                return {
                    "sim_id": sim_id,
                    "status": "skipped",
                    "path": str(target_file),
                    "shape": cached_shape,
                    "bytes": target_file.stat().st_size,
                }
            data = data_ds[:]  # (T, H, W, C) or (T, C, H, W)
            if data.shape[-1] == 2:
                # (T, H, W, 2) -> (T, 2, H, W)
                u = data[..., 0]
                v = data[..., 1]
            else:
                u = data[:, 0]
                v = data[:, 1]
        elif "u" in f and "v" in f:
            shape_t, shape_h, shape_w = f["u"].shape
            if f["v"].shape != f["u"].shape:
                raise ValueError(f"u/v shape mismatch in {h5_path}")
            if target_file.exists() and not overwrite:
                cached_shape = _validate_cached_tensor(
                    target_file, target_res, _prefix_length(shape_t, prefix_frames)
                )
                return {
                    "sim_id": sim_id,
                    "status": "skipped",
                    "path": str(target_file),
                    "shape": cached_shape,
                    "bytes": target_file.stat().st_size,
                }
            u = f["u"][:]
            v = f["v"][:]
        else:
            raise KeyError(f"Cannot identify velocity channels in {h5_path}")

    shape_t, shape_h, shape_w = u.shape
    if shape_h % target_res[0] != 0 or shape_w % target_res[1] != 0:
        raise ValueError(
            f"Target resolution {target_res} must evenly divide raw shape ({shape_h}, {shape_w})"
        )
    sh = shape_h // target_res[0]
    sw = shape_w // target_res[1]

    u_down = u[:, ::sh, ::sw].astype(np.float32, copy=False)
    v_down = v[:, ::sh, ::sw].astype(np.float32, copy=False)

    if prefix_frames is not None:
        max_t = _prefix_length(shape_t, prefix_frames)
        u_down = u_down[:max_t]
        v_down = v_down[:max_t]

    stacked = np.stack([u_down, v_down], axis=1)  # (T, 2, H, W)
    tensor = torch.from_numpy(stacked).contiguous()

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_file = target_file.with_name(f".{target_file.name}.tmp.{os.getpid()}")
    try:
        torch.save(tensor, temp_file)
        os.replace(temp_file, target_file)
    finally:
        temp_file.unlink(missing_ok=True)

    return {
        "sim_id": sim_id,
        "status": "converted",
        "path": str(target_file),
        "shape": list(tensor.shape),
        "bytes": target_file.stat().st_size,
    }


def preprocess_domain(
    input_dir: Path,
    output_dir: Path,
    target_res: Tuple[int, int] = (64, 128),
    prefix_frames: Optional[int | float] = None,
    overwrite: bool = False,
    num_workers: int = 4,
    desc: str = "Processing",
) -> Dict[str, Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_files = sorted(input_dir.glob("*.arrow"))
    h5_files = sorted(input_dir.glob("*.h5"))

    meta_dict = {}

    if arrow_files:
        tasks = [
            (f, output_dir, target_res, prefix_frames, overwrite)
            for f in arrow_files
        ]
        with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(process_single_arrow, *t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{desc} (Arrow)"):
                for item in fut.result():
                    meta_dict[item["sim_id"]] = item

    elif h5_files:
        tasks = [
            (f, output_dir, target_res, prefix_frames, overwrite)
            for f in h5_files
        ]
        with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(process_single_h5, *t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{desc} (HDF5)"):
                res = fut.result()
                if res:
                    meta_dict[res["sim_id"]] = res

    return meta_dict
