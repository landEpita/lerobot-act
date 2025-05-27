"""
delete_episode.py — Suppression d’un épisode + renumérotation complète
====================================================================
Prend en charge l’arborescence fixe :
```
/ data/chunk-000/episode_000000.parquet
/ videos/chunk-000/observation.images.(camD|webcam)/episode_000000.mp4
/ images/observation.images.(camD|webcam)/episode_000000/frame_000000.png
/ meta/episodes_stats.jsonl  (JSON Lines)
/ meta/episodes.jsonl        (JSON Lines ou liste JSON)
/ meta/episodes.json         (dict ou liste)
/ meta/info.json             (compteurs globaux)
```

Le script :
* Supprime les fichiers/dirs d’un épisode donné.
* Renomme **répertoires** images/…/`episode_XXXXXX` et vidéos `.mp4`.
* Met à jour Parquet (`episode_index`), side‑cars JSON/CSV.
* Met à jour `episodes_stats.jsonl`, `episodes.jsonl`, `episodes.json`.
* Ajuste `total_episodes`, `total_frames`, `total_videos`, et `splits.train` dans `meta/info.json`.

--- Usage ---
```bash
python delete_episode.py /path/to/dataset  32  --verbose
```
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

PAD = 6
STEM_RE = re.compile(r"^episode_(\d{6})$")
PATCH_KEYS = {"episode_index", "index"}

# ───────── helpers ─────────

def iter_files(root: Path) -> Iterable[Path]:
    yield from (p for p in root.rglob("*") if p.is_file())


def iter_dirs(root: Path) -> Iterable[Path]:
    yield from (d for d in root.rglob("*") if d.is_dir())


def ep_id_from_stem(name: str) -> Optional[int]:
    m = STEM_RE.match(name)
    return int(m.group(1)) if m else None


def episode_id(path: Path) -> Optional[int]:
    return ep_id_from_stem(path.stem)


def _add(val: Any, off: int):
    if isinstance(val, int):
        return val + off
    if isinstance(val, list):
        return [_add(x, off) for x in val]
    return val


def _patch(obj: Any, off: int):
    if isinstance(obj, dict):
        return {k: (_add(v, off) if k in PATCH_KEYS else _patch(v, off)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_patch(x, off) for x in obj]
    return obj

# ───────── file patchers ─────────

def patch_parquet(path: Path, off: int) -> int:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return 0
    nrows = len(df)
    if "episode_index" in df.columns and pd.api.types.is_integer_dtype(df["episode_index"]):
        df["episode_index"] += off
        df.to_parquet(path, index=False)
    return nrows


def patch_json(path: Path, off: int):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    patched = _patch(data, off)
    path.write_text(json.dumps(patched, indent=2))


def patch_csv(path: Path, off: int):
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] += off
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda s: str(int(s) + off) if str(s).isdigit() else s)
    df.to_csv(path, index=False)

# ───────── JSONL helpers ─────────

def rewrite_jsonl(path: Path, ep_id: int, off: int, verbose: bool):
    new_lines = []
    with path.open() as fh:
        txt = fh.read().strip()
    if txt.startswith("[") and txt.endswith("]"):
        # file actually contains a JSON array
        try:
            data = json.loads(txt)
            if isinstance(data, list):
                out = []
                for obj in data:
                    if isinstance(obj, dict) and obj.get("episode_index") == ep_id:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get("episode_index"), int) and obj["episode_index"] > ep_id:
                        obj = _patch(obj, off)
                    out.append(obj)
                path.write_text(json.dumps(out, indent=2))
                if verbose:
                    print(path.name, "JSON list updated")
                return
        except Exception:
            pass  # fall through to line‑wise processing

    # line‑wise JSONL
    for line in txt.splitlines():
        try:
            obj = json.loads(line)
            idx = obj.get("episode_index")
            if idx == ep_id:
                continue
            if isinstance(idx, int) and idx > ep_id:
                obj = _patch(obj, off)
            new_lines.append(json.dumps(obj))
        except Exception:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n")
    if verbose:
        print(path.name, "JSONL updated")

# ───────── main routine ─────────

def delete_episode(ds_dir: Path, ep_id: int, *, verbose: bool = False):
    ds = ds_dir.resolve()
    if not ds.is_dir():
        raise FileNotFoundError(ds)

    tgt_stem = f"episode_{ep_id:0{PAD}d}"
    frames_removed = 0
    videos_removed = 0

    # 1) Delete Parquet / videos / images dirs
    for path in list(iter_files(ds)):
        if path.stem != tgt_stem:
            continue
        if verbose:
            print("delete", path)
        if path.suffix.lower() == ".parquet":
            frames_removed += patch_parquet(path, 0)  # count rows before removal
        elif path.suffix.lower() == ".mp4":
            videos_removed += 1
        path.unlink(missing_ok=True)

    # delete image frame directory if exists
    for d in list(iter_dirs(ds)):
        if d.name == tgt_stem:
            if verbose:
                print("remove dir", d)
            shutil.rmtree(d, ignore_errors=True)

    # 2) Shift higher episode files and dirs
    higher_ids = sorted(id_ for id_ in {episode_id(p) for p in iter_files(ds)} if id_ and id_ > ep_id)
    for old in higher_ids:
        new = old - 1
        old_stem = f"episode_{old:0{PAD}d}"
        new_stem = f"episode_{new:0{PAD}d}"

        # rename files
        for f in list(iter_files(ds)):
            if f.stem != old_stem:
                continue
            new_file = f.with_name(new_stem + f.suffix)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), new_file)
            suf = new_file.suffix.lower()
            if suf == ".parquet":
                patch_parquet(new_file, -1)
            elif suf == ".json":
                patch_json(new_file, -1)
            elif suf in {".csv", ".tsv"}:
                patch_csv(new_file, -1)

        # rename image dirs (camD & webcam)
        for img_dir in list(iter_dirs(ds)):
            if img_dir.name == old_stem:
                new_dir = img_dir.with_name(new_stem)
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_dir), new_dir)

    # 3) Update JSONL metadata files
    for name in ["meta/episodes_stats.jsonl", "meta/episodes.jsonl"]:
        path = ds / name
        if path.exists():
            rewrite_jsonl(path, ep_id, -1, verbose)

    # 5) Update meta/info.json counts) Update meta/info.json counts
    info = ds / "meta/info.json"
    if info.exists():
        try:
            meta = json.loads(info.read_text())
        except Exception:
            meta = None
        if isinstance(meta, dict):
            if isinstance(meta.get("total_episodes"), int):
                meta["total_episodes"] = max(0, meta["total_episodes"] - 1)
            if isinstance(meta.get("total_frames"), int):
                meta["total_frames"] = max(0, meta["total_frames"] - frames_removed)
            if isinstance(meta.get("total_videos"), int):
                meta["total_videos"] = max(0, meta["total_videos"] - videos_removed)
            if isinstance(meta.get("splits"), dict) and isinstance(meta["splits"].get("train"), str):
                start, _, end = meta["splits"]["train"].partition(":")
                try:
                    meta["splits"]["train"] = f"{start}:{int(end)-1}"
                except Exception:
                    pass
            info.write_text(json.dumps(meta, indent=2))
            if verbose:
                print("meta/info.json updated")

    if verbose:
        print("✅ Episode", ep_id, "deleted and dataset renumbered")

# ───────── CLI ─────────

def _cli():
    ap = argparse.ArgumentParser(description="Delete an episode and renumber dataset files & metadata")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("episode", type=int)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    delete_episode(args.dataset, args.episode, verbose=args.verbose)


if __name__ == "__main__":
    _cli()
