"""Extract normalized face embeddings from folders organized by identity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Folder containing one subfolder per identity")
    p.add_argument("--output-dir", default="data/features")
    p.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    p.add_argument("--det-size", type=int, default=640)
    return p.parse_args()

def main():
    args = parse_args()
    import cv2
    from insightface.app import FaceAnalysis
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")
    app = FaceAnalysis(name=args.model, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))
    features, labels, skipped = [], [], []
    identity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not identity_dirs:
        raise SystemExit("Expected identity subfolders but none were found")
    for identity in identity_dirs:
        for path in sorted(identity.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                skipped.append({"path": str(path), "reason": "unreadable image"}); continue
            faces = app.get(image)
            if not faces:
                skipped.append({"path": str(path), "reason": "no face detected"}); continue
            face = max(faces, key=lambda item: float(item.det_score))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                skipped.append({"path": str(path), "reason": "invalid embedding"}); continue
            features.append(embedding); labels.append(identity.name)
    if len(features) < 4 or len(set(labels)) < 2:
        raise SystemExit("Need at least four detected faces from at least two identity folders")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", np.stack(features).astype(np.float32), allow_pickle=False)
    np.save(out / "y.npy", np.asarray(labels, dtype=str), allow_pickle=False)
    (out / "extraction_summary.json").write_text(json.dumps({
        "embedding_model": args.model, "embedding_dimension": int(features[0].shape[0]),
        "embedded_images": len(features), "identity_count": len(set(labels)),
        "skipped_count": len(skipped), "skipped_examples": skipped[:100]
    }, indent=2), encoding="utf-8")
    print(f"Embedded {len(features)} images over {len(set(labels))} identities; skipped {len(skipped)}.")

if __name__ == "__main__":
    main()
