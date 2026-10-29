"""Extract normalized face embeddings from folders organized by identity."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def normalize_identity(name: str) -> str:
    """Normalize common class-folder prefixes so source/external labels can match."""
    label = re.sub(r"^pins[_\s]+", "", name.strip(), flags=re.IGNORECASE)
    label = re.sub(r"[_-]+", " ", label)
    return " ".join(label.casefold().split())

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing one subfolder per identity")
    parser.add_argument("--output-dir", default="data/features")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto",
                         help="auto uses CUDAExecutionProvider if installed; otherwise CPU")
    return parser.parse_args()

def main():
    args = parse_args()
    import cv2
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")
    available = set(ort.get_available_providers())
    if args.provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise SystemExit(f"CUDA provider requested but unavailable. Installed providers: {sorted(available)}")
    use_cuda = args.provider == "cuda" or (args.provider == "auto" and "CUDAExecutionProvider" in available)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=args.model, providers=providers)
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=(args.det_size, args.det_size))

    features, labels, skipped = [], [], []
    identity_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(identity_dirs) < 2:
        raise SystemExit("Expected at least two identity subfolders; inspect the extracted directory structure first")
    for identity_dir in identity_dirs:
        label = normalize_identity(identity_dir.name)
        if not label:
            skipped.append({"path": str(identity_dir), "reason": "empty identity label"})
            continue
        for image_path in sorted(identity_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped.append({"path": str(image_path), "reason": "unreadable image"})
                continue
            faces = app.get(image)
            if not faces:
                skipped.append({"path": str(image_path), "reason": "no face detected"})
                continue
            face = max(faces, key=lambda item: float(item.det_score))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                skipped.append({"path": str(image_path), "reason": "invalid embedding"})
                continue
            features.append(embedding)
            labels.append(label)
    if len(features) < 4 or len(set(labels)) < 2:
        raise SystemExit("Need at least four detected faces across at least two identity folders")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", np.stack(features).astype(np.float32), allow_pickle=False)
    np.save(output_dir / "y.npy", np.asarray(labels, dtype=str), allow_pickle=False)
    (output_dir / "extraction_summary.json").write_text(json.dumps({
        "embedding_model": args.model, "providers": providers, "embedding_dimension": int(features[0].shape[0]),
        "embedded_images": len(features), "identity_count": len(set(labels)),
        "class_counts": {label: labels.count(label) for label in sorted(set(labels))},
        "skipped_count": len(skipped), "skipped_examples": skipped[:100]
    }, indent=2), encoding="utf-8")
    print(f"Embedded {len(features)} images across {len(set(labels))} identities; skipped {len(skipped)}.")
    print(f"Providers requested: {providers}. Extraction summary: {output_dir / 'extraction_summary.json'}")

if __name__ == "__main__":
    main()
