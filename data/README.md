# Local datasets and embeddings

Keep source face images and derived embeddings on private/local storage; they are excluded from Git. Training uses `data/features/X.npy` (float embeddings of shape N x D) and `data/features/y.npy` (identity labels of shape N, saved as strings). Generate them with `scripts/extract_embeddings_insightface.py`. Use the same detector, encoder and preprocessing across all data. Keep updated A/B/C samples separate when evaluating robustness. Do not commit biometric images or embeddings to this public repository.
