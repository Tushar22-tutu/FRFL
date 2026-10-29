# FRFL — Federated Face Recognition

Research pipeline for federated classification over precomputed face embeddings. It includes an MLP classifier, FedProx local training, a FedAvg baseline, L2 drift-weighted aggregation, external evaluation utilities, tests, and GitHub Actions.

Important: automated tests use synthetic data and do not establish real face-recognition performance. Report real metrics only after running documented experiments on held-out data.

## Setup
Python 3.10+. Install dependencies with `pip install -r requirements.txt` and `pip install -e .`.

Optional image extraction dependencies: `pip install -r requirements-vision.txt`. Organize images into folders by identity, then run `python scripts/extract_embeddings_insightface.py --input-dir path/to/dataset --output-dir data/features`.

## Train
Run `python train.py --x data/features/X.npy --y data/features/y.npy --clients 3 --rounds 5 --local-epochs 3`. Outputs include a JSON report and checkpoints for both aggregation methods.

## Evaluate A/B/C
Extract each updated-sample set separately using the same encoder. Keep them out of training if measuring appearance-shift robustness. Evaluate with `python evaluate_external.py --model outputs/drift_weighted.pt --x data/updated_A/X.npy --y data/updated_A/y.npy --output outputs/eval_A.json`.

## Tests
Run `python -m pytest -v`. CI runs the same test suite on pushes and pull requests.

## Limitations and privacy
Drift weighting is an experimental heuristic, not an accuracy or privacy guarantee. Prefer identity/session-disjoint test splits, and claim improvement only when comparing methods on the same held-out data. Face images and embeddings are biometric data; this public repository intentionally excludes the large dataset archives and generated features. Keep them local unless you have appropriate rights and consent to publish them.


## Run the real-data experiment in Colab

Open the committed notebook: [FRFL_Real_Data_Experiment.ipynb](https://colab.research.google.com/github/Tushar22-tutu/FRFL/blob/main/notebooks/FRFL_Real_Data_Experiment.ipynb). Upload the source dataset ZIP and optional updated-sample A/B/C ZIPs to a private Google Drive folder named FRFL_data using the exact filenames shown in the notebook. Select a GPU runtime for faster embedding extraction, then run cells from top to bottom.

The notebook extracts embeddings locally in Colab, runs the test suite, compares FedAvg and drift-weighted aggregation on a shared held-out split, and attempts external A/B/C evaluation only when identity folders can be detected. If an updated-sample archive is flat or labels cannot be matched, it skips those metrics rather than assigning guessed labels. The notebook has been added to the repository, but no real-data results have been run or committed from this chat environment.

Do not upload the biometric dataset ZIPs, raw face images, or derived embeddings to this public repository.
