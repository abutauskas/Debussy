# Data (not stored in this repository)

The scripts expect these files in `data/`. Several datasets are gated, so they are downloaded by you, never committed.

| File | Source | How |
|---|---|---|
| `w2c_when2call_train_pref.jsonl`, `w2c_when2call_test_mcq.jsonl` | `nvidia/When2Call` | download from Hugging Face (public) |
| `wg_wildguard_train.parquet`, `wg_wildguard_test.parquet` | `allenai/wildguardmix` | **gated**: accept the terms, then download with your `HF_TOKEN` |
| `hle/hle_test.parquet` | `cais/hle` | **gated** (manual approval); evaluation only |
| `mmlupro_test.parquet` | `TIGER-Lab/MMLU-Pro` | public |
| `omni_public/omni_public_v1_1.csv` | `ArtificialAnalysis/AA-Omniscience-Public-v1.1` | public (Apache-2.0) |

Example (Hugging Face token in the `HF_TOKEN` environment variable; never commit it):

```bash
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data/wg_wildguard_test.parquet \
  https://huggingface.co/datasets/allenai/wildguardmix/resolve/main/test/wildguard_test.parquet
```

Per-question result files (`data/*.json`) contain test questions' probabilities and labels; keep them local.
