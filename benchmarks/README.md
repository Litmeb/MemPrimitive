# Benchmarks

This directory stores benchmark data downloaded directly into the project.

- `MSC/`
  Source dataset: `nayohan/multi_session_chat` on Hugging Face.
  Files downloaded: dataset card plus train/validation/test parquet shards.
- `DMR/`
  Source dataset: `MemGPT/MSC-Self-Instruct` on Hugging Face.
  Files downloaded: dataset card plus `msc_self_instruct.jsonl`.
- `LongMemEval/`
  Source dataset: `xiaowu0162/longmemeval-cleaned` on Hugging Face.
  Files downloaded: dataset card plus `longmemeval_oracle.json`, `longmemeval_s_cleaned.json`, and `longmemeval_m_cleaned.json`.
- `LoCoMo/`
  Source repository: `snap-research/locomo` on GitHub.
  Files copied: repo `README`, `LICENSE`, `data/locomo10.json`, and `data/msc_personas_all.json`.

Notes:

- These are raw benchmark assets only. No local harness or preprocessing wrapper was added in this task.
- `LongMemEval/longmemeval_m_cleaned.json` is very large and takes about 2.2 GB on disk.
- `LoCoMo` is distributed under the repository's included `LICENSE.txt` and is not mirrored here beyond the core benchmark files.
