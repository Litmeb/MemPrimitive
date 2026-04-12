---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: dataset
    dtype: string
  - name: dialoug_id
    dtype: int64
  - name: session_id
    dtype: int64
  - name: persona1
    sequence: string
  - name: persona2
    sequence: string
  - name: dialogue
    sequence: string
  - name: speaker
    sequence: string
  splits:
  - name: train
    num_bytes: 30863868
    num_examples: 17940
  - name: validation
    num_bytes: 6329337
    num_examples: 3000
  - name: test
    num_bytes: 5867348
    num_examples: 2505
  download_size: 0
  dataset_size: 43060553
---
# Dataset Card for "multi_session_chat"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)