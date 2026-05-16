# Experiment Inputs

This directory is the experiment-facing input surface.

```text
lrobench/
  data.jsonl
  embeddings.npy
  embeddings.ids.jsonl
  embeddings.meta.json

fever/
  data.jsonl
  embeddings.npy          # not present yet
```

`data.jsonl` must use `id/query/document/groundtruth`.

`embeddings.npy` must have a matching `embeddings.ids.jsonl` sidecar and must cover every `id` in `data.jsonl`.

Current status:

- `lrobench` is ready for CGSD with `DIM=2560`.
- `fever` has the JSONL data, but still needs full pair embeddings before `cgsd_prepare.py` can pass.
