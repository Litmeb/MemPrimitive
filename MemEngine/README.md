## Local memengine patch notes

This folder contains notebooks that use the `memengine` Python package installed in the current conda environment.

### Patch: make `LLMJudge` score parsing robust (JSON / numeric / fallback)

- **Why**: Some API models (especially free/preview ones) occasionally return explanatory text even when asked to output only a number.
  The upstream `memengine.function.Judge.LLMJudge` used `float(eval(res))` without exception handling, which can crash with
  `SyntaxError` when the output is not a Python numeric expression.
- **What changed**: Patched the installed package file:
  - `d:\Anaconda\envs\py312pt291cu128\Lib\site-packages\memengine\function\Judge.py`
  - `LLMJudge.__post_scale__` now tries, in order:
    1) parse strict JSON like `{"score": 7}` (also accepts `{"value": 7}`), including embedded `{...}` blocks
    2) guarded legacy `float(eval(text))` (so `"7/10"` still works, but never crashes)
    3) regex fallback: first number in the text
    4) default fallback score (configurable via `default_score`, defaults to `5.0`)

### Important

- This is a **local environment patch** to `site-packages`. Reinstalling/upgrading `memengine` or recreating the conda env may overwrite it.
- If you want this fix to be reproducible across machines, consider vendoring/forking `memengine` or adding a small wrapper judge in your project code.

