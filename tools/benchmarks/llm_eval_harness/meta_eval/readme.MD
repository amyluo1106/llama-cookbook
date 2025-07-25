add `vllm_blendkv.py` to `lm_eval/models` folder

add `vllm_blendkv` to imports in `lm_eval/models/__init__.py`

install `lmcache` from https://github.com/amyluo1106/LMCache/tree/benchmarking

for different recompute ratios/check_layers, edit `self.common_metadata` in `lmcache/v1/compute/blend/blender.py`