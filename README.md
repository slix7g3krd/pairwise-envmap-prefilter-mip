# pairwise-envmap-prefilter-mip

## 领域

游戏、图形与多媒体

## 主题

环境贴图预过滤

## 说明

GGX 卷积、mip 映射与立方体接缝。

初始状态下 `graphics/envmap_prefilter_mip.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
