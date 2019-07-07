# 目的

- パッチをPCAにかけて分布が一致することを示す

## 手順

### 1. パッチの抽出をする

```
python extract_patch.py hogehoge
```

### 2. 推定をおこなう

```
python ouput_inferred_patches.py hogehoge
```

### 3. PCAをかける

```
python apply_pca_to_patches.py hogehoge
```
