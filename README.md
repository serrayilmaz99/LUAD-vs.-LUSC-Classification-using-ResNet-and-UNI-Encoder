This project implements an end-to-end deep learning pipeline for classifying lung adenocarcinoma (LUAD) and lung squamous cell carcinoma (LUSC) from H&E-stained whole-slide images (WSIs) using the TransMIL framework. The project focuses on weakly supervised slide-level classification using patch-based representations. Additional visualization methods for the transformer-based model are implemented in `create_heatmaps.py`.

Two different feature extraction approaches were evaluated:
ResNet-50 (CNN-based encoder)
UNI (Transformer-based foundation model)

We follow the TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification [NeurIPS 2021] framework.


### WSI Segmentation and Patching
```python
 preprocessing/create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```

### Feature Extraction
```python
CUDA_VISIBLE_DEVICES=0,1 python preprocessing/extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

### Train

```python
python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```

### Test

```python
python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```







