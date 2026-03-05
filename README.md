We follow the TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification [NeurIPS 2021] pipeline.


# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch

# Feature Extraction
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs

# Train

```python
python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```

# Test

```python
python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```
