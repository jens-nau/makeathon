# makeathon
Revolutionizing medical imaging.

## Training


### ACL Multi-Plane

```python
python ResNet18-Spatial-Attention/Multi-Plane/train.py -t acl --prefix_name multi_acl --mod mp1 -d /home/jensnaumann/disk-1/MRNet/ --epochs 20
```

### ACL Axial

#### Train

```python
python Single_plane/train.py -t acl -p axial --prefix_name acl_axial_early_att -d /home/jensnaumann/disk-1/MRNet/ --epochs 7
```

#### Evaluate

```python
python Single_plane/output.py -t acl -p axial --model_name model_fold0_test_acl_axial_val_auc_0.9221_train_auc_0.9057_epoch_7.pth -d /home/jensnaumann/disk-1/MRNet/ -md /home/jensnaumann/makeathon/models/
```

## License
[MIT](https://choosealicense.com/licenses/mit/)