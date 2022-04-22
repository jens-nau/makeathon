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
python Single_plane/train.py -t acl -p axial --prefix_name test -d /home/jensnaumann/disk-1/MRNet/ --epochs 7
```

#### Evaluate

```python
python ResNet18-Spatial-Attention/Single_plane/evaluate.py -t acl -p axial --model_name model_fold0_single_acl_axial_acl_axial_val_auc_0.9297_train_auc_0.8941_epoch_5.pth -d /home/jensnaumann/disk-1/MRNet/ -md /home/jensnaumann/makeathon/models/
```

## License
[MIT](https://choosealicense.com/licenses/mit/)