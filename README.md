# makeathon
Revolutionizing medical imaging.

## Training


### ACL Multi-Plane

#### Train

```python
python Multi-Plane/train.py -t acl --prefix_name multi_acl --mod mp1 -d /home/jensnaumann/disk-1/MRNet/ --epochs 30
```

#### Evaluate

```python
python  Multi-Plane/output.py -t acl --model_name model_fold0_multi_acl_acl_val_auc_0.9007_train_auc_0.9550_epoch_10.pth -d /home/jensnaumann/disk-1/MRNet/ -md /home/jensnaumann/makeathon/models/
```

### Meniscus Multi-Plane

#### Train

```python
python Multi-Plane/train.py -t meniscus --prefix_name multi_meniscus --mod mp1 -d /home/jensnaumann/disk-1/MRNet/ --epochs 30
```

#### Evaluate

```python
python  Multi-Plane/output.py -t meniscus --model_name model_fold0_multi_acl_acl_val_auc_0.9007_train_auc_0.9550_epoch_10.pth -d /home/jensnaumann/disk-1/MRNet/ -md /home/jensnaumann/makeathon/models/
```


### ACL Axial

#### Train

```python
python Single_plane/train.py -t acl -p axial --prefix_name acl_axial_early_att -d /home/jensnaumann/disk-1/MRNet/ --epochs 20
```

#### Evaluate

```python
python Single_plane/output.py -t acl -p axial --model_name model_fold0_acl_axial_early_att_acl_axial_val_auc_0.9245_train_auc_0.9796_epoch_16.pth -d /home/jensnaumann/disk-1/MRNet/ -md /home/jensnaumann/makeathon/models/
```

## License
[MIT](https://choosealicense.com/licenses/mit/)