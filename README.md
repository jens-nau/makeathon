# makeathon
Revolutionizing medical imaging.

## Training


### ACL Multi-Plane

```python
python ResNet18-Spatial-Attention/Multi-Plane/train.py -t acl --prefix_name multi_acl --mod mp1 -d /home/jensnau/datasets/MRNet --epochs 20
```

### ACL Axial

```python
python ResNet18-Spatial-Attention/Single_plane/train.py -t acl -p axial --prefix_name single_acl_axial -d /home/jensnau/datasets/MRNet --epochs 7
```

## License
[MIT](https://choosealicense.com/licenses/mit/)