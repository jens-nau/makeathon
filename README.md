# MedApp | See more by doing less.
TUM.ai Makeathon - April 2022 | NetApp Challenge

## Pitch Deck

See [TUMai_AIPenguins_MedApp.pdf](TUMai_AIPenguins_MedApp.pdf)

## Miro

[Miro](https://miro.com/welcomeonboard/bHBtMEdjVGRDOGluTVVqa1BHVG9ucnZuYjJpUEM1ZFFKUWh2UGdlQVBtc1VzRG80OEE0NkQ5Y25iNnVlNUVHRXwzMDc0NDU3MzUyODAwODEzMjU0?share_link_id=784885959645)

## Screenshots

<img src="/images/Screenshot_1.png"/>
<img src="/images/Screenshot_2.png"/>
<img src="/images/Screenshot_3.png"/>

## Architecture

Our solution is based on a state-of-the-art image classification and regression approach that leverages spatial attention. This architecture enables both the classification of injuries and anomalies as well as the regression of anatomical features such as the tibial slope. The spatial attention layer not only improves the performance of the model, but also enables the creation of visual heat maps indicating where anomalies were found or, more generally, which parts of the image contributed to the neural network's decision.

The architecture is based on a ResNet-18 backbone extended by a spatial attention layer. The backbone has been pretrained on ImageNet and was finetuned on the specific tasks of injury / anomaly classification or feature regression.

<img src="/images/Original-ResNet-18-Architecture.png"/>

## Dataset

To fine-tune our model, we used the MRNet dataset from Stanford (https://stanfordmlgroup.github.io/competitions/mrnet/). In addition, we had a subset of this dataset manually labeled by an MRI expert to train for anatomical features which are not available by default in the dataset. 

## References

Our model and ideas are mainly based on the following research:
<ul>
  <li>https://arxiv.org/abs/2108.08136</li>
  <li>https://arxiv.org/abs/2010.01947</li>
  <li>https://arxiv.org/abs/1512.03385</li>
</ul>

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Visualization

See [dashboard.ipynb](/visualization/dashboard.ipynb)

### Training 

#### Train

```python
python Multi-Plane/train.py -t %FEATURE% --prefix_name multi_acl --mod mp1 -d %DATA-DIR% --epochs 30
```

#### Evaluate

```python
python  Multi-Plane/evaluate.py -t %FEATURE% --model_name %MODEL-NAME% -d %DATA-DIR% -md %MODEL-DIR%
```

#### Output

```python
python  Multi-Plane/output.py -t %FEATURE% --model_name %MODEL-NAME% -d %DATA-DIR% -md %MODEL-DIR%
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
