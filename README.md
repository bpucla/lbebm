## Trajectory Prediction with Latent Space Energy-Based Model

## Requirements:
```
torch==1.4.0
scipy==1.4.1
numpy==1.18.5
pandas==0.24.2
```


## Environment
Code is tested on Ubuntu 16.04, Python 3.5, NVIDIA 1080 Ti, CUDA 10.1

## Dataset
SDD is available in ```dataset/sdd```.

Other datasets are in google drive: https://drive.google.com/drive/folders/1TaRvS_nA90KGKVgThIP-IIIbIKdmuIKe?usp=sharing

You can download and put them in ```dataset```

## Pretrained Models
A pretrained model for SDD is available in ```saved_models```. 

More models are in google drive: https://drive.google.com/drive/folders/1Pu7ggwIDknk5Xs-MfKQHNnH0-UdMa1E3?usp=sharing

You can download them and put them in ```saved_models```.


## Run Trained Models
```
# SDD
python lbebm_sdd.py

# ETH
python lbebm_eth.py
```

## Train New Models

You can train new models by simply removing the ```model_path```.

Change ```--dataset_name``` to train other datasets in eth-ucy.

```
# SDD
python lbebm_sdd.py --model_path ''

# ETH
python lbebm_eth.py --model_path ''
```
## Contact
Please contact Bo Pang (bopang@g.ucla.edu) for questions.
