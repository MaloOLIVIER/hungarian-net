<div align="center">
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.4-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐙 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://docs.ray.io/en/latest/tune/"><img alt="Ray" src="https://img.shields.io/badge/Ray 2.40-blue?style=for-the-badge&logo=ray&logoColor=cyan"></a>
<a href="https://github.com/aimhubio/aim"><img alt="Pytest" src="https://img.shields.io/badge/Pytest 8.3-gray?style=for-the-badge&logo=pytest&logoColor=green"></a>
</div>

# Hungarian Network (Hnet)

The Hungarian Network (Hnet) is the deep-learning-based implementation of the popular [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) that helps solve the assignment problem. Implementing this algorithm using a DNN allows us to integrate it with other deep-learning tasks that require permutation invariant training (PIT) and train them in a completely differentiable manner without the need of PIT. Some examples of deep-learning tasks that required PIT are multi-source localization (DOA estimation), and source separation. One such implementation of [multi-source localization and tracking using Hnet is available here.](https://github.com/sharathadavanne/doa-net) If you want to read more about [generic approaches to sound localizatoin and tracking then check here](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-and-tracking). If you are using this repo in any format, then please consider citing the following paper. 

> Sharath Adavanne*, Archontis Politis* and Tuomas Virtanen, "[Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers](https://arxiv.org/pdf/2111.00030.pdf)" in the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2021)

## METHOD
 
The Hnet architecture is as shown below. The input is the pair-wise distance matrix **D**, that is mapped to the association matrix **A**. 

<p align="center">
   <img src="https://github.com/sharathadavanne/hungarian-net/blob/master/images/HungarianNet.png" width="400" title="Hungarian Network (Hnet) Architecture">
</p>

The output **A** of Hnet is in the continuous range of [0 1], where indices with value 1 suggests the true associations. 


## Getting Started

This repository consists of three Python scripts 
* The `generate_hnet_training_data.py` is a standalone script that generates data to train the Hnet for multi-source localization as described in the cited paper above. 
* The `train_hnet.py` script consists of all the training, model, and feature parameters. 
* `visualize_hnet_results.py` script to visualize the Hnet output
 

### Prerequisites

The provided codebase has been tested on Python 3.12 and Torch 2.5


### Training the Hnet

In order to quickly train Hnet follow the steps below.
* First, create the training data by running `generate_hnet_training_data.py`. It generates pairs of distance matrix *D* and corresponding associate matrix *A*. These distance matrices are currently set to be angular distance between two polar coordinates on a unit-sphere. You can modify this code to the range of distances corresponding to your task. Read the comments to understand more details.

```
python3 generate_hnet_training_data.py
```

* You can now train the Hnet using default parameters. It can train on quickly even on a laptop CPU. GPU is not mandatory.
```
python3 train_hnet.py
```

* You can visualize the output of Hnet with the following command. 
```
python3 visualize_hnet_results.py
```

## License
The repository is licensed under the [TAU License](LICENSE.md).
