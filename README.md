<div align="center">

<p align="center">
  <img src="./logo-hnet.png" style="object-fit:contain; width:250px; height:250px;">
</p>

<div align='center'>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.4-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐙 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://docs.ray.io/en/latest/tune/"><img alt="Ray" src="https://img.shields.io/badge/Ray 2.40-blue?style=for-the-badge&logo=ray&logoColor=cyan"></a>
<a href="https://github.com/aimhubio/aim"><img alt="Pytest" src="https://img.shields.io/badge/Pytest 8.3-gray?style=for-the-badge&logo=pytest&logoColor=green"></a>
</div>

Differentiable assignation problem resolution by deep learning method.

</div>

## Requirements
```pip install -r requirements.txt``` 

## Getting Started
 
* `generate_hnet_training_data.py` generates synthetic distance matrices and association matrices. 
* Then `run.py` to train Hnet with the generated data.

---

> Sharath Adavanne*, Archontis Politis* and Tuomas Virtanen, "[Differentiable Tracking-Based Training of Deep Learning Sound Source Localizers](https://arxiv.org/pdf/2111.00030.pdf)" in the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2021)

## License
The repository is licensed under the [TAU License](LICENSE.md).
