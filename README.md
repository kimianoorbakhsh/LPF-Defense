# LPF-Defense

The Code and Data for the paper ["LPF-Defense: 3D Adversarial Defense
based on Frequency Analysis"](https://arxiv.org/abs/2202.11287), by Hanieh Naderi, Kimia Noorbakhsh*, Arian Etemadi*, and Shohreh Kasaei.

# Prerequisites
This code depends on the following packages:

 1. [pyshtools](https://shtools.github.io/SHTOOLS/using-with-python.html)
 2. [PyTorch](https://pytorch.org/)
 3. `torchvision`
 4. `numpy`
 5. `matplotlib`

# Code Structure
The instructions on running each part of the code is explained in the README file of each folder separately. 

# Data
The ModelNet40 data we used in our paper is uploaded in folder `model/Data/`. The scanobjectnn Dataset and the subset of ShapeNet Dataset used in the paper are available at [https://hkust-vgd.github.io/scanobjectnn/](https://hkust-vgd.github.io/scanobjectnn/) and [https://github.com/thuml/Metasets](https://github.com/thuml/Metasets).

## Citation
If you use parts of the code in this repository for your own research, please consider citing:

```
@article{naderi2022lpfdefense,
        title={LPF-Defense: 3D Adversarial Defense based on Frequency Analysis}, 
        author={Hanieh Naderi and Arian Etemadi and Kimia Noorbakhsh and Shohreh Kasaei},
        journal={arXiv preprint arXiv:2202.11287},
        year={2022}
}
```
