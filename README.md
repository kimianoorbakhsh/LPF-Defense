# LPF-Defense

The Code and Data for the paper ["LPF-Defense: 3D Adversarial Defense
based on Frequency Analysis"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271388), by Hanieh Naderi, Kimia Noorbakhsh*, Arian Etemadi*, and Shohreh Kasaei.

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
@article{10.1371/journal.pone.0271388,
    doi = {10.1371/journal.pone.0271388},
    author = {Naderi, Hanieh AND Noorbakhsh, Kimia AND Etemadi, Arian AND Kasaei, Shohreh},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {LPF-Defense: 3D adversarial defense based on frequency analysis},
    year = {2023},
    month = {02},
    volume = {18},
    url = {https://doi.org/10.1371/journal.pone.0271388},
    pages = {1-19},
    number = {2},
}
```
