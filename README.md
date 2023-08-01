# LEXNet: A Lightweight, Efficient and Explainable-by-Design Convolutional Neural Network for Internet Traffic Classification
This repository contains the Python implementation of LEXNet as described in 
the paper [A Lightweight, Efficient and Explainable-by-Design Convolutional Neural Network for Internet Traffic Classification](https://arxiv.org/pdf/2202.05535).

<p align="center">
<img src="/images/class_prototypes_application1.png" width="60%">
</p>

## Requirements
LEXNet has been implemented in Python 3 with the following packages:
* numpy
* opencv
* pandas
* pytorch with cuda
* pyyaml
* scikit-learn
* scipy

## Usage
Run `main.py` with the following argument:

* configuration: name of the configuration file (string)

```
python main.py --config configuration/config.yml
```

## Citation
```
@article{Fauvel23LEXNet,
  author = {Fauvel, K. and F. Chen and D. Rossi},
  title = {{A Lightweight, Efficient and Explainable-by-Design Convolutional Neural Network for Internet Traffic Classification}},
  journal = {Proceedings of the 29th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year = {2023},
}
```
