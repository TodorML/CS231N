# CS231n — Deep Learning for Computer Vision

My solutions to the first two assignments of Stanford's **CS231n** (Spring 2024). Core layers, classifiers, and optimizers are implemented **from scratch in NumPy** — the point is to write the forward/backward passes by hand before reaching for autograd.

Course: [cs231n.github.io](https://cs231n.github.io/) · Python 3.11

## Assignment 1 — image classification & shallow models
CIFAR-10 pipeline from raw pixels up to a small neural net, with an emphasis on vectorized NumPy.

- `knn.ipynb` — k-Nearest Neighbor (two-loop → vectorized distances; cross-validation for `k`)
- `svm.ipynb` — multiclass SVM (hinge) loss + analytic gradient
- `softmax.ipynb` — numerically stable softmax / cross-entropy classifier
- `two_layer_net.ipynb` — two-layer net with a modular forward/backward design
- `features.ipynb` — same classifiers on HOG + color-histogram features vs. raw pixels

## Assignment 2 — deep nets & CNNs
Deeper networks via a modular layer API (each layer = `forward` + `backward`), then conv nets and PyTorch.

- `FullyConnectedNets.ipynb` — arbitrarily deep FC nets; SGD+Momentum, RMSProp, Adam
- `BatchNormalization.ipynb` — batch norm & layer norm, forward/backward
- `Dropout.ipynb` — inverted dropout as a regularizer
- `ConvolutionalNetworks.ipynb` — naive conv/max-pool, spatial batch/group norm, 3-layer ConvNet (+ fast im2col/Cython layers)
- `PyTorch.ipynb` — the same ideas in PyTorch on CIFAR-10

## Running locally
Designed for Colab (the supported path). Locally:

```bash
cd .../assignments/assignment2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # assignment1 just needs numpy/scipy/matplotlib/jupyter
cd cs231n/datasets && ./get_datasets.sh && cd ../..
python setup.py build_ext --inplace      # assignment2: build fast conv layers
jupyter notebook
```

The pinned versions in `assignment2/requirements.txt` track the original course env and may need relaxing on Python 3.11 - if so, install current NumPy, SciPy, Matplotlib, Jupyter, Cython, and PyTorch.

## Notes
All starter code and assignment design are by the Stanford CS231n staff; this repo holds my solutions to the `# TODO` sections. If you're taking the course, use it as a reference and write your own, the Honor Code applies.
