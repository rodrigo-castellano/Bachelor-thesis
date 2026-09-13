# Identifying cosmic rays with machine learning

Bachelor thesis, Physics BSc, Universidad de Granada.

Cosmic rays reach the atmosphere and produce showers of secondary particles.
The problem is to work backwards from what reaches the ground, or from the
light the shower emits, to the primary particle that caused it — gamma,
proton, electron, iron and others.

The thesis attacks this twice, with different data and different methods.

The full write-up is [`TFG.pdf`](TFG.pdf).

---

## Part 1 — ground-level particle counts

**[`part-1-corsika/`](part-1-corsika/)**

Data simulated with **CORSIKA**, the air-shower simulator used by the Pierre
Auger Observatory. Each event is described by a handful of physical
quantities: the total number of particles at ground level, how many are
muons, how many are electromagnetic, the zenith angle, and the energy.

Seven supervised classifiers are built on these features and compared:

| | |
|---|---|
| [`knn.ipynb`](part-1-corsika/knn.ipynb) | k-nearest neighbours, including a from-scratch implementation |
| [`svm.ipynb`](part-1-corsika/svm.ipynb) | support vector machine |
| [`decision-tree.ipynb`](part-1-corsika/decision-tree.ipynb) | decision tree |
| [`random-forest.ipynb`](part-1-corsika/random-forest.ipynb) | random forest |
| [`xgboost.ipynb`](part-1-corsika/xgboost.ipynb) | gradient boosting |
| [`logistic-regression.ipynb`](part-1-corsika/logistic-regression.ipynb) | logistic regression |
| [`neural-network.ipynb`](part-1-corsika/neural-network.ipynb) | fully connected neural network |

## Part 2 — Cherenkov telescope images

**[`part-2-cta/`](part-2-cta/)**

Data simulated for the **Cherenkov Telescope Array**. A shower emits Cherenkov
light, and the telescope camera records it as an intensity map over a
hexagonal pixel grid — so each event is an image rather than a feature vector,
and the natural tool is a convolutional network.

The pipeline is a Python package driven from the command line:

```
part-2-cta/
  main.py        entry point
  CTA/
    dataset.py   loading, preprocessing, image export, gamma shifting
    model.py     architecture, training, testing, grid search, plots
    visualize.py scatter and histogram plots of the raw data
```

```sh
python main.py --data          # preprocess the simulated events into arrays
python main.py --CNN           # train the convolutional network
python main.py --CNN --load_model
python main.py --NN            # the fully connected baseline
python main.py --predict       # classify and plot the predictions
```

`Model` takes the number of classes as a parameter, so the same code covers
the binary gamma-vs-background problems and the full seven-particle one.

### Notebooks

[`part-2-cta/notebooks/`](part-2-cta/notebooks/) holds the exploratory work
the package was distilled from. They are kept because they carry their
results — accuracy figures, confusion matrices and training curves from the
runs — which the package code does not:

- `data-wrangling` — turning the hexagonal camera grid into arrays a CNN can
  consume
- `gamma-vs-proton`, `gamma-vs-electron`, `gamma-vs-all` — separating gamma
  showers from each background in turn
- `general-classification` and `general-classification-v2` — the full
  multi-class problem
- `all-particles-transfer-learning` — the largest notebook here, and the one
  piece of work not in the package: rather than training from scratch it
  adapts ImageNet networks (MobileNet, Xception, InceptionResNetV2) to the
  telescope images, tiling the single-channel data to three channels and
  resizing to 96×96, then tunes them with a TensorBoard HParams sweep

## Extras

**[`extras/`](extras/)** — related work that sits outside the thesis proper:
a Monte Carlo simulation, an MNIST warm-up, and
[`quantum-ml/`](extras/quantum-ml/), an exploration of quantum machine
learning with Qiskit, PyTorch and TensorFlow Quantum.

---

```
part-1-corsika/   seven classifiers on CORSIKA ground-level data
part-2-cta/       CNNs on Cherenkov Telescope Array images
  cnn/            the same models as a Python package
extras/           Monte Carlo, MNIST warm-up, quantum ML
TFG.pdf           the thesis
```
