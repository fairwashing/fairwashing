# Fairwashing explanations with off-manifold detergent
This is the code-repository for the ICML 2020 paper __Fairwashing explanations with off-manifold detergent__. ([PDF](https://proceedings.icml.cc/static/paper_files/icml/2020/3760-Paper.pdf))

## Installation
Clone the repository and install it as a module:
```bash
$ git clone https://github.com/fairwashing/fairwashing
$ cd fairwashing
$ pip install .
```

## Pre-trained models
Our pre-trained models can be downloaded by executing:
```bash
$ bash script/download_models.sh
```
A folder `share/models` will be created in the current working directory.


The file-names are `<dataset>_<type>.pth`, where `<dataset>` is either cifar, mnist or fmnist.
The `<type>` field semantics are:
- `acc`: trained for accuracy (teacher model)
- `grad`: fairwashed for gradient
- `xgrad`: fairwashed for gradient times input
- `intgrad`: fairwashed for integrated gradients
- `lrp`: fairwashed for layerwise relevance propagation


The folder `defended` contains the same models, but where the fairwashing training was done with tangent-space
projection enabled.

## Running
The experiments consist of two parts:
- fairwashing models in [script/fairwashing.py](script/fairwashing.py)
- computing tangent-space projectors in [script/projectors.py](script/projectors.py)

both use the `fairwasher` module.


## Fairwashing models
Download our pre-trained models. The fairwashing CLI has two modes:
```bash
$ python script/fairwashing.py --help
Usage: fairwashing.py [OPTIONS] COMMAND [ARGS]...

  Command-line interface for training and evaluation of fairwashing models.

Options:
  --help  Show this message and exit.

Commands:
  evaluate  Evaluate how well student model reproduces target explanation,...
  train     Train a model to produce a target explanation while keeping the...
```

### Training
Following options are available for training:
```bash
$ python script/fairwashing.py train --help
Usage: fairwashing.py train [OPTIONS]

  Train a model to produce a target explanation while keeping the output
  similar to a teacher model.

Options:
  --dataroot DIRECTORY            path to store datasets
  --dataset [cifar|mnist|fmnist]
                                  dataset to train on
  --method [lrp|grad|xgrad|intgrad]
                                  explanation method
  --teacher-model FILE            model checkpoint
  --batch-size INTEGER            batch size of CIFAR10 images
  --num-workers INTEGER           number data loading workers
  --target FILE                   target heatmap
  --lr FLOAT                      learning rate
  --n-epochs INTEGER              number of epochs
  --checkpt FILE                  checkpoint to continue training from
  --alpha FLOAT                   weight factor between model and hm mse
  --out-dir DIRECTORY             directory to which checkpts will be saved
  --proj-file FILE                projector on tangent space
  --val-interval INTEGER          interval when validation is performed
  --optimizer [adam|sgd]          optimizer for training
  --loss [mse|ce]                 loss function between student and teacher output
  --lr-schedule CSINTS            epochs at which to divide lr by 10
  --help                          Show this message and exit.
```

Training on FashionMNIST can be initiated in the following way:
```bash
$ python script/fairwashing.py train \
    --dataroot "share/data" \
    --dataset "fmnist" \
    --method "grad" \
    --teacher-model "share/models/fmnist_acc.pth" \
    --batch-size 128 \
    --target "share/targets/42.png" \
    --lr 5e-5 \
    --n-epochs 100 \
    --checkpt "share/models/fmnist_acc.pth" \
    --alpha 0.8 \
    --out-dir "var/params" \
    --val-interval 5 \
    --optimizer "adam" \
    --loss "mse"
```
This will train the model on FashionMNIST for 100 epochs, and validate and save the parameters every 5 epochs.
The parameters after 100 epochs will be stored at `var/params/fairwashed_fmnist_grad_100.pth`.

### Evaluation
For evaluation, the following options are available:
```bash
$ python script/fairwashing.py evaluate --help
Usage: fairwashing.py evaluate [OPTIONS]

  Evaluate how well student model reproduces target explanation, and how
  similar it is to a teacher model.

Options:
  --dataroot PATH                 path to store datasets
  --dataset [cifar|mnist|fmnist]  dataset to train on
  --method [lrp|grad|xgrad|intgrad]
                                  explanation method
  --model FILE                    student model parameters
  --proj-file FILE                projector on tangent space
  --target FILE                   target heatmap
  --teacher FILE                  teacher to compare to
  --batch-size INTEGER            batch size
  --help                          Show this message and exit.
```

Evaluation of the previously fairwashed model, or here for our pre-trained version, can be done in the following way:
```bash
$ python script/fairwashing.py evaluate \
    --dataroot "share/data" \
    --dataset "fmnist" \
    --method "grad" \
    --model "share/models/fmnist_grad.pth" \
    --teacher-model "share/models/fmnist_acc.pth" \
    --target "share/targets/42.png" \
    --batch-size 128
```

## Computing TSP-Projectors
The tangent-space projector computation script `script/projectors` provides two modes:
```bash
$ python script/projectors.py --help
Usage: projectors.py [OPTIONS] COMMAND [ARGS]...

  A command-line interface to compute tangent-space projectors.

Options:
  --help  Show this message and exit.

Commands:
  ae      A command-line interface to train autoencoders for tangent-space...
  linear  A command-line interface to compute linear tangent-space...
```

### Linear TSP-Projectors
The linear tsp-projector command-line interface provides following options:
```bash
$ python script/projectors.py linear --help
Usage: projectors.py linear [OPTIONS]

  A command-line interface to compute linear tangent-space projectors for
  densely-sampled datasets.

Options:
  --dataroot DIRECTORY            path to store datasets
  --dataset [cifar|mnist|fmnist]  dataset to train on
  --neighbours INTEGER            number of nearest neighbours to use
  --d-singular INTEGER            number of singular dimension for the tsp-
                                  projection

  --download / --no-download      download data if not available
  --overwrite / --no-overwrite    overwrite output if it exists
  --all-data / --no-all-data      also compute projectors for training split
  --strict-mode / --no-strict-mode
                                  tangent planes strictly contain data points
  --save FILE                     target hdf5 file to store singular vectors
  --batch-size INTEGER            size of mini-batches for svd
  --targets CSINTS                classes to compute for
  --help                          Show this message and exit.
```

Projectors for classes 0 and 1 of the test split for FashionMNIST can be computed with the following command:
```bash
$ python script/projectors.py linear \
    --dataroot "share/data" \
    --dataset "fmnist" \
    --neighbours 32 \
    --d-singular 16 \
    --save "var/tsp/fmnist.h5" \
    --batch-size 128 \
    --targets 0,1
```
The result will be stored in `var/tsp/fmnist.h5`.
Note that you can append more classes to the same file later by re-running the command and changing the `--targets` option.
This only works if `--d-singular` has the same values, though `--neighbours` should also have the same value to not
estimate tangent-space differently for different classes.
Projectors for unspecified classes will be set to zero, so unless they are not needed, all classes should be computed
in one execution to prevent using zero projectors by accident.

## TSP-Projected Fairwashing
Both `script/fairwashing.py`'s commands `train` and `evaluate` support the option `--proj-file`, which can be pointed
to a valid projector file, i.e.:
```bash
$ python script/fairwashing.py evaluate \
    --dataroot "share/data" \
    --dataset "fmnist" \
    --method "grad" \
    --model "share/models/fmnist_grad.pth" \
    --teacher-model "share/models/fmnist_acc.pth" \
    --target "share/targets/42.png" \
    --batch-size 128 \
    --proj-file "var/tsp/fmnist.h5"
```
When using projectors for training, remember to call `script/projectors.py linear` with the option `--all-data`.
