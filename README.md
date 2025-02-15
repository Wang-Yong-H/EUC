# On Adversarial Robust Generalization of DNNs for Remote Sensing Image Classification

This repository contains the code of data algorithms,

# Pre-trained Models

Please find the pre-trained models through this OneDrive [sharepoint.](https://emckclac-my.sharepoint.com/:f:/g/personal/k19010102_kcl_ac_uk/EnVH6skz4q1FoamAcrPRkdgBpNEDkpL9cSIgttJDjKs1AQ)
 and pre-trained models from the (https://openreview.net/forum?id=y4uc4NtTWaq)" published at ICLR 2023.

We validate the effectiveness and feasibility of our method through extensive experiments on two commonly used remote sensing image classification datasets NWPU-RESSC45, RSSCN7,
and WHU-RS19, demonstrating its superiority over previous methods, particularly in the context of improving the adversarial robust generalization of DNNs. Index Terms—Remote sensing ima

# Files

* `data/`: dataset
* `model/`: model checkpoints
  * `trained/`: saved model checkpoints
* `output/`: experiment logs
* `src/`: source code
  * `train.py`: training models
  * `adversary.py`: evaluating adversarial robustness
  * `utils`: shared utilities such as training, evaluation, log, printing, adversary, multiprocessing distribution
  * `model/`: model architectures
  * `data/`: data processing
    * `idbh.py`: **the implementation of Cropshift and IDBH**
  * `config/`: configurations for training and adversarial evaluation
    * `config.py`: hyper-parameters shared between `src/train.py` and `src/adversary.py`
    * `train.py`: training specific configurations
    * `adversary.py`: evaluation specific configurations

# Requirements

The development environment is:

1. Python 3.8.13
2. PyTorch 1.11.0 + torchvision 0.12.0

The remaining dependencies are specified in the file `requirements.txt` and can be easily installed via the command:

```p
pip install -r requirements.txt
```

To prepare the involved dataset, an optional parameter `--download` should be specified in the running command. The program will download the required files automatically. This functionality currently doesn't support the dataset Tiny ImageNet.

# Dependencies

* The training script is based on [Combating-RO-AdvLC](https://github.com/TreeLLi/Combating-RO-AdvLC)
* the code of Wide ResNet is a revised version of [wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch).
* the code of PreAct ResNet is from [Alleviate-Robust_Overfitting](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* Stochastic Weight Averaging (SWA): [Alleviate-Robust_Overfitting](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* Hessian spectrum computation: [PyHessian](https://github.com/amirgholami/PyHessian)

# Training
(PGD-1)  -a wresnet --depth 34 

python src/train.py -a paresnet --depth 18 -d WHU

python src/train.py -a wresnet --depth 34 -d NWPU


# To train a Wide  PreAct ResNet18 on  RSSCN7 using PGD10 with PGD , run:
python src/train.py -a paresnet --depth 18 -d RSSCN7

# To train a Wide  PreAct ResNet18 on  WHU-RS19
python src/train.py -a wresnet --depth 34 -d WHU

# To train a Wide ResNet34-10 on  RSSCN7 using PGD10 with SWA , run:


python src/train.py -a wresnet --depth 34 -d RSSCN7 --swa 0 0.001 1


```

Please refer to the specific configuration file for the details of hyperparameters. Particularly, `--swa 0 0.001 1` means that SWA begins from the 0th epoch, the decay weight is 0.001, and models are averaged every 1 iteration.

# Evaluation

For each training, the checkpoints will be saved in `model/trained/{log}` where {log} is the name of the experiment logbook (by default, is `log`). Each instance of training is tagged with a unique identifier, found in the logbook `output/log/{log}.json`, and that id is later used to load the well-trained model for the evaluation.

To evaluate the robustness of the "best" checkpoint against PGD50, run:

```
python src/adversary.py 0000 -v pgd -a PGD --max_iter 50
```

Similarly against AutoAttack (AA), run:

```
python src/adversary.py 0000 -v pgd -a AA
```

where "0000" should be replaced the real identifier to be evaluated.
