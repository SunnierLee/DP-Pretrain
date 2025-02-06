<div align=center>
  
# From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis
</div>

This is the official implementaion of paper ***From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis***. This repository contains Pytorch training code and evaluation code. DP-FETA is a Differetial Privacy (DP) image generation tool, which leverages the DP technique to generate synthetic data to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.


## 1. Contents
- From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Evaluation](#34-evaluation)

## 2. Introduction

Differentially private (DP) image data synthesis aims to generate synthetic images from a sensitive image dataset while preserving privacy, so organizations can share and utilize synthetic images. Although previous methods have achieved great progress, especially in training diffusion models on sensitive images with DP Stochastic Gradient Descent (DP-SGD), they still suffer from unsatisfactory performance. In this work, inspired by curriculum learning, we propose a two-stage DP image synthesis framework, where diffusion models learn to generate DP synthetic images from easy to hard. Unlike existing methods that directly use DP-SGD to train models, we propose an easy stage in the beginning, where diffusion models learn simple features of the sensitive images. To facilitate this easy stage, we propose to use central images, simply aggregations of random samples of the sensitive dataset. Intuitively, although those central images do not show details, they demonstrate useful characteristics of all images and only incur minimal `DP cost', thus helping early-phase model training.

## 3. Get Start
We provide an example for how to reproduce the results on MNIST in our paper. Suppose you had 4 GPUs on your device.

### 3.1 Installation

To setup the environment of DP-FETA, we use `conda` to manage our dependencies. Our developers are conducted using `CUDA 11.8`. 

Run the following commands to install DP-FETA:
 ```
conda create -n dp-feta python=3.8 -y && conda activate dp-feta
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd opacus
pip install -e .
 ```

### 3.2 Dataset and Files Preparation

Preprocess dataset.
```
# download and preprocess MNIST
python precompute_data_mnist_fid_statistics.py
```

### 3.3 Training
First, query mean images from MNIST.
```
python extract_mean.py
```
And then, we pre-train the diffusion model on these mean images.
```
python main.py --mode pretrain --config configs/mnist_28/pretrain.yaml --workdir pretrain_mnist28_e1_mean_ch22_at14_n5_sig5_q0.1 --data.path=mean_mnist_5_5_0.1/noisy
```
After pre-training, we fine-tune the diffusion model on MNIST with DP-SGD.
```
python main.py --mode train --config configs/mnist_28/train_eps_1.0.yaml --workdir mnist28_e1_mean_ch22_at14_n5_sig5_q0.1 --data.path=data
```

After training, the FID will be saved in `./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1/stdout.txt` and the synthetic images will be saved in `./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1/samples60000_acc`

### 3.4 Evaluation

Use 60,000 synthetic images to train a CNN classifier.

```
python downstream_classification.py --train_dir ./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1/samples60000_acc/samples --test_dir ./data/ --out_dir ./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1/samples60000_acc/
```
The Acc on the testset will be saved into `./mnist28_e1_mean_ch22_at14_n5_sig5_q0.1/samples60000_acc/evaluation_downstream_acc_log.txt`.

