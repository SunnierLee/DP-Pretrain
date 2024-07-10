<div align=center>
  
# Differentially Private Self-Pretraining Diffusion Models for the Private Image Synthesis
</div>

This is the official implementaion of paper ***Differentially Private Self-Pretraining Diffusion Models for the Private Image Synthesis***. This repository contains Pytorch training code and evaluation code. PRIVIMAGE is a Differetial Privacy (DP) image generation tool, which leverages the DP technique to generate synthetic data to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.


## 1. Contents
- Differentially Private Self-Pretraining Diffusion Models for the Private Image Synthesis
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Inference](#34-inference)

## 2. Introduction

Differential Privacy (DP) image data synthesis aims to generate synthetic images from a sensitive image dataset while preserving the privacy of the individual images in the dataset, allowing organizations to share and utilize synthetic images without privacy concerns. Although previous methods have achieved great progress, especially training diffusion models on sensitive images with DP Stochastic Gradient Descent (DP-SGD), they still suffer from unsatisfactory performance degradation due to the problematic convergence of DP-SGD, especially on some complex image datasets. Motivated by recent success in which researchers leveraged a public image dataset to pre-train diffusion models before DP-SGD training, this paper explores how to pre-train diffusion models without using a public dataset. We propose a novel DP pre-training method, termed Differentially Private Self-Pretraining (DPP). DPP first queries center images from the sensitive image dataset with suitable Gaussian injected for satisfying DP. Then, these center images are post-enhanced by an augmentation algorithm bag, and enhanced images are used for pre-training diffusion models. Finally, we fine-tune the pre-trained diffusion models on the sensitive dataset using DP-SGD. Extensive experiments demonstrate that, on the average of four investigated image datasets, the FID and classification accuracy of the downstream task of synthetic images from DPP is 33.1% lower and 2.1% higher than the state-of-the-art method that does not use public data. Additional experiments show that when the public dataset used for pre-training is explicitly distinct from the sensitive dataset we aim to protect, DPP still achieves most-ofthe-best performance without using an additional public dataset.

## 3. Get Start
We provide an example for how to reproduce the results on MNIST in our paper. Suppose you had 4 GPUs on your device.

### 3.1 Installation

To setup the environment of DPP, we use `conda` to manage our dependencies. Our developers are conducted using `CUDA 11.8`. 

Run the following commands to install PRIVIMAGE:
 ```
conda create -n privimage python=3.8 -y && conda activate privimage
pip install --upgrade pip
pip install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
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

### 3.3 Pre-training
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

