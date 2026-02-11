# AQVR-ICM

This repository provides the official implementation of the paper: **"Training-Free Adaptive Quantization for Variable Rate Image Coding for Machines."**

Our paper is available at [arXiv](https://arxiv.org/abs/2511.05836). 

## Installation

#### 1. Clone this repository:

```bash
git clone
cd AQVR-ICM
```

#### 2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Running the evaluation is a two-step process. First, you must download the pre-trained model weights, and then you can run the evaluation script.

#### 1. Prerequisites: Download Model Weights

This implementation relies on pre-trained model weights from the SA-ICM project.
Visit the [SA-ICM repository](https://github.com/final-0/SA-ICM?tab=readme-ov-file) to download the required checkpoint.
In our paper, we used the "icm_78.pth.tar" checkpoint.
Place the downloaded model weight in the "param" folder.

The expected directory structure should be:
```bash
param ---- param_details.txt
       |-- icm_78.pth.tar
```

#### 2. Run Evaluation

You can now run the evaluation using the coding.py script. 
The --d command-line argument corresponds to the parameter $d$ mentioned in our paper.
You can set --d to any value greater than 0.

Argument descriptions:

- --checkpoint: Path to the downloaded SA-ICM model checkpoint.
- --input: Path to your input image or a directory of images.
- --save_path: Directory where the encoded results will be saved.
- --d: The quantization parameter (corresponds to d in the paper).

Example command (using $d = 8$):
```bash
python coding.py --checkpoint param/icm_78.pth.tar --input input_image --save-path /path/to/save --d 8
```
