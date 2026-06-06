# Camera-aware Label Noise Suppression (CLNS)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)


**Official implementation of "CLNS: Camera-aware Label Noise Suppression for
Unsupervised Visible-Infrared Person Re-Identification"**


> **Abstract:** Unsupervised visible-infrared person re-identification (US-VI-ReID) retrieves pedestrian images across modalities without manual annotations. To address camera-specific
biases that fragment identities and amplify label noise, we propose the Camera-aware
Label Noise Suppression (CLNS) framework, a coarse-to-fine pipeline that progressively purifies noise. Specifically, the Camera-aware Prototype Calibration (CPC)
module exploits cross-camera consistency to rectify structural errors and construct reliable prototypes. Building on this, Optimal Transport Prototype Matching (OTPM)
achieves global cross-modality alignment. At the instance level, Neighbor-guided
Camera-domain Learning (NCL) densifies feature distributions using soft supervision,
while a Noise-aware Memory Updating (NMU) strategy prevents error accumulation.
On the SYSU-MM01, RegDB, and LLCM datasets, CLNS achieves Rank-1 (mAP)
accuracies of 69.5% (65.6%), 94.8% (91.8%), and 54.6% (57.8%), respectively, significantly surpassing state-of-the-art methods. The code will be released at https:
//github.com/apatientzsc/CLNS.


---
## 📢 News

*   **`2026-06-06`**: Pre-trained models and inference code are now available. Check out the results section to download them.

---
## 📋 Table of Contents
- [🛠️ Requirements & Installation](#️-requirements--installation)
- [📁 Dataset Preparation](#-dataset-preparation)
- [🎯 Performance](#-performance)
- [📥 Model Zoo](#-model-zoo)

---
## 🛠️ Requirements & Installation

### Prerequisites
- Python 3.10
- PyTorch 2.1.1
- CUDA 11.8
- Linux or macOS (Windows is not officially supported)

### Step-by-Step Installation

We highly recommend using a Conda environment to manage dependencies.

1.  **Clone the repository**
    ```shell
    git clone https://github.com/apatientzsc/CLNS.git
    cd CLNS
    ```

2.  **Create and activate the Conda environment**
    ```shell
    conda create --name CLNS python=3.10 -y
    conda activate CLNS
    ```

3.  **Install PyTorch and CUDA dependencies**
    ```shell
    # It's recommended to follow the official PyTorch installation guide for your specific system
    # but here is the command for the specified version:
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Install other dependencies**
    ```shell
    conda env update --file environment.yml
    ```

5.  **Compile C++/CUDA extensions**
    ```shell
    cd code/extension
    sh make.sh
    cd ../..
    ```

Now, your environment is set up and ready for training and testing.

---
## 📁 Dataset Preparation

1.  **Create a dataset directory**
    ```shell
    mkdir -p dataset
    ```

2.  **Download and Process Datasets**

    Download the following person Re-ID datasets and organize them within the `dataset/` directory.

    - [SYSU-MM01](http://www.sysu-hcp.net/sysumm01/)
    - [RegDB](https://github.com/shizenglin/Link-Partial-ReID)
    - [LLCM](https://github.com/ZYK100/LLCM)

3.  **Pre-processing**
    Use the provided scripts in the `code/data_process/pre_process` directory to process each dataset. For example:

    ```shell
    # Example for processing SYSU-MM01
    python code/data_process/pre_process/process_sysu.py --data_path ./dataset/SYSU-MM01

    # Example for processing RegDB
    python code/data_process/pre_process/process_regdb.py --data_path ./dataset/RegDB

    # Example for processing LLCM
    python code/data_process/pre_process/process_llcm.py --data_path ./dataset/LLCM
    ```

    *Note: Please adjust the script commands according to the actual usage instructions in the corresponding Python files.*

After running the scripts, your dataset directory should be structured properly for training and testing.

---
## 🎯 Performance

Our CLNS method achieves state-of-the-art performance on three popular cross-modal Re-ID benchmarks.
### Results on RegDB

| Mode      | Rank-1 |  mAP   |  mINP  | 
|:----------|:------:|:------:|:------:|
| VIS to IR | 94.79% | 91.83% | 85.8%  |
| IR to VIS | 95.33% | 91.79% | 84.25% | 

### Results on SYSU-MM01

| Mode          | Rank-1 |  mAP  |  mINP |
|:--------------|:------:|:-----:|:-----:|
| All Search    | 69.49% | 65.64%| 51.2% |
| Indoor Search | 74.67% | 78.95%|75.31% |

### Results on LLCM

| Mode      | Rank-1 |  mAP  | mINP  |
|:----------|:------:|:-----:|:-----:|
| VIS to IR | 54.6%  | 57.8% | 51.9% |
| IR to VIS | 46.9%  | 52.6% | 48.8% |

---
## 📥 Model Zoo

You can download our pre-trained models from the links below. To use them, place the downloaded `.pth` files in the `logs/` directory (or as specified in your testing script).

### RegDB, SYSU-MM01 and LLCM Models
[📥 Download from GoogleDrive](https://drive.google.com/file/d/1awkIVqgyloiL18O2b2YwIYXkAaHffT85/view?usp=drive_link)

## 📥 Model Zoo

You can download our environment from the links below. Our environment is base on ubuntu 20.04LTS.

### Ubuntu environment
[📥 Download from GoogleDrive](https://drive.google.com/file/d/11vHhTZFqe3Pn1QoPAi5wBItiEvf0jPIF/view?usp=drive_link)

