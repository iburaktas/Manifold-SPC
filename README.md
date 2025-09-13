# High-Dimensional Statistical Process Control via Manifold Fitting and Learning

This repository contains the code and experiments for the paper:

[**High-Dimensional Statistical Process Control via Manifold Fitting and Learning**](https://arxiv.org/abs/xxxx.xxxxx)



## Abstract 

This paper addresses the Statistical Process Control (SPC) of high-dimensional, dynamic industrial processes from two complementary perspectives: manifold fitting and manifold learning, both of which assume data lies on an underlying nonlinear, lower dimensional space. We propose two distinct monitoring frameworks for online or 'phase II' Statistical Process Control (SPC). The first method leverages state-of-the-art techniques in manifold fitting to accurately approximate the manifold where the data resides within the ambient high-dimensional space. It then monitors deviations from this manifold using a novel scalar distribution-free control chart. In contrast, the second method adopts a more traditional approach, akin to those used in linear dimensionality reduction SPC techniques, by first embedding the data into a lower-dimensional space before monitoring the embedded observations. We prove how both methods provide a controllable Type I error probability, after which they are contrasted for their corresponding fault detection ability. Extensive numerical experiments on a synthetic process and on a replicated Tennessee Eastman Process show that the conceptually simpler manifold-fitting approach achieves performance competitive with, and sometimes superior to, the more classical lower-dimensional manifold monitoring methods. In addition, we demonstrate the practical applicability of the proposed manifold-fitting approach by successfully detecting surface anomalies in a real image dataset of electrical commutators.

## This Repository

This repository implements the two SPC frameworks described in the paper.  
- The code is organized as a **pipeline**, which can be adapted for any high-dimensional dataset.  
- Example scripts are included to reproduce scaled-down versions of the experiments presented in the paper.  
- Users can connect their own high-dimensional datasets to run SPC directly in our proposed framework.

# Installation

Clone the repository:

```bash
git clone https://github.com/iburaktas/Manifold-SPC.git
cd Manifold-SPC
pip install -r requirements.txt
```
The control charts DFEWMA and DFUC are implemented in C++ via pybind11 and must be compiled before use.

```bash
cd src/utils/ControlChart
python setup.py build_ext --inplace
```

# Project Structure

<pre>
Manifold-SPC/
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
│
├── main.py                  # For the scaled-down versions of the experiments
│
├── experiments/             # Where the results are written
│   ├── KolektorRun/
│   ├── SyntheticProcessRun/
│   │   ├── MF
│   │   ├── PCA
│   │   ├── NPE
│   │   ├── LPP
|   |   ├── rl_summary.csv  # The results are reported here
│   └── TERun/
│
├── figures/
│   ├── Figure_maker.py     # Reproduce most of the figures shown in the paper
│
├── src/
│   ├── data/     # Reproduce most of the figures shown in the paper
│   │   ├── KolektorSDD/                    # Data for KolektorSDD
│   │   ├── TennesseeEastmanProcess/        # TE Files
│   │   │   ├── Data/                       
│   │   │   ├── TE/                         # TE Simulator
│   │   │   │   ├──RunSimWithFaults_parallel.m  # Generates TE data from setup.csv file
│   │   ├── data_loader.py                  # DATA LOADER
│   ├── models/
│   │   │   ├── ysl23.py                    # Manifold Fitting Method
│   │   │   ├── lpp.py                      # LPP
│   │   │   ├── npe.py                      # NPE
│   ├── utils/
│   │   ├── ControlChart/                   # DFUC and DFEWMA control charts
│   │   ├── Filters.py                      # Fits AR model
│   │   ├── functions.py                    # Some functions
│   │   ├── runner.py                       # EXPERIMENT RUNNER
│   │   ├── ExperimentConfig.py             # EXPERIMENT CONFIG FOR THE RUNNER
</pre>


# Usage

## Reproduce Scaled-Down Versions of the Experiments

We provide example commands to reproduce simplified versions of the experiments from the paper:

```bash
# Tennessee Eastman Process (TEP)
python main.py --exp-type TE --exp-name TERun --N 10 --fault-no 4 --amplitude 0.1 --n-workers 8

# KolektorSDD
python main.py --exp-type KolektorSDD --exp-name KolektorRun --n-workers 1 --sigma-estimate 0 --alpha 0.005

# Synthetic Process
python main.py --exp-type SyntheticProcess --exp-name SyntheticProcessRun --N 100 --fault-no 4 --amplitude 10
```
## Using the Pipeline

```python
from src.utils.data_loader import DataLoader_
from src.utils.ExperimentConfig import ExperimentConfig
from src.utils.runner import Runner

# create DataLoader_ as in pytorch DataLoader fashion.
class MyLoader(DataLoader_):
    def __init__(self, N: int, data: np.ndarray):
        super().__init__()   
        self.N = N
        self.data = data
    def __len__(self):
        return self.N
    def get_item(self, idx):
        return data[idx]

my_data_loader= MyLoader(N,data)

# Next define the experiment config for the Runner, for instance
my_cfg = ExperimentConfig()
my_cfg.exp_name = "MyExperiment"
my_cfg.methods = ["LPP","MF"]

# Now create the runner and let it run
my_runner = Runner(dataloader=my_data_loader, cfg=my_cfg)
my_runner.run()
```