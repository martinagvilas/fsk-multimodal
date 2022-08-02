# Probing feature-based semantic knowledge in V+L DNNs

{TODO: write description of the study and link to pre-print}

## 1. Download dataset information
You will need to:

1. Clone this repository
```
git clone https://github.com/martinagvilas/fsk-multimodal.git
cd fsk-multimodal
```

2. Run the following command to download some annotation files and the THINGS dataset:
```
prepare_data.sh
```
You will be prompted for a password to download the THINGS dataset.
Read the usage terms and retrieve this password [here](https://osf.io/srv7t).

{TODO: Explain annotations files and give credits}

4. Download the semantic feature norms dataset from McRae 2005 using 
[this link](https://sites.google.com/site/kenmcraelab/norms-data).

{TODO: how to ensure the availability of these files}


## 2. Install software
We recommend that you use a [conda environment](https://docs.conda.io/projects/conda/en/latest/index.html) to install the software necessary for running the experiments.

Step into your clone folder, and run the following commands:
```
conda create --name fsk -python=3.9
conda activate fsk
pip install -r requirements.txt
pip install .
```

## 3. Run experiments