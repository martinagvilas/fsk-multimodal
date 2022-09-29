# Probing feature-based semantic knowledge in Vision+Language deep neural networks

## 1. Download dataset information
You will need to:

1. Clone this repository
```
git clone (anon)
cd fsk-multimodal
```

2. Run the following command to download some annotation files and the THINGS dataset:
```
prepare_data.sh
```
You will be prompted for a password to download the THINGS dataset.
Read the usage terms and retrieve this password [here](https://osf.io/srv7t).


3. Download the semantic feature norms dataset from McRae 2005 using 
[this link](https://sites.google.com/site/kenmcraelab/norms-data).


## 2. Install software
We recommend that you use a [conda environment](https://docs.conda.io/projects/conda/en/latest/index.html) to install the software necessary for running the experiments.

Step into your clone folder, and run the following commands:
```
conda create --name fsk python=3.9
conda activate fsk
pip install -r requirements.txt
pip install .
```

## 3. Run experiments
To compute the matching between each image and semantic feature using all models, 
run:

```
 python -m fsk.it_match.run -m all -pp {your path to the project folder}
```

You can also select which model to run by changing the value of the `-m` flag.

To compute the representational similarity analysis, run:

```
 python -m fsk.rsa.run -m all -pp {your path to the project folder}
```

You will also need to run the following:

```
 python -m fsk.rsa.pred_similarity -m {} -pp {your path to the project folder}
```
and

```
 python -m fsk.rsa.pred_similarity -m {} -pp {your path to the project folder}
```
for every model of {clip, vilt albef}.


To compute the mutual information analysis, run:

```
 python -m fsk.feature_repr.mutual_information -m {} -pp {your path to the project folder}
```
for every model in {clip, vilt albef}.

Once you got all the set of results, you can generate the figures and tables 
for the manuscript with the notebook "demo_results.ipynb"