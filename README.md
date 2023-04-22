# deep_packet_project
[CS536 Project] Deep packet: a novel approach for encrypted traffic classification using deep learning
Contributions:
* MLP and SAE models in [ml/models.py](./ml/models.py)
* training code refactored in [train.py](./train.py)

## How to Use

* Clone the project: `$ git clone https://github.com/gtmdotme/cs536_bonus_project`
* Create environment via conda
    * For Mac
      ```bash
      conda env create -f env_mac.yaml
      ```
    * For Linux (CPU only)
      ```bash
      conda env create -f env_linux_cpu.yaml
      ```
    * For Linux (CUDA 10.2)
      ```bash
      conda env create -f env_linux_cuda102.yaml
      ```
    * For Linux (CUDA 11.3)
      ```bash
      conda env create -f env_linux_cuda113.yaml
      ```
* Now, activate the environment before running any scripts: `$ conda activate deep_packet`

## Data Pre-processing

Download the train and test set created at [here](https://drive.google.com/file/d/1EF2MYyxMOWppCUXlte8lopkytMyiuQu_/view?usp=sharing), or download
  the [full dataset](https://www.unb.ca/cic/datasets/vpn.html) if you want to process the data from scratch.
  
```bash
python preprocessing.py -s /path/to/CompletePcap/ -t processed_data
```

## Create Train and Test

To create different train test split, use different `seeds` in file [create_train_test_set.py](./create_train_test_set.py)
```bash
python create_train_test_set.py -s processed_data -t train_test_data
```

## Train Model

```bash
python train.py -d <path/to/train/data> -m <path/to/save-models> -t <task>
```

Application Classification

* MLP Model: `$ python train.py -d train_test_data/application_classification/train.parquet -m model/v1.application_classification.mlp.model -t app -a mlp`

* SAE Model: `$ python train.py -d train_test_data/application_classification/train.parquet -m model/v1.application_classification.sae.model -t app -a sae`

* CNN Model: `$ python train.py -d train_test_data/application_classification/train.parquet -m model/v1.application_classification.cnn.model -t app -a cnn`

Traffic Classification

* MLP Model: `$ python train.py -d train_test_data/traffic_classification/train.parquet -m model/v1.traffic_classification.mlp.model -t app -a mlp`

* SAE Model: `$ python train.py -d train_test_data/traffic_classification/train.parquet -m model/v1.traffic_classification.sae.model -t app -a sae`

* CNN Model: `$ python train.py -d train_test_data/traffic_classification/train.parquet -m model/v1.traffic_classification.cnn.model -t app -a cnn`


## Evaluation Result (CNN)
Run the [main-evaluate.ipynb](./main-evaluate.ipynb) by replacing the model name with the model file used while training.
