# Curvature-Aware Multi-View Graph Pooling with Negative Curvature Rewiring (MVRPool)

## 1.Overview
The code for paper "Curvature-Aware Multi-View Graph Pooling with Negative Curvature Rewiring". 

![image](data/Fig2.jpg)

The repository is organized as follows:
* **`MVRPool/`** - Contains model training scripts and parameter configuration code
* **`model/`** - Implements the core model architecture and baseline models
* **`data/`** - Stores required datasets (both raw and preprocessed data)
* **`results/`** - Saves intermediate files and output results generated during model execution

## 2.Dependencies
* python == 3.9
* numpy == 1.24.3
* pandas == 2.3.3
* scikit_learn == 1.7.2
* torch == 2.5.1+cu124
* torch_geometric == 2.6.1
* torchinfo == 1.8.0 
* torch_cluster == 1.6.3+pt25cu124
* torch_scatter == 2.1.2+pt25cu124
* torch_sparse == 0.6.18+pt25cu124
* torch_spline_conv == 1.2.2+pt25cu124
* torchvision == 0.20.1+cu124

## 3.Supported datasets
TuDataset: 
* `PROTEINS`
* `NCI1`
* `NCI109`
* `Mutagenicity`
* `ENZYMES`
* `PTC_MR`
* `PTC_FM`
* `PTC_FR`
* `PTC_MM`
* `IMDB-BINARY`
* `IMDB-MULTI`
* `MUTAG`

Datasets mentioned above will be downloaded automatically using PyG's API when running the code.

For more information about the datasets, please refer to the [TUDataset documentation](https://chrsmrrs.github.io/datasets/).

## 4.Contacts
If you have any questions, please email Wenju Hou (wjhou23@mails.jlu.edu.cn)
