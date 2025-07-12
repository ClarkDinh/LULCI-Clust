# LULCI-Clust

**LULCI-Clust** is a deep learning-based framework for unsupervised **Land Use and Land Cover (LULC)** image clustering. It integrates powerful vision and dimensionality reduction models to capture complex spatial patterns and semantic information from satellite or aerial imagery, making it scalable and robust across diverse geographical datasets.

## 🔍 Overview

Land Use and Land Cover analysis is crucial for understanding spatial dynamics in environmental monitoring, urban planning, and sustainable development. Traditional clustering methods often fail to scale or adapt to complex LULC patterns. **LULCI-Clust** addresses these challenges with a hybrid framework that combines:

- 🧠 **Vision Transformer (ViT)** for extracting high-level semantic features from image patches  
- 🔄 **Variational Autoencoder (VAE)** for learning latent structural representations  
- 🔽 **UMAP** for effective dimensionality reduction  
- 🔢 **k-means++** for final clustering

## 📂 Repository Structure

```
├── eu_jpg/                     # European Union Urban Atlas sample images
├── japan_jpg/                  # Japan LULC imagery
├── vietnam_jpg/                # Vietnam LULC imagery
├── _main_experiment_eu.ipynb   # Notebook for EU dataset experiments
├── _main_experiment_jp.ipynb   # Notebook for Japan dataset experiments
├── _main_experiment_vn.ipynb   # Notebook for Vietnam dataset experiments
├── LICENSE                     # Apache 2.0 License
└── README.md                   # Project documentation
```

## 🚀 Quickstart

### 1. Set up the environment

```bash
conda create --name lulc python=3.11
conda activate lulc
conda install matplotlib seaborn scipy scikit-learn jupyter pandas ipykernel
python -m ipykernel install --user --name lulc
pip install transformers torch torchvision
pip install -U sentence-transformers
pip install tf_keras tensorflow umap-learn
```

### 2. Run the Notebooks

Choose the appropriate notebook based on the dataset you'd like to analyze:

- `_main_experiment_eu.ipynb` – for Urban Atlas LULC 2018 (EU)
- `_main_experiment_jp.ipynb` – for Japan imagery
- `_main_experiment_vn.ipynb` – for Vietnam imagery

## 🧪 Datasets

This repository includes sample datasets from:

- **Urban Atlas LULC 2018**
- **Japan** – recent high-resolution LULC maps
- **Vietnam** – urban and rural LULC regions

## 📈 Applications

- Geospatial and Remote Sensing
- Urban and Environmental Planning
- Agriculture and Forestry Monitoring
- Transportation and Infrastructure Development

## 📜 License

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](LICENSE) file for details.

## 🌐 Citation and Acknowledgment

If you use this work in your research, please cite our paper (link will be provided soon).  
The source code and datasets are part of the forthcoming **LULCMiner** project:  
🔗 https://github.com/ClarkDinh/LULCMiner
