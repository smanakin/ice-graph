# Project IceGraph: An End-to-End MLOps System for Sea Ice Tracking



This repository contains the complete source code for **Project IceGraph**, a production-grade MLOps system designed to track the drift of sea ice floes from multi-modal satellite data.

The project's mission is to move beyond static sea ice *classification* and build a dynamic *tracking* system. It ingests, processes, and analyzes high-volume satellite imagery to model sea ice as an interactive graph, enabling the prediction of ice floe movement over time.

This system is built entirely on **Microsoft Azure** using **professional DevOps practices**, including full automation (CI/CD), containerization (Docker), and large-scale data processing (Spark).

## 🎯 Core Features

* **Multi-Modal Data Fusion:** Ingests and fuses high-resolution **Sentinel-1 (SAR)** radar data with low-resolution **AMSR2 (microwave radiometer)** data.
* **CV Instance Segmentation:** Uses a **Vision Transformer (SegFormer)**, pre-trained with Self-Supervised Learning (SSL), to accurately identify and segment individual ice floes, even in noisy SAR images.
* **GNN-based Tracking:** Models the segmented ice floes as **nodes in a graph**. A **Graph Neural Network (GNN)** is then trained to solve the matching problem, tracking the correspondence of each ice floe from one day to the next.
* **Distributed Data Engineering:** A robust **PySpark** pipeline on **Azure Databricks** handles all pre-processing (speckle filtering, calibration, co-registration) at terabyte scale.
* **End-to-End MLOps:** A fully automated **CI/CD pipeline** (using GitHub Actions) builds, tests, and deploys the system.
* **Dual-Deployment:**
    1.  **Batch API:** A **PySpark** batch-scoring pipeline on **Databricks** for processing years of archival data.
    2.  **Real-Time API:** A **FastAPI** application deployed to **Azure Kubernetes Service (AKS)** for on-demand inference.

## 🏛️ System Architecture

This diagram outlines the complete MLOps lifecycle for the project, from data ingestion to model deployment.

### Data sources
- https://browser.dataspace.copernicus.eu/
- https://search.asf.alaska.edu/#/
- https://search.earthdata.nasa.gov/#
  
*(This is where you should insert the `architecture.png` diagram you create)*

1.  **Ingestion:** `Azure Data Factory (ADF)` pulls raw S1/AMSR2 data into `Azure Blob Storage (Bronze)`.
2.  **ETL (Processing):** `Azure Databricks (PySpark)` reads from `Bronze`, runs speckle filtering and fusion, and writes to a `Gold (Delta Lake)` layer.
3.  **Training:** `Azure Machine Learning (AML)` trains the CV and GNN models using a **GPU Compute Cluster**, reading data from the `Gold` layer and saving models to the `AML Model Registry`.
4.  **CI/CD (Build):** A `GitHub Actions` workflow triggers on a `git push`. It tests the code, builds a production **Docker** image, and pushes it to `Azure Container Registry (ACR)`.
5.  **Deployment (API):** A second `GitHub Actions` workflow deploys the new image from ACR to the `Azure Kubernetes Service (AKS)` cluster for the real-time API.
6.  **Deployment (Batch):** An `Azure Databricks Workflow` runs a PySpark job that loads the model from the `AML Registry` to score data in bulk.

## 🛠️ Tech Stack

* **Cloud Platform:** Microsoft Azure
* **Data Processing:** Azure Databricks, Apache Spark (PySpark), Delta Lake
* **Data Ingestion:** Azure Data Factory
* **MLOps:** Azure Machine Learning (AML)
* **Deployment:** Azure Kubernetes Service (AKS), Docker, FastAPI
* **CI/CD:** GitHub Actions
* **Models:** PyTorch, PyTorch Geometric (GNNs), Hugging Face Transformers (CV)
* **Geospatial:** GDAL, Rasterio, Apache Sedona (GeoSpark)
* **Dev Tools:** Git, Conda, Pre-commit (Black, Flake8)

## 🚀 How to Run

### 1. Setup Local Environment

This project is managed with `conda` and `pre-commit`.

```bash
# Clone the repository
git clone [https://github.com/YourUsername/IceGraph.git](https://github.com/YourUsername/IceGraph.git)
cd IceGraph

# 1. Create the conda environment
conda env create -f environment.yml
conda activate icegraph

# 2. Install Git hooks (for auto-formatting)
pre-commit install
```

### 2. Run the Data Pipeline

The core data processing logic is run as a Spark job on Azure Databricks. See `notebooks/01_databricks_etl.ipynb` for details.

### 3. Train the Model

Model training is orchestrated by Azure Machine Learning.

```bash
# Run the main training script (submits to AML)
python scripts/train.py --model_type 'segformer' --learning_rate 0.0001
```

### 4. Run CI/CD

The automated CI/CD pipeline is defined in `.github/workflows/`. It runs automatically on any push or pull request to the `main` branch.
