# tick_classification_project

This project implements a full workflow for tick image classification using BioCLIP embeddings and SVM few-shot classification.

## Data Exploration (01)

The initial step involves exploring the tick image dataset to understand its structure and characteristics. This includes visualizing sample images, analyzing class distributions, and preparing the data for downstream tasks.

## BioCLIP Inference (02)

In this stage, we use the BioCLIP model to extract embeddings from the tick images. The embeddings capture meaningful biological features that are useful for classification. Cached embeddings are stored to speed up subsequent processing and avoid redundant computation.

## SVM Few-Shot Classification (Blocks 0–6)

Using the cached BioCLIP embeddings, we perform few-shot classification with Support Vector Machines (SVM). This approach allows effective classification with limited labeled examples. Results and evaluation metrics from the SVM classification are saved for analysis.

## Cached Embeddings and Results

- Cached embeddings generated during the BioCLIP inference step are stored in the `cached_embeddings/` directory.
- Classification results and evaluation outputs from the SVM few-shot workflow are saved in the `results/` directory.

This structured workflow enables efficient and reproducible tick species classification leveraging state-of-the-art biological image embeddings combined with classical machine learning methods.
