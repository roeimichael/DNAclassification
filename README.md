# COVID-19 Variant Classification using Deep Learning

This project explores the application of deep learning techniques to classify COVID-19 virus variants based on lineage information. Leveraging the power of convolutional neural networks (CNN), the system achieves a high level of accuracy.

## Overview

The project involves the following key steps:
- **Data Extraction**: Pulling data from the [COVID-19 Data Portal](https://www.covid19dataportal.org/) and handling millions of sequences.
- **Data Processing**: Encoding, cleaning, and resizing sequences.
- **Model Building**: Constructing a complex CNN, consisting of convolutional layers, batch normalization, ReLU activation, max pooling, dropout, and fully connected layers.
- **Training & Evaluation**: Training the model with various hyperparameters and evaluating the performance using metrics like accuracy, precision, top-3 accuracy, and top-5 accuracy.

## Requirements

The project utilizes the following libraries:
- pandas
- numpy
- torch
- scikit-learn
- biopython

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/roeimichael/DNAclassification
cd DNAclassification
pip install -r requirements.txt
```

## Usage

Follow these steps to run the code:

1. **Download Data**: Run `ProduceData.py` to download the necessary data locally.
2. **Encode Data**: Execute `preprocessing.py` to encode the dataset.
3. **Train Model**: Now, with the dataset ready, run `modelTraining.py` for training.
4. **Visualize Results**: Use `ProcessingResults.py` to display the final results in graph format.

_Note: Some parameters can be adjusted directly in the code, and they may not be as modular as could be._


##Contributors
- Roei Michael - 322989666
- Bar Daabul - 324079243

