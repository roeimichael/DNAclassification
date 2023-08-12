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

\```bash
git clone [https://github.com/yourusername/COVID-19-Variant-Classification.git](https://github.com/roeimichael/DNAclassification)
cd DNAclassification
pip install -r requirements.txt
\```

## Usage

- to run the code you would first need to download the data locally by running ProduceData.py 
- than encode it using preprocessing.py
- now that you have the dataset all ready for training run modelTraining.py
- and to show final results in graph formatting  run ProcessingResults.py
- **make sure to notice that some of the parameters are adjustable from the code itself and arent as modular and easy to use as could be**

##Contributors
- Roei Michael - 322989666
- Bar Daabul - 324079243

