# Few-Shot Learning for Reducing Data Dependency in AI Models

A deep learning project implementing Few-Shot Learning using Prototypical Networks to reduce the data requirements of AI models in medical image classification.

## Project Overview

Traditional deep learning models require large labeled datasets to achieve high accuracy. In domains such as medical imaging, collecting labeled data is expensive and time-consuming.

This project explores Few-Shot Learning, a paradigm where models learn to generalize from very small training samples.

The model learns a feature embedding space where samples belonging to the same class are clustered together, enabling accurate classification even with very limited labeled data.

This repository implements:

Prototypical Networks (ProtoNet)

Few-shot episodic training

Baseline CNN comparison

Embedding visualization using t-SNE

## Key Features

Few-Shot Learning implementation using Prototypical Networks

Training with N-way K-shot learning

Baseline CNN model for comparison

Medical MNIST dataset support

Visualization of learned embeddings

Modular PyTorch implementation

## Dataset

The project uses the Medical MNIST dataset, which contains grayscale medical images across multiple diagnostic categories.

Dataset classes used:

AbdomenCT

BreastMRI

ChestCT

CXR

Hand

HeadCT

Each image represents a medical scan belonging to a specific anatomical region.

Image Properties

| Property          | Value     |
| ----------------- | --------- |
| Image Type        | Grayscale |
| Input Size        | 28×28     |
| Channels          | 1         |
| Number of Classes | 6         |


## Few-Shot Learning Concept

Few-Shot Learning trains models using episodes instead of standard batches.

Each episode contains:

Support Set → few labeled examples per class

Query Set → samples used to evaluate classification

Example Episode

5-way 5-shot learning:

5 classes

5 support samples per class

Query samples used for evaluation

The model learns to compare samples rather than memorize data.

## Prototypical Networks

Prototypical Networks represent each class using a prototype vector.

A prototype is the mean embedding of all support samples belonging to that class.

Prototype Formula

<img width="694" height="274" alt="image" src="https://github.com/user-attachments/assets/338c7479-d82c-40cc-8336-95cb6b405bad" />


## Classification Rule

<img width="686" height="319" alt="image" src="https://github.com/user-attachments/assets/8708350b-7800-45f8-9f06-1eb643e56ed5" />

## Model Architecture

The encoder network converts images into embedding vectors.

Protonet Encoder

<img width="662" height="438" alt="image" src="https://github.com/user-attachments/assets/bee8affa-1f42-482d-b366-66d269dc893b" />


The embedding space allows similar images to be grouped together.

## Baseline CNN Model

A conventional CNN classifier is implemented for comparison.

Architecture:

<img width="589" height="209" alt="image" src="https://github.com/user-attachments/assets/9bfd8b6f-4bf8-4bb8-8fd9-24b8c0a794bc" />

Unlike ProtoNet, this model requires large datasets to generalize effectively.

## Training Methodology

The ProtoNet is trained using episodic training, which simulates few-shot learning scenarios during training.

Each training episode:

1. Sample N classes

2. Select K support samples per class

3. Compute class prototypes

4. Embed query samples

5. Compute distances to prototypes

6. Apply softmax loss

7. Update model parameters

## Training Configuration

| Parameter     | Value   |
| ------------- | ------- |
| Framework     | PyTorch |
| Optimizer     | Adam    |
| Learning Rate | 0.001   |
| Epochs        | 20      |
| N-way         | 5       |
| K-shot        | 5       |
| Query Samples | 5       |

## Visualization using t-SNE

To analyze learned representations, embeddings are projected to 2D space using t-SNE.

Purpose:

1. Evaluate cluster separation

2. Validate embedding quality

3. Visualize class relationships

Well-trained ProtoNet embeddings show clear cluster separation between classes.

Project Structure

<img width="737" height="545" alt="image" src="https://github.com/user-attachments/assets/3c1bd97d-e57c-4e83-9004-5ad931ad08c7" />


## Installation

Clone the repository:

git clone https://github.com/yourusername/few-shot-learning.git
cd few-shot-learning

Install dependencies:

pip install -r requirements.txt

Dependencies include:

1. PyTorch

2. NumPy

3. Scikit-learn

4. Matplotlib

5. Pillow

## Running the Project

Launch Jupyter Notebook

Open: few_shot_learning.ipynb

Run all cells sequentially to:

1. Load dataset

2. Train ProtoNet

3. Train baseline CNN

4. Evaluate results

5. Visualize embeddings

## Results

The ProtoNet model demonstrates strong performance even with limited training samples.

Observations:

1. Few-shot learning significantly reduces data requirements

2. ProtoNet embeddings form well-separated clusters

3. Baseline CNN requires larger datasets to perform similarly

## Applications

Few-Shot Learning is particularly useful in domains with limited labeled data.

Applications include:

1. Medical image classification

2. Rare disease detection

3. Low-resource NLP tasks

4. Fraud detection

5. Robotics and autonomous systems

## Future Work

Possible improvements include:

1. Implement Siamese Networks

2. Explore MAML (Model-Agnostic Meta Learning)

3. Use deeper architectures like ResNet

4. Increase dataset diversity

5. Hyperparameter optimization

## Research References

Key papers related to this work:

Prototypical Networks for Few-Shot Learning
Jake Snell, Kevin Swersky, Richard Zemel
NeurIPS 2017

Matching Networks for One Shot Learning
Vinyals et al.

Model-Agnostic Meta-Learning (MAML)
Finn et al.

Author

Ansh Kumar Sonker
B.Tech Computer Science Engineering

Project Focus:

1. Few-Shot Learning

2. Machine Learning

3. Medical Image Classification

4. Data Efficient AI

## License

This project is intended for academic and educational purposes.
