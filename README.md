Description
This project explores a hybrid approach that combines Reinforcement Learning (RL) with meta-decision-making strategies to intelligently manage an experience buffer used for training a Convolutional Neural Network (CNN) on the MNIST dataset. The agent learns which data to collect and prioritize for improving classification accuracy.

Features
Custom experience buffer with sampling strategy
CNN model trained on prioritized data
Cosine similarity for relevance scoring
Evaluation on the MNIST dataset
Reinforcement-learning-inspired data selection logic

Core Concepts
MetaDecisionNet: Using an RL-like policy to determine which training samples are most useful.
Experience Buffer: Maintains a pool of training samples and selects the most relevant ones for training.
Cosine Similarity: Scores similarity between data features to guide selection.
Convolutional Neural Network: A basic CNN implemented in PyTorch for digit classification.

File Structure
final_experience_buffer_and_cnn_model.ipynb: Main notebook combining data selection strategy and CNN training.

🛠️ Requirements
Python 3.x
PyTorch
torchvision
numpy
scikit-learn

You can install the necessary packages using:
bash
Copy
Edit
pip install torch torchvision numpy scikit-learn

Results
After training, the model is evaluated on a test set to compute accuracy. The training loop and evaluation logic are provided in the notebook.

How to Run
Open the notebook in Google Colab or Jupyter Lab (CPU run)
Open the notebook in CREW or any GPU supported domain (GPU run, faster)

Run all cells to:
Install dependencies (if needed).
Load and preprocess MNIST.
Initialize the experience buffer and model.
Train and evaluate.

Acknowledgments
MNIST dataset provided by Yann LeCun et al.
PyTorch team for the deep learning framework

