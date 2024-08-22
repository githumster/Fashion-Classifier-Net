# FashionClassifierNet

In this project, a basic neural network was implemented with all the necessary layers to ensure its correct operation. The network was subsequently tested on the Fashion MNIST dataset to evaluate its performance in clothing item classification.

## Implemented Components

### Layers
The following layers were implemented to construct the neural network:
- **Linear**: Fully connected layer for dense connections between neurons.
- **BatchNormalization**: Normalizes the inputs to improve training stability.
- **Dropout**: Regularization technique to prevent overfitting.
- **Sequential**: Container to stack layers in sequence.

### Criteria
The following loss functions were used to evaluate model performance:
- **MSECriterion**: Mean Squared Error for regression tasks.
- **ClassNLLCriterion**: Negative Log-Likelihood, typically used for classification.

### Activations
Several activation functions were implemented to introduce non-linearity:
- **SoftMax**: Converts logits to probabilities.
- **LogSoftMax**: Combines log and softmax for numerical stability.
- **ReLU**: Rectified Linear Unit, a popular activation function.
- **LeakyReLU**: Variation of ReLU allowing a small, non-zero gradient when the unit is not active.
- **ELU**: Exponential Linear Unit, smoother alternative to ReLU.
- **SoftPlus**: Smooth approximation of ReLU.

### Optimizers
To train the network, the following optimizer was implemented:
- **SGD with Momentum**: Stochastic Gradient Descent with momentum to accelerate convergence.

### Testing
For each layer and component, tests were written to verify correct functionality and ensure robust performance.

## Dataset

To evaluate the performance of the neural network, the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset was selected. This dataset consists of 70,000 grayscale images in 10 categories, each representing a different type of clothing item (e.g., T-shirts, trousers, shoes). The images are 28x28 pixels in size, making it a challenging yet manageable dataset for deep learning models.

### Categories
The dataset includes the following categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Installation

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FashionClassifierNet.git
   cd FashionClassifierNet
