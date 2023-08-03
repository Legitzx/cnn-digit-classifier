# Digit Classifier Convolutional Neural Network (CNN)

### Simple convolutional neural network built using just Python and NumPy that classifies digits using the MNIST dataset. 

### Best accuracy on test set to date: 93.9% (training time is the constraint)

### Convolutional Neural Network Architecture:

Input: 28x28 (784 pixels) image (MNIST)

Convolutional Layer #1:
 - Filter size: 5x5
 - Num Filters: 32
 - Feature Map (Output): 28 - 5 + 1 = 24x24
 
Max Pooling Layer #1:
 - Filter size: 2x2
 - Num Filters 32
 - Output: 24 / 2 = 12x12

Convolutional Layer #2:
 - Filter size: 5x5
 - Num Filters: 64
 - Feature Map (Output): 12 - 5 + 1 = 8x8

Max Pooling Layer #2:
 - Filter size: 2x2
 - Num Filters 64
 - Output: 8 / 2 = 4x4

[Flatten] -> 64 4x4 = 1024 input neurons as a column vector

Dense #1:
 - Num Neurons: 128
 - Activation: ReLU
 
Dense #2: (Output)
 - Num Neurons: 10
 - Activation: Softmax

Network Features:
 - Optimizer: Stochastic Gradient Descent (SGD)
 - Loss Function: Categorical Cross-Entropy
 - Weight Initialization: Kaiming (He) Weight Initialization (fixed exploding gradients)


#### Useful Resources:
 - Forward Pass Explanations:
   - [Forward Pass Explanation YouTube Video](https://youtu.be/HGwBXDKFk9I)
   - [MIT Deep Learning CNN Lecture YouTube Video](https://youtu.be/NmLK_WQBxB4)
   - [Understanding CNNs with multiple Convolutional Layers](https://youtu.be/VF4BDE7uqY0)
 - Backward Pass Explanations:
   - [CNN Backpropagation Medium Post](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)