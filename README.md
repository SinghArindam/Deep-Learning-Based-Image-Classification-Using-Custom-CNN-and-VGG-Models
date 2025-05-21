# Deep Learning-Based Image Classification Using Custom CNN and VGG Models

Comparison of Image Classification Architectures: VGG16, VGG19, and a Custom CNN


## Objective

The objective is to implement and demonstrate the architecture and capability of VGG-16, VGG-19, and a custom CNN for image classification. This involves building the model structures, training them on a standard dataset, and evaluating their performance.

## Dataset

The comparison utilizes the CIFAR-10 dataset.
* **Description**: Contains 60,000 color images of size $32 \times 32$ pixels, categorized into 10 distinct classes.
* **Classes**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
* **Split**: 50,000 training images (5,000 per class) and 10,000 test images (1,000 per class).

## Architecture Overviews

### VGG Networks (VGG16 and VGG19)

VGG networks are deep CNNs known for their simplicity and effectiveness in image recognition tasks. A key characteristic is the use of very small $3 \times 3$ convolutional filters throughout the network. The architecture consists of convolutional layers followed by max-pooling layers, and then fully connected layers. ReLU activation is used after every convolutional and fully connected layer, except the final output layer. The input to the network is typically a fixed-size RGB image, often $224 \times 224$.

### Custom CNN

Implemented a custom CNN model for image classification on CIFAR-10. The architecture is inspired by residual networks (ResNet) due to the presence of skip connections (identity mappings) within blocks. The model consists of several convolutional layers, batch normalization, ReLU activations, and max-pooling, structured into multiple blocks with residual connections. The final layers include average pooling and a fully connected layer for classification.

## VGG16 vs VGG19 vs Custom CNN: Key Differences

The primary differences lie in their depth, the specific arrangement of layers, and the use of architectural elements like residual connections.

| Feature                | VGG16                     | VGG19                     | Custom CNN (Resnet18) |
| :--------------------- | :------------------------ | :------------------------ | :------------------------- |
| Number of Weight Layers | 16                        | 19                        | Varies (depends on block configuration) |
| Convolutional Layers   | 13                        | 16                        | Multiple, organized in blocks |
| Fully Connected Layers | 3                         | 3                         | Typically 1 (for classification) |
| Architectural Element  | Sequential convolutional/pooling layers | Sequential convolutional/pooling layers | Includes Residual Blocks (Skip Connections) |
| Input Size (typical)   | $224 \times 224$          | $224 \times 224$          | $32 \times 32$ (for CIFAR-10) |

## Implementation Details

* **VGG16/VGG19**: Implemented using PyTorch. Trained on CIFAR-10 resized to $224 \times 224$. Optimizer: SGD (LR 0.001, Momentum 0.9). Loss: CrossEntropyLoss. Trained for 10 epochs.
* **Custom CNN**: Implemented using PyTorch. Trained on CIFAR-10 ($32 \times 32$). Optimizer: Adam. Loss: CrossEntropyLoss. Trained for 10 epochs.

## Results

The models were trained and evaluated on the CIFAR-10 dataset.

* **VGG-16 Test Accuracy**: ~80.98% (after 10 epochs)
* **VGG-19 Test Accuracy**: ~80.80% (after 10 epochs)
* **Custom CNN Test Accuracy**: ~87.68% (after 10 epochs)
* **Custom CNN Test Accuracy**: ~95.20%% (after 60 epochs)


## Losses and Validations
![training_history_pytorch.png](assets/training_history_pytorch.png?raw=true "training_history_pytorch.png")

## Conclusion

All three architectures demonstrate the capability to classify images on the CIFAR-10 dataset. While VGG16 and VGG19, with their focus on stacked small filters, achieve good performance, the custom CNN, which incorporates residual connections, shows a higher test accuracy in this comparison. This highlights the effectiveness of residual connections in potentially improving the performance of deep neural networks by mitigating issues like vanishing gradients.

This README provides a basic comparison. For a more detailed analysis, refer to the original documents and the provided code implementations.