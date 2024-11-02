# Flower Species Classification with CNN
This project classifies images of flowers into five species using a Convolutional Neural Network (CNN). The model is trained on a dataset with images of lotus, orchids, roses, sunflowers, lilis and tulips.

## Project Overview
This deep learning project implements a CNN-based classification model that achieves high accuracy in categorizing flower species. Techniques like transfer learning with EfficientNetB0, VGG16, and InceptionV3 have been explored to optimize performance. The final model is saved as final_model.h5 for future predictions and integration into applications.

## Table of Contents
Project overview  
Dataset  
Model Architecture  
Training and evaluation  
Results  
Future Work  
Contributing  
License  

## Dataset
The dataset consists of images of five types of flowers:

Lotus: 1000 images  
Orchids: 1000 images  
Rose: 784 images  
Sunflower: 1000 images  
Lily: 1000 images  
Tulip: 1000 images  
Each flower category has its own folder in the dataset. Images are used for training and validating the model to classify new flower images accurately.  

## Model Architecture
The project uses multiple CNN architectures to test and optimize model performance:

**EfficientNetB0:** Provided the best results with balanced accuracy and minimal overfitting.  
**VGG16:** Delivered good results but had overfitting issues.  
**InceptionV3:** Helped in achieving lower loss values, especially for the 5-class problem.  
The final model, based on EfficientNetB0, is saved as final_model.h5.  

## Training and Evaluation
The model is trained on TensorFlow with Jupyter Notebook. Transfer learning was used, leveraging pre-trained models to enhance accuracy. After training for 80 epochs, the model achieved a validation accuracy of 90%.

Current Performance Metrics:  
Training Accuracy: 88.5%  
Validation Accuracy: 90%  
Cross-Entropy Loss: Targeted below 0.2 for robust predictions  

## Results
After extensive training, the model accurately predicts flower species. However, there is an issue where the model predicts "tulip" for all custom inputs, indicating potential class imbalance or dataset issues. This will be addressed in future updates.

## Future Work
Resolve the "tulip" bias in predictions on custom inputs.  
Experiment with additional data augmentation techniques.  
Explore ensemble methods like bagging to further improve accuracy.  
Develop an interactive user interface for easy prediction.  

## Contributing
Contributions are welcome! Please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
