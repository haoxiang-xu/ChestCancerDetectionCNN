# Lung Cancer Classification CNN

### About this project
###### Since the traditional manual method of classifying lung cancer CT photos is very time-consuming, a new automated, highly accurate classification method is urgently needed. This has led to the use of the Convolutional Neural Network (CNN) for image classification and recognition as the best choice in the medical field. However, due to the problem of vanishing gradient when the layer depth of these models is too deep, the gradient will be vanishingly small and this leads to extremely low learning efficiency, which may even be close to 0. In this paper, a reimplemented CNN model based on ResNet is used to improve low learning efficiency. Three different CNNs are also compared and illustrate that the deeper neural networks have better learning efficiency and higher accuracy for classifying lung CT images. Specifically, LeNet, AlexNet, and ResNet are reimplemented, where the first two CNNs represent traditional models. By comparing the accuracy of the three models, we conclude that the traditional model is sufficient for the task of classifying lung cancer models and has an acceptable accuracy rate when the number of training sessions reaches a certain level, but it is not competitive in learning efficiency and accuracy for the same number of training sessions compared to the subsequent models with more neurons.

### How to run the code
- Clone github repository https://github.com/bananamilkt/ChestCancerDetectionCNN.git
- install required python packages 
  - flask
  - numpy
  - tensorflow
  - matplotlib.pyplot
  - tensorflow_wavelets.Layers.DWT
- run CNN.ipynb to train the models, since there's no pretrained data stored in the repository. All importing code are already included.
- run _init_.py which is a flask base python web page. It will load up the pretrained model and classify images.

### Source of data
##### Dataset
- https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
    ###### Author Page https://www.kaggle.com/mohamedhanyyy
    ###### Chest CT-Scan images Dataset provided by Mohamed Hany Data Scientist at QN Academy.

### Credits of implementation
##### Referencing Code
- https://www.kaggle.com/code/likithavadlapudi/notebook095d438171
    ###### Author Page https://www.kaggle.com/likithavadlapudi 
    ###### CNN implementation for Chest Cancer Detection by Likitha Vadlapudi student at vignan's institute of information and technology.
- https://keras.io/api/data_loading/image/
- https://www.tensorflow.org/tutorials/images/cnn


