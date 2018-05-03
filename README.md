# Skin-Cancer-Detection: A neural network based classifier for benign and malign skin lesions

## Authors
Florence Lopez, Jonas Einig and Julian Späth (Students at University of Tübingen)

## The Project
Skin cancer is one of the most common cancer diseases on the whole world. Every year more or less than 18.000 people are taken by this type of cancer and it is responsible for five percent of the cancer deaths. An early recognition of the malign skin lesion is thereby urgent for a suitable treatment.

The classifier developed in this work is based on the paper "Dermatologist-level classification of skin cancer with deep neural networks" of [Esteva et. al. (2017)](https://www.nature.com/articles/nature21056) and classifies dermoscopy or high resolution images of a skin lesion reliably into malign and benign lesions. Still it should be used with caution. 

Our final classifier achieved a sensitivity of 71%, specificity of 89% and a F2-Score of 0.6 with the decision threshold set to 0.35, where all above is classified as malign.

## Tools 
### Implementation
+ [Python](https://www.python.org/download/releases/3.0/) - Version 3.6
+ [Tensorflow](https://www.tensorflow.org) - Version 1.8
### Evaluation
+ [MATLAB](https://de.mathworks.com/products/matlab.html) - Version 9.4
+ [scikit-learn](http://scikit-learn.org/stable/) - Version 0.19

## Android App
We are also working on an [Android App](https://github.com/spaethju/skin-cancer-detection-app) which allows a mobile application of the classifier.
