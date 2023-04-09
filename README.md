# Breast-Cancer-Diagnosis-CNN
Breast Cancer Diagnosis Improvement Based on Image Processing Using Machine Learning Methods (CNN)

> **PSOWNNs-CNN: A Computational Radiology for Breast Cancer Diagnosis Improvement Based on Image Processing Using Machine Learning Methods**
>
> https://www.hindawi.com/journals/cin/2022/5667264/

If you find our work useful in your research please consider citing our paper:

```
@article{nomani2022psownns,
  title={PSOWNNs-CNN: a computational radiology for breast cancer diagnosis improvement based on image processing using machine learning methods},
  author={Nomani, Ashkan and Ansari, Yasaman and Nasirpour, Mohammad Hossein and Masoumian, Armin and Pour, Ehsan Sadeghi and Valizadeh, Amin},
  journal={Computational Intelligence and Neuroscience},
  volume={2022},
  year={2022},
  publisher={Hindawi}
}
```

## Description
This repository contains an implementation of a Convolutional Neural Network (CNN) for breast cancer detection. The BreastCancerCNN class is an object-oriented implementation of a CNN using TensorFlow 2. The model is trained on a dataset of breast cancer images and labels, and the trained model is evaluated on a separate test set. The goal of this project is to develop an accurate model for early breast cancer detection.

## Installation
To use the BreastCancerCNN class, you will need to have the following packages installed:

- TensorFlow 2
- NumPy
- scikit-learn

You can install these packages using the following command:
```
pip install -r requirements.txt
```
## Usage
To use the 'BreastCancerCNN' class, follow these steps:

Clone the repository to your local machine.

Download the breast cancer images and labels dataset and save them as 'breast_cancer_images.npy' and 'breast_cancer_labels.npy', respectively, in the repository's root directory.

Run the 'run.py' script to train and evaluate the model.

The 'run.py' script loads the dataset, trains the model, and evaluates its performance on a separate test set. To run the script, use the following command:

```
python run.py
```

## Dataset
The breast cancer images and labels dataset is not included in this repository due to its large size. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) or from other sources.

## Results

![](https://github.com/ArminMasoumian/Breast-Cancer-Diagnosis-CNN/blob/main/BCD-Results.png)

## Credits
The BreastCancerCNN class was developed by Armin Masoumian and is licensed under the [MIT License](https://github.com/ArminMasoumian/Breast-Cancer-Diagnosis-CNN/blob/main/LICENSE).

## References

- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
