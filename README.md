# Semi-Supervised Methods for Classification of Hyperspectral Images with Deep Learning
This repo is the main repositiory of my master thesis at Politecnico di Milano and KTH 2020. 

## Abstract
Hyperspectral images (HSI) can reveal more patterns than regular images. The dimensionality is high with a wider spectrum for each pixel. Few labeled datasets exists while unlabeled data is abundant. This makes semi-supervised learning well suited for HSI classification. Leveraging new research in deep learning and semi-supervised methods, two models called FixMatch and Mean Teacher was adapted to gauge the effectiveness of consistency regularization methods for semi-supervised learning on HSI classification. 

Traditional machine learning methods such as SVM, Random Forest and XGBoost was compared in conjunction with two semi-supervised machine learning methods, TSVM and QN-S3VM, as baselines. The semi-supervised deep learning models was tested with two networks, a 3D and 1D CNN. 

To enable the use of consistency regularization several new data augmentation methods was adapted to the HSI data. Current methods are few and most rely on labeled data, which is not available in this setting. The data augmentation methods presented proved useful and was adapted in a automatic augmentation scheme. 

The accuracy of the baseline and semi-supervised methods showed that the SVM was best in all cases. Neither semi-supervised method showed consistently better performance than their supervised equivalent.

## Structure
The notebooks were mostly made for exploring different options for research, code parts, applications and bugs. 

The datasets used are taken from https://tinyurl.com/ieee-grsl and are presented by Nalepa et al in https://arxiv.org/pdf/1811.03707.pdf to remedy a common validation issue in HSI classification.

Initial framework for testing is credited to DeepHyperX https://github.com/nshaud/DeepHyperX.

In all scripts the general path has been anonymized. Some parts are somewhat hard-coded, like the paths, and need to be added correctly.
