# Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-Pytorch-Implementation
## Overview
### Data
The database consists of 49 512*512 image pairs and was split into 43 and 6 train and test sets, respectively. In order to use the code in this repository, the histology images and corresponding segmentation maps need to be merged as shown below.
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/Slide_11_11_2.png).

## Baseline
The baseline architecture was U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). The model was trained for 200 epochs using binary crossentropy loss. The dice similarity coefficient was 0.637.

## GAN-based Segmentation
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/gan_image.png)

## Dependencies
 - [x] Pytorch >= 0.4.0
 - [x] Python >= 3.5
 - [x] Numpy 


Run main.py
You will see the predicted results of test image in data/membrane/test
## Results
### Qualitative
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/results.png)

### Quantitative 
| Model  | Dice Measure |
| ------------- | ------------- |
| Unet with no data augmentation (Baseline) | 0.637  |
|  Unet with data augmentation  | 0.693  |
|  cGAN with data augmentation  | 0.723  |

