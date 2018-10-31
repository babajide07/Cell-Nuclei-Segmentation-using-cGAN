# Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-Pytorch-Implementation
## Overview
### Data
The database consists of 49 512*512 image pairs and was split into 43 and 6 train and test sets, respectively. In order to directly process the image, the histology images and corresponding segmentation maps are merged as shown below.
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/Slide_11_11_2.png).

## Baseline
The baseline architecture was U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
## GAN-based Segmentation
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/gan_image.png)

## Results
### Qualitative
![alt text](https://github.com/babajide07/Cell-Nuclei-Segmentation-from-Histology-images-using-Conditional-Generative-Adversarial-Network-/blob/master/Results/results.png)

### Quantitative 
| Model  | Dice Measure |
| ------------- | ------------- |
| Unet with no data augmentation (Baseline) | 0.637  |
|  Unet with data augmentation  | 0.693  |
|  cGAN with data augmentation  | 0.723  |

