#### This is the code for paper:

##### A hybrid model of ghost-convolution enlightened transformer for effective diagnosis of grape leaf disease and pest

DOI: https://doi.org/10.1016/j.jksuci.2022.03.006

https://www.sciencedirect.com/science/article/pii/S131915782200088X

##### Abstract

Disease and pest are the main factors causing grape yield reduction. Correct and timely identification of these symptoms are necessary for the vineyard. However, the commonly used CNN models limit their performance on leaf images with complex backgrounds, due to the lack of global receptive field. 

In this article, we propose an effective and accurate approach based on Ghost-convolution and Transformer networks for diagnosing grape leaf in field.

![Fig. 5.tif](C:\Users\luxy\OneDrive%20-%20zju.edu.cn\Code\pycode\project\p4_grape2021\GeT_model\model_pth\GeT_r.png)

First, a grape leaf disease and pest dataset containing 11 classes and 12,615 images, namely GLDP12k is collected. Ghost network is adopted as the convolutional backbone to generate intermediate feature maps with cheap linear operations. Transformer encoders with multi-head self-attention are integrated behind to extract deep semantic features. Then we get the Ghost enlightened Transformer model, namely GeT. After analyzing five hyper-parameters, the optimized GeT is transfer-learnt from ImageNet which provides a 4.3% accuracy bonus. As the results show, with 180 frame-per-second, 1.16 M weights and 98.14% accuracy, GeT surpasses other models, and is 1.7 times faster and 3.6 times lighter than MobilenetV3_large (97.7%). This study shows that the GeT model is effective and provides an optional benchmark for field grape leaf diagnosis.

##### Dependencies

pytorch

torchvision

timm

##### Use

The **GeTTest.py** is used for loading an grape leaf image and **predict** the symptom.
