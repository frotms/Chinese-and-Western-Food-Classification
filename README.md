# Chinese and Western Food Classification
This repo contains code for running food classification by pytorch. It uses 208 chinese food classes and 101 western food classes(309-classes).  
This repo is designed for those who want to start their projects of image classification.  
It provides fast experiment setup and attempts to maximize the number of projects killed within the given time.  
It includes a few Convolutional Neural Network modules.You can build your own dnn easily.  
## Requirements
* torch==0.4.1
* torchvision==0.2.1
* opencv
* Pillow
## model
- [mobilenetV2](https://github.com/ericsun99/MobileNet-V2-Pytorch) (`mobilenet_v2`)
### pre-trained model
you can download pretrain model in model_dir(`CWFood_model.pth`)
## usage
### Inference
	python3 inference.py --image test.jpg
![](https://i.imgur.com/golIRSs.jpg)  
top-5:  
Mapo_Tofu: 70.02458572387695%  
Kung_Pao_Chicken: 5.765869095921516%  
Spicy_Chicken: 5.33505454659462%  
Home_style_sauteed_Tofu: 3.10797281563282%  
Double_cooked_pork_slices: 2.423858270049095%    
![](https://i.imgur.com/CRrh3ul.jpg)  
top-5:  
pizza: 57.676124572753906%  
garlic_bread: 8.819431811571121%  
macaroni_and_cheese: 6.301581114530563%  
paella: 4.138006269931793%  
Pizza: 3.8569435477256775%  
### Experiments
There is integrated with the project using `tensorboardX` library which porved to be very useful as there is no official visualization library in pytorch. There is the learning curves for the food dataset experiment(top-1 acc: 69.66%).
![](https://i.imgur.com/dpeqoZQ.jpg)  
### Labels
[train_data.txt](https://github.com/frotms/Chinese-and-Western-Food-Classification/blob/master/labels/train_data.txt)  
[val_data.txt](https://github.com/frotms/Chinese-and-Western-Food-Classification/blob/master/labels/val_data.txt)  
## References
1.[https://sites.google.com/view/chinesefoodnet](https://sites.google.com/view/chinesefoodnet)  
2.[http://www.vision.ee.ethz.ch/datasets_extra/food-101](http://www.vision.ee.ethz.ch/datasets_extra/food-101)  
3.[https://pytorch.org](https://pytorch.org)  
4.[https://github.com/ericsun99/MobileNet-V2-Pytorch](https://github.com/ericsun99/MobileNet-V2-Pytorch)  
5.[https://github.com/frotms/image_classification_pytorch](https://github.com/frotms/image_classification_pytorch)  
