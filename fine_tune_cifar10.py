'''Obtain pirated model by fine tuning or feature extraction

by ZHENG yue, Wang Si
2021-12-27
'''


import sys
sys.path.append('/mnt/ssd1/zhengyue/Models/')

#from __future__ import print_function
#from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from datetime import datetime
import os
import copy
from definition import train_model, set_parameter_requires_grad, initialize_model
# from CIFAR_pretrained_models_master.cifar_pretrainedmodels import resnet # original manuscript
from pytorch_resnet_cifar10_master import resnet 
# Import vgg11-cifar10, densenet161-cifar10 pretrained models
from PyTorch_CIFAR10_master.cifar10_models import vgg, densenet

def main():

	#print("PyTorch Version: ",torch.__version__)
	#print("Torchvision Version: ",torchvision.__version__)

	now = datetime.now()
	start = now.strftime("%D:%H:%M:%S")
	
	num_classes = 10
	batch_size = 128
	num_epochs = 100
	input_size = 32
	# Flag for feature extracting. When False, we finetune the whole model,
	# when True we only update the reshaped layer params
	
	
	with open('./log.txt', 'a') as f:
		f.write('-----'*10+'\n')
		f.write(start+'\n')
		f.write('num_epochs:{}\n'.format(num_epochs))
		f.write('model name: cifar10-{}\n'.format(model_name))
		f.write('fine tune or feature extraction: {}\n'.format(sel))

	
	data_dir = "./datasets/cifar10"
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	a = False
	
	if model_name == 'resnet20':
		path = '/mnt/ssd1/zhengyue/Models/pytorch_resnet_cifar10_master/save_resnet20/checkpoint.th'
		checkpoint = torch.load(path)
		model_ft = torch.nn.DataParallel(resnet.__dict__['resnet20']())
		model_ft.cuda()
		model_ft.load_state_dict(checkpoint['state_dict'])
		a = True
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.module.linear.in_features
		model_ft.module.linear = nn.Linear(num_ftrs,num_classes)
		
	elif model_name == 'vgg11':
		model_ft = vgg.vgg11_bn(pretrained=True)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		
	elif model_name == 'densenet161':
		model_ft = densenet.densenet161(pretrained=True,device=device)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs,num_classes)
		
	else:
		print('Unknown model name input!')
	

	# Print the model we just instantiated
	print(model_ft)
	model_ft = model_ft.to(device)


	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomCrop(size=32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(
			mean=[0.4914, 0.4822, 0.4465], 
			std=[0.2023, 0.1994, 0.2010])
		]),
		'val': transforms.Compose([
			transforms.RandomCrop(size=32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(
			mean=[0.4914, 0.4822, 0.4465], 
			std=[0.2023, 0.1994, 0.2010])
		]),
	}


	print("Initializing Datasets and Dataloaders...")

	# Create training and validation datasets
	image_datasets = {
		'train': torchvision.datasets.CIFAR10(root=data_dir, train=True,download=False, transform=data_transforms['train']),
		'val': torchvision.datasets.CIFAR10(root=data_dir, train=False,download=False, transform=data_transforms['val'])
	}
	# Create training and validation dataloaders
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

	

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
	params_to_update = model_ft.parameters()
	print("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name,param in model_ft.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print("\t",name)
	else:
		for name,param in model_ft.named_parameters():
			if param.requires_grad == True:
				print("\t",name)

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

	# Setup the loss fxn
	criterion = nn.CrossEntropyLoss()
	
	stat_epoch = 0
	# Train and evaluate
	model_ft, hist, loss, epoch = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, stat_epoch, num_epochs=num_epochs, is_inception=(model_name=="inception"))
	
	saved_PATH = "./saved_models/cifar10_ftfe/"
	if a:
		print('cannot save model due to AttributeError: Cannot pickle local object "BasicBlock.__init__.<locals>.<lambda>"')
	else:
		torch.save(model_ft, os.path.join(saved_PATH, sel+'_'+model_name+"_model_"+str(stat_epoch)+"_"+str(epoch)+".pth"))
		
	torch.save({
			'epoch': epoch,
			'model_state_dict': model_ft.state_dict(),
			'optimizer_state_dict': optimizer_ft.state_dict(),
			'loss': loss,
			}, os.path.join(saved_PATH,sel+'_'+model_name+"_checkpoint_"+str(stat_epoch)+"_"+str(epoch)+".pth"))

if __name__ == "__main__":

	model_name_list = ['resnet20', 'resnet20']
	sel_list = ['ft', 'fe']
	for i in range(len(model_name_list)):
		model_name = model_name_list[i]
		sel = sel_list[i]
		if sel == 'fe':
			feature_extract = True
		else:
			feature_extract = False
			
		print('Start fine tune or feature extraction cifar10-'+ model_name+'......')
		main()
	
