'''Obtain pirated model by fine tuning or feature extraction

by ZHENG yue, Wang Si
2021-12-27
'''

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
from definition import train_model, set_parameter_requires_grad,initialize_model



def main():

	#print("PyTorch Version: ",torch.__version__)
	#print("Torchvision Version: ",torchvision.__version__)


	num_classes = 10
	batch_size = 64
	num_epochs = 100
	input_size = 224
	data_dir = "./datasets/imagenet10_data"
	model_path = '/mnt/ssd1/zhengyue/Models'

	
	now = datetime.now()
	start = now.strftime("%D:%H:%M:%S")

	with open('./log.txt', 'a') as f:
		f.write('-----'*10+'\n')
		f.write(start+'\n')
		f.write('num_epochs:{}\n'.format(num_epochs))
		f.write('model name: imagenet10-{}\n'.format(model_name))
		f.write('fine tune or feature extraction: {}\n'.format(sel))

	
	if model_name == 'resnet18':
		imagenet10_resnet18 = model_path + '/Target_models/imagenet10_resnet18_model_0_29.pth'
		model_ft = torch.load(imagenet10_resnet18)
		
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs,num_classes)
		
	elif model_name == 'vgg11':
		imagenet10_vgg11 = model_path + '/Target_models/imagenet10_vgg11_model_0_49.pth'
		model_ft = torch.load(imagenet10_vgg11)
		
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		
	elif model_name == 'densenet161':
		imagenet10_densenet161 = model_path + '/Target_models/imagenet10_densenet161_model_0_99.pth'
		model_ft = torch.load(imagenet10_densenet161)
		
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs,num_classes)
		
	else:
		print('Unknown model name input!')


	# Print the model we just instantiated
	print(model_ft)


	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}


	print("Initializing Datasets and Dataloaders...")

	# Create training and validation datasets
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	# Create training and validation dataloaders
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

	# Detect if we have a GPU available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# Send the model to GPU
	model_ft = model_ft.to(device)

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

	saved_PATH = "./saved_models/imagenet10_ftfe/"  
	torch.save(model_ft, os.path.join(saved_PATH, sel+'_'+model_name+"_model_"+str(stat_epoch)+"_"+str(epoch)+".pth"))
	torch.save({
			'epoch': epoch,
			'model_state_dict': model_ft.state_dict(),
			'optimizer_state_dict': optimizer_ft.state_dict(),
			'loss': loss,
			}, os.path.join(saved_PATH, sel+'_'+model_name+"checkpoint_"+str(stat_epoch)+"_"+str(epoch)+".pth"))

if __name__ == "__main__":
	model_name_list = ['resnet18', 'resnet18', 'vgg11', 'vgg11', 'densenet161', 'densenet161']
	sel_list = ['ft', 'fe','ft', 'fe','ft', 'fe']
	for i in range(len(model_name_list)):
		model_name = model_name_list[i]
		sel = sel_list[i]
		if sel == 'fe':
			feature_extract = True
		else:
			feature_extract = False
			
		print('Start fine tune or feature extraction ImageNet10-'+ model_name+'......')
		main()
