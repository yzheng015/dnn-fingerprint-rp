import sys
sys.path.append('/mnt/ssd1/zhengyue/Models/')


import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
import os
import torch.nn.utils.prune as prune
# Import resnet-cifar dataset pretrained models
from CIFAR_pretrained_models_master.cifar_pretrainedmodels import resnet

# get cifar100 model architecture for vgg11, densenet161
sys.path.append('/mnt/ssd1/zhengyue/Models/pytorch-cifar100-master') 
# Import cifar100-vgg11, cifar100-densenet161 pretrained models
from models.vgg import vgg11_bn
from models.densenet import densenet161


from definition_wp import train_model, set_parameter_requires_grad, initialize_model
from datetime import datetime

#from PyTorch_CIFAR10_master.cifar10_models import *
def main():

	#print("PyTorch Version: ",torch.__version__)
	#print("Torchvision Version: ",torchvision.__version__)

	now = datetime.now()
	start = now.strftime("%D:%H:%M:%S")
	
	with open('./log.txt', 'a') as f:
		f.write('-----'*10+'\n')
		f.write(start+'--filter pruning--\n')
		f.write('model name: cifar100-{}\n'.format(model_name))
		f.write('frac: {}/16\n'.format(frac1))

	frac = frac1/16
	batch_size = 128
	num_epochs = 100
	input_size = 32
	feature_extract = False
	data_dir = "./datasets/cifar100"

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if model_name  == 'resnet20':
		model = resnet.cifar_resnet20(pretrained='cifar100')

		parameters_to_prune = (
			(model.conv1, 'weight'),
			(model.layer1[0].conv1, 'weight'),
			(model.layer1[0].conv2, 'weight'),
			(model.layer1[1].conv1, 'weight'),
			(model.layer1[1].conv2, 'weight'),
			(model.layer1[2].conv1, 'weight'),
			(model.layer1[2].conv2, 'weight'),
			(model.layer2[0].conv1, 'weight'),
			(model.layer2[0].conv2, 'weight'),
			(model.layer2[1].conv1, 'weight'),
			(model.layer2[1].conv2, 'weight'),
			(model.layer2[2].conv1, 'weight'),
			(model.layer2[2].conv2, 'weight'),
			(model.layer3[0].conv1, 'weight'),
			(model.layer3[0].conv2, 'weight'),
			(model.layer3[1].conv1, 'weight'),
			(model.layer3[1].conv2, 'weight'),
			(model.layer3[2].conv1, 'weight'),
			(model.layer3[2].conv2, 'weight'),
		)
	elif model_name == 'vgg11':
		path = '/mnt/ssd1/zhengyue/Models/Target_models/cifar100-vgg11-192-best.pth'
		model = vgg11_bn()
		model.load_state_dict(torch.load(path))
		parameters_to_prune = (
			(model.features[0], 'weight'),
			(model.features[4], 'weight'),
			(model.features[8], 'weight'),
			(model.features[11], 'weight'),
			(model.features[15], 'weight'),
			(model.features[18], 'weight'),
			(model.features[22], 'weight'),
			(model.features[25], 'weight'),
		)

	elif model_name == 'densenet161':
		path = '/mnt/ssd1/zhengyue/Models/Target_models/cifar100-densenet161-183-best.pth'
		model = densenet161()
		model.load_state_dict(torch.load(path))
		parameters_to_prune = (
			(model.conv1, 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_0.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_0.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_1.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_1.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_2.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_2.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_3.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_3.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_4.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_4.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_5.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_0.bottle_neck_layer_5.bottle_neck[5], 'weight'),
			(model.features.transition_layer_0.down_sample[1], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_0.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_0.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_1.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_1.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_2.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_2.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_3.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_3.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_4.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_4.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_5.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_5.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_6.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_6.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_7.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_7.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_8.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_8.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_9.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_9.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_10.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_10.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_11.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_1.bottle_neck_layer_11.bottle_neck[5], 'weight'),
			(model.features.transition_layer_1.down_sample[1], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_0.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_0.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_1.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_1.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_2.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_2.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_3.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_3.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_4.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_4.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_5.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_5.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_6.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_6.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_7.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_7.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_8.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_8.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_9.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_9.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_10.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_10.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_11.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_11.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_12.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_12.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_13.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_13.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_14.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_14.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_15.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_15.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_16.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_16.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_17.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_17.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_18.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_18.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_19.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_19.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_20.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_20.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_21.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_21.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_22.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_22.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_23.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_23.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_24.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_24.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_25.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_25.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_26.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_26.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_27.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_27.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_28.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_28.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_29.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_29.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_30.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_30.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_31.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_31.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_32.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_32.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_33.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_33.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_34.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_34.bottle_neck[5], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_35.bottle_neck[2], 'weight'),
			(model.features.dense_block_layer_2.bottle_neck_layer_35.bottle_neck[5], 'weight'),
			(model.features.transition_layer_2.down_sample[1], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_0.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_0.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_1.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_1.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_2.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_2.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_3.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_3.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_4.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_4.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_5.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_5.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_6.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_6.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_7.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_7.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_8.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_8.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_9.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_9.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_10.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_10.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_11.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_11.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_12.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_12.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_13.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_13.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_14.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_14.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_15.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_15.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_16.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_16.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_17.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_17.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_18.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_18.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_19.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_19.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_20.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_20.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_21.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_21.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_22.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_22.bottle_neck[5], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_23.bottle_neck[2], 'weight'),
			(model.features.dense_block3.bottle_neck_layer_23.bottle_neck[5], 'weight'),)
	else:
		print('Unknown model name input!')
	
	model = model.to(device)
	####### manual filter_prune, l1 norm ##########

	for turple in parameters_to_prune:
		num_filter = turple[0].weight.shape[0]
		l1 = torch.sum(torch.abs(turple[0].weight),[1,2,3])
		min_k = int(frac*num_filter)
		values, inds = torch.topk(l1,min_k,largest=False)
		with torch.no_grad():
			 turple[0].weight[[inds.detach().cpu().numpy()]]=0
		
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomCrop(size=32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(
			mean=[0.5071, 0.4865, 0.4409], 
			std=[0.2009, 0.1984, 0.2023])
		]),
		'val': transforms.Compose([
			transforms.RandomCrop(size=32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(
			mean=[0.5071, 0.4865, 0.4409], 
			std=[0.2009, 0.1984, 0.2023])
		]),
	}


	print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
	image_datasets = {
		'train': torchvision.datasets.CIFAR100(root=data_dir, train=True,download=False, transform=data_transforms['train']),
		'val': torchvision.datasets.CIFAR100(root=data_dir, train=False,download=False, transform=data_transforms['val'])
	}
# Create training and validation dataloaders
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
	
	set_parameter_requires_grad(model, feature_extract)
	params_to_update = model.parameters()
	print("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print("\t",name)
	else:
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				print("\t",name)
	

	#print("frac: 11/16"+"\n")

	optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
	

	criterion = nn.CrossEntropyLoss()
	
	stat_epoch = 0
# Train and evaluate
	model_ft, hist, loss, epoch = train_model(model, dataloaders_dict, criterion, optimizer_ft, stat_epoch, num_epochs=num_epochs, is_inception=(model_name=="inception"))
	print("frac: "+str(frac1)+"/16"+"\n")
	saved_PATH = "./saved_models/cifar100_"+model_name+"_fp/"
	torch.save(model_ft, os.path.join(saved_PATH,"fp"+str(frac1)+"_model_"+str(stat_epoch)+"_"+str(epoch)+".pth"))
	torch.save({
			'epoch': epoch,
			'model_state_dict': model_ft.state_dict(),
			'optimizer_state_dict': optimizer_ft.state_dict(),
			'loss': loss,
			}, os.path.join(saved_PATH,"fp"+str(frac1)+"_checkpoint_"+str(stat_epoch)+"_"+str(epoch)+".pth"))

if __name__ == "__main__":
	
	#model_name = input('Model to be tested: ')
	#frac1 = int(input('frac 1-16: '))
	model_name_list = ['vgg11', 'densenet161']
	#frac1_list = [10, 13, 12, 12]
	#for i in range(len(model_name_list)):
	for model_name in model_name_list:
		for frac1 in range(1, 10):
		#model_name = model_name_list[i]
		#frac1 = frac1_list[i]
			main()
