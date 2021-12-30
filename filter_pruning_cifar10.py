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
# from CIFAR_pretrained_models_master.cifar_pretrainedmodels import resnet # original manuscript
from pytorch_resnet_cifar10_master import resnet 
# Import vgg11-cifar10, densenet161-cifar10 pretrained models
from PyTorch_CIFAR10_master.cifar10_models import vgg, densenet

from definition_wp import train_model, set_parameter_requires_grad, initialize_model
#from definition_wp import set_parameter_requires_grad
#from definition_wp import initialize_model
from datetime import datetime


def main():

	#print("PyTorch Version: ",torch.__version__)
	#print("Torchvision Version: ",torchvision.__version__)
	now = datetime.now()
	start = now.strftime("%D:%H:%M:%S")
	
	with open('./log.txt', 'a') as f:
		f.write('-----'*10+'\n')
		f.write(start+'\n')
		f.write('model name: cifar10-{}\n'.format(model_name))
		f.write('frac: {}/16\n'.format(frac1))
	
	frac = frac1/16
	num_epochs = 100
	input_size = 32
	feature_extract = False
	data_dir = "./datasets/cifar10"
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	a = False
	
	if model_name  == 'resnet20':
		#model = resnet.cifar_resnet20(pretrained='cifar10')
		path = '/mnt/ssd1/zhengyue/Models/pytorch_resnet_cifar10_master/save_resnet20/checkpoint.th'
		checkpoint = torch.load(path)
		model = torch.nn.DataParallel(resnet.__dict__['resnet20']())
		model.cuda()
		model.load_state_dict(checkpoint['state_dict'])
		a = True
		parameters_to_prune = (
			(model.module.conv1, 'weight'),
			(model.module.layer1[0].conv1, 'weight'),
			(model.module.layer1[0].conv2, 'weight'),
			(model.module.layer1[1].conv1, 'weight'),
			(model.module.layer1[1].conv2, 'weight'),
			(model.module.layer1[2].conv1, 'weight'),
			(model.module.layer1[2].conv2, 'weight'),
			(model.module.layer2[0].conv1, 'weight'),
			(model.module.layer2[0].conv2, 'weight'),
			(model.module.layer2[1].conv1, 'weight'),
			(model.module.layer2[1].conv2, 'weight'),
			(model.module.layer2[2].conv1, 'weight'),
			(model.module.layer2[2].conv2, 'weight'),
			(model.module.layer3[0].conv1, 'weight'),
			(model.module.layer3[0].conv2, 'weight'),
			(model.module.layer3[1].conv1, 'weight'),
			(model.module.layer3[1].conv2, 'weight'),
			(model.module.layer3[2].conv1, 'weight'),
			(model.module.layer3[2].conv2, 'weight'),)
		
	elif model_name == 'vgg11':
		model = vgg.vgg11_bn(pretrained=True)
		parameters_to_prune = (
			(model.features[0], 'weight'),
			(model.features[4], 'weight'),
			(model.features[8], 'weight'),
			(model.features[11], 'weight'),
			(model.features[15], 'weight'),
			(model.features[18], 'weight'),
			(model.features[22], 'weight'),
			(model.features[25], 'weight'),)
		
	elif model_name == 'densenet161':
		model = densenet.densenet161(pretrained=True,device=device)
		parameters_to_prune = (
			(model.features.conv0, 'weight'),
			(model.features.denseblock1.denselayer1.conv1, 'weight'),
			(model.features.denseblock1.denselayer1.conv2, 'weight'),
			(model.features.denseblock1.denselayer2.conv1, 'weight'),
			(model.features.denseblock1.denselayer2.conv2, 'weight'),
			(model.features.denseblock1.denselayer3.conv1, 'weight'),
			(model.features.denseblock1.denselayer3.conv2, 'weight'),
			(model.features.denseblock1.denselayer4.conv1, 'weight'),
			(model.features.denseblock1.denselayer4.conv2, 'weight'),
			(model.features.denseblock1.denselayer5.conv1, 'weight'),
			(model.features.denseblock1.denselayer5.conv2, 'weight'),
			(model.features.denseblock1.denselayer6.conv1, 'weight'),
			(model.features.denseblock1.denselayer6.conv2, 'weight'),
			(model.features.transition1.conv, 'weight'),
			(model.features.denseblock2.denselayer1.conv1, 'weight'),
			(model.features.denseblock2.denselayer1.conv2, 'weight'),
			(model.features.denseblock2.denselayer2.conv1, 'weight'),
			(model.features.denseblock2.denselayer2.conv2, 'weight'),
			(model.features.denseblock2.denselayer3.conv1, 'weight'),
			(model.features.denseblock2.denselayer3.conv2, 'weight'),
			(model.features.denseblock2.denselayer4.conv1, 'weight'),
			(model.features.denseblock2.denselayer4.conv2, 'weight'),
			(model.features.denseblock2.denselayer5.conv1, 'weight'),
			(model.features.denseblock2.denselayer5.conv2, 'weight'),
			(model.features.denseblock2.denselayer6.conv1, 'weight'),
			(model.features.denseblock2.denselayer6.conv2, 'weight'),
			(model.features.denseblock2.denselayer7.conv1, 'weight'),
			(model.features.denseblock2.denselayer7.conv2, 'weight'),
			(model.features.denseblock2.denselayer8.conv1, 'weight'),
			(model.features.denseblock2.denselayer8.conv2, 'weight'),
			(model.features.denseblock2.denselayer9.conv1, 'weight'),
			(model.features.denseblock2.denselayer9.conv2, 'weight'),
			(model.features.denseblock2.denselayer10.conv1, 'weight'),
			(model.features.denseblock2.denselayer10.conv2, 'weight'),
			(model.features.denseblock2.denselayer11.conv1, 'weight'),
			(model.features.denseblock2.denselayer11.conv2, 'weight'),
			(model.features.denseblock2.denselayer12.conv1, 'weight'),
			(model.features.denseblock2.denselayer12.conv2, 'weight'),
			(model.features.transition2.conv, 'weight'),
			(model.features.denseblock3.denselayer1.conv1, 'weight'),
			(model.features.denseblock3.denselayer1.conv2, 'weight'),
			(model.features.denseblock3.denselayer2.conv1, 'weight'),
			(model.features.denseblock3.denselayer2.conv2, 'weight'),
			(model.features.denseblock3.denselayer3.conv1, 'weight'),
			(model.features.denseblock3.denselayer3.conv2, 'weight'),
			(model.features.denseblock3.denselayer4.conv1, 'weight'),
			(model.features.denseblock3.denselayer4.conv2, 'weight'),
			(model.features.denseblock3.denselayer5.conv1, 'weight'),
			(model.features.denseblock3.denselayer5.conv2, 'weight'),
			(model.features.denseblock3.denselayer6.conv1, 'weight'),
			(model.features.denseblock3.denselayer6.conv2, 'weight'),
			(model.features.denseblock3.denselayer7.conv1, 'weight'),
			(model.features.denseblock3.denselayer7.conv2, 'weight'),
			(model.features.denseblock3.denselayer8.conv1, 'weight'),
			(model.features.denseblock3.denselayer8.conv2, 'weight'),
			(model.features.denseblock3.denselayer9.conv1, 'weight'),
			(model.features.denseblock3.denselayer9.conv2, 'weight'),
			(model.features.denseblock3.denselayer10.conv1, 'weight'),
			(model.features.denseblock3.denselayer10.conv2, 'weight'),
			(model.features.denseblock3.denselayer11.conv1, 'weight'),
			(model.features.denseblock3.denselayer11.conv2, 'weight'),
			(model.features.denseblock3.denselayer12.conv1, 'weight'),
			(model.features.denseblock3.denselayer12.conv2, 'weight'),
			(model.features.denseblock3.denselayer13.conv1, 'weight'),
			(model.features.denseblock3.denselayer13.conv2, 'weight'),
			(model.features.denseblock3.denselayer14.conv1, 'weight'),
			(model.features.denseblock3.denselayer14.conv2, 'weight'),
			(model.features.denseblock3.denselayer15.conv1, 'weight'),
			(model.features.denseblock3.denselayer15.conv2, 'weight'),
			(model.features.denseblock3.denselayer16.conv1, 'weight'),
			(model.features.denseblock3.denselayer16.conv2, 'weight'),
			(model.features.denseblock3.denselayer17.conv1, 'weight'),
			(model.features.denseblock3.denselayer17.conv2, 'weight'),
			(model.features.denseblock3.denselayer18.conv1, 'weight'),
			(model.features.denseblock3.denselayer18.conv2, 'weight'),
			(model.features.denseblock3.denselayer19.conv1, 'weight'),
			(model.features.denseblock3.denselayer19.conv2, 'weight'),
			(model.features.denseblock3.denselayer20.conv1, 'weight'),
			(model.features.denseblock3.denselayer20.conv2, 'weight'),
			(model.features.denseblock3.denselayer21.conv1, 'weight'),
			(model.features.denseblock3.denselayer21.conv2, 'weight'),
			(model.features.denseblock3.denselayer22.conv1, 'weight'),
			(model.features.denseblock3.denselayer22.conv2, 'weight'),
			(model.features.denseblock3.denselayer23.conv1, 'weight'),
			(model.features.denseblock3.denselayer23.conv2, 'weight'),
			(model.features.denseblock3.denselayer24.conv1, 'weight'),
			(model.features.denseblock3.denselayer24.conv2, 'weight'),
			(model.features.denseblock3.denselayer25.conv1, 'weight'),
			(model.features.denseblock3.denselayer25.conv2, 'weight'),
			(model.features.denseblock3.denselayer26.conv1, 'weight'),
			(model.features.denseblock3.denselayer26.conv2, 'weight'),
			(model.features.denseblock3.denselayer27.conv1, 'weight'),
			(model.features.denseblock3.denselayer27.conv2, 'weight'),
			(model.features.denseblock3.denselayer28.conv1, 'weight'),
			(model.features.denseblock3.denselayer28.conv2, 'weight'),
			(model.features.denseblock3.denselayer29.conv1, 'weight'),
			(model.features.denseblock3.denselayer29.conv2, 'weight'),
			(model.features.denseblock3.denselayer30.conv1, 'weight'),
			(model.features.denseblock3.denselayer30.conv2, 'weight'),
			(model.features.denseblock3.denselayer31.conv1, 'weight'),
			(model.features.denseblock3.denselayer31.conv2, 'weight'),
			(model.features.denseblock3.denselayer32.conv1, 'weight'),
			(model.features.denseblock3.denselayer32.conv2, 'weight'),
			(model.features.denseblock3.denselayer33.conv1, 'weight'),
			(model.features.denseblock3.denselayer33.conv2, 'weight'),
			(model.features.denseblock3.denselayer34.conv1, 'weight'),
			(model.features.denseblock3.denselayer34.conv2, 'weight'),
			(model.features.denseblock3.denselayer35.conv1, 'weight'),
			(model.features.denseblock3.denselayer35.conv2, 'weight'),
			(model.features.denseblock3.denselayer36.conv1, 'weight'),
			(model.features.denseblock3.denselayer36.conv2, 'weight'),
			(model.features.transition3.conv, 'weight'),
			(model.features.denseblock4.denselayer1.conv1, 'weight'),
			(model.features.denseblock4.denselayer1.conv2, 'weight'),
			(model.features.denseblock4.denselayer2.conv1, 'weight'),
			(model.features.denseblock4.denselayer2.conv2, 'weight'),
			(model.features.denseblock4.denselayer3.conv1, 'weight'),
			(model.features.denseblock4.denselayer3.conv2, 'weight'),
			(model.features.denseblock4.denselayer4.conv1, 'weight'),
			(model.features.denseblock4.denselayer4.conv2, 'weight'),
			(model.features.denseblock4.denselayer5.conv1, 'weight'),
			(model.features.denseblock4.denselayer5.conv2, 'weight'),
			(model.features.denseblock4.denselayer6.conv1, 'weight'),
			(model.features.denseblock4.denselayer6.conv2, 'weight'),
			(model.features.denseblock4.denselayer7.conv1, 'weight'),
			(model.features.denseblock4.denselayer7.conv2, 'weight'),
			(model.features.denseblock4.denselayer8.conv1, 'weight'),
			(model.features.denseblock4.denselayer8.conv2, 'weight'),
			(model.features.denseblock4.denselayer9.conv1, 'weight'),
			(model.features.denseblock4.denselayer9.conv2, 'weight'),
			(model.features.denseblock4.denselayer10.conv1, 'weight'),
			(model.features.denseblock4.denselayer10.conv2, 'weight'),
			(model.features.denseblock4.denselayer11.conv1, 'weight'),
			(model.features.denseblock4.denselayer11.conv2, 'weight'),
			(model.features.denseblock4.denselayer12.conv1, 'weight'),
			(model.features.denseblock4.denselayer12.conv2, 'weight'),
			(model.features.denseblock4.denselayer13.conv1, 'weight'),
			(model.features.denseblock4.denselayer13.conv2, 'weight'),
			(model.features.denseblock4.denselayer14.conv1, 'weight'),
			(model.features.denseblock4.denselayer14.conv2, 'weight'),
			(model.features.denseblock4.denselayer15.conv1, 'weight'),
			(model.features.denseblock4.denselayer15.conv2, 'weight'),
			(model.features.denseblock4.denselayer16.conv1, 'weight'),
			(model.features.denseblock4.denselayer16.conv2, 'weight'),
			(model.features.denseblock4.denselayer17.conv1, 'weight'),
			(model.features.denseblock4.denselayer17.conv2, 'weight'),
			(model.features.denseblock4.denselayer18.conv1, 'weight'),
			(model.features.denseblock4.denselayer18.conv2, 'weight'),
			(model.features.denseblock4.denselayer19.conv1, 'weight'),
			(model.features.denseblock4.denselayer19.conv2, 'weight'),
			(model.features.denseblock4.denselayer20.conv1, 'weight'),
			(model.features.denseblock4.denselayer20.conv2, 'weight'),
			(model.features.denseblock4.denselayer21.conv1, 'weight'),
			(model.features.denseblock4.denselayer21.conv2, 'weight'),
			(model.features.denseblock4.denselayer22.conv1, 'weight'),
			(model.features.denseblock4.denselayer22.conv2, 'weight'),
			(model.features.denseblock4.denselayer23.conv1, 'weight'),
			(model.features.denseblock4.denselayer23.conv2, 'weight'),
			(model.features.denseblock4.denselayer24.conv1, 'weight'),
			(model.features.denseblock4.denselayer24.conv2, 'weight'))	
	else:
		print('Unknown model name input!')
		
	batch_size = 128	
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

	optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
	

	criterion = nn.CrossEntropyLoss()
	
	stat_epoch = 0
# Train and evaluate
	model_ft, hist, loss, epoch = train_model(model, dataloaders_dict, criterion, optimizer_ft, stat_epoch, num_epochs=num_epochs, is_inception=(model_name=="inception"))
	print("frac: "+str(frac1)+"\n")
	saved_PATH = "./saved_models/cifar10_"+model_name+"_fp/"
	if a:
		print('This resnet20-cifar10 model_ft cannot be saved!')
	else:
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
	model_name_list = ['resnet20','resnet20','resnet20']
	frac1_list = [9,10,11]
	for i in range(len(model_name_list)):
		model_name = model_name_list[i]
		frac1 = frac1_list[i]
		print('Start Filter Pruning cifar10-'+ model_name+'......')
		main()
