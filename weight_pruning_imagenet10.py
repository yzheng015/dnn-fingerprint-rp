import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
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
#from CIFAR_pretrained_models_master.cifar_pretrainedmodels import resnet
from definition_wp_imagenet10 import train_model,set_parameter_requires_grad,initialize_model


from datetime import datetime

def main():

	#print("PyTorch Version: ",torch.__version__)
	#print("Torchvision Version: ",torchvision.__version__)
  
	frac = frac1/10
	
	now = datetime.now()
	start = now.strftime("%D:%H:%M:%S")	
	
	with open('./log.txt', 'a') as f:
		f.write('-----'*10+'\n')
		f.write(start+'\n')
		f.write('model name: wp_Imagenet10-{}\n'.format(model_name))
		f.write('frac: {}/10\n'.format(frac1))

	model_path = '/mnt/ssd1/zhengyue/Models'
	data_dir = "./datasets/imagenet10_data"


	batch_size = 64
	num_epochs = 100
	input_size = 224

	feature_extract = False
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Change GPU, 0 or 1

	# Initialize the model for this run
	#PATH="./imagenet10_vgg11_saved_model/vgg11_2_model_0_99.pth"

	
	if model_name == 'resnet18':
		imagenet10_resnet18 = model_path + '/Target_models/imagenet10_resnet18_model_0_29.pth'
		model = torch.load(imagenet10_resnet18)
		parameters_to_prune = (
			(model.conv1, 'weight'),
			(model.layer1[0].conv1, 'weight'),
			(model.layer1[0].conv2, 'weight'),
			(model.layer1[1].conv1, 'weight'),
			(model.layer1[1].conv2, 'weight'),
			(model.layer2[0].conv1, 'weight'),
			(model.layer2[0].conv2, 'weight'),
			(model.layer2[1].conv1, 'weight'),
			(model.layer2[1].conv2, 'weight'),
			(model.layer3[0].conv1, 'weight'),
			(model.layer3[0].conv2, 'weight'),
			(model.layer3[1].conv1, 'weight'),
			(model.layer3[1].conv2, 'weight'),
			(model.layer4[0].conv1, 'weight'),
			(model.layer4[0].conv2, 'weight'),
			(model.layer4[1].conv1, 'weight'),
			(model.layer4[1].conv2, 'weight'),
			(model.fc, 'weight'),
			(model.fc, 'bias'),
		)
	elif model_name == 'vgg11':
		imagenet10_vgg11 = model_path + '/Target_models/imagenet10_vgg11_model_0_49.pth'
		model = torch.load(imagenet10_vgg11)
		parameters_to_prune = (
			(model.features[0], 'weight'),
			(model.features[3], 'weight'),
			(model.features[6], 'weight'),
			(model.features[8], 'weight'),
			(model.features[11], 'weight'),
			(model.features[13], 'weight'),
			(model.features[16], 'weight'),
			(model.features[18], 'weight'),
			(model.features[0], 'bias'),
			(model.features[3], 'bias'),
			(model.features[6], 'bias'),
			(model.features[8], 'bias'),
			(model.features[11], 'bias'),
			(model.features[13], 'bias'),
			(model.features[16], 'bias'),
			(model.features[18], 'bias'),
			(model.classifier[0], 'weight'),
			(model.classifier[3], 'weight'),
			(model.classifier[6], 'weight'),
			(model.classifier[0], 'bias'),
			(model.classifier[3], 'bias'),
			(model.classifier[6], 'bias'),)
	elif model_name == 'densenet161':
		imagenet10_densenet161 = model_path + '/Target_models/imagenet10_densenet161_model_0_99.pth'
		model = torch.load(imagenet10_densenet161)	
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
			(model.features.denseblock4.denselayer24.conv2, 'weight'),
			(model.classifier, 'weight'),
			(model.classifier, 'bias'))
	
	model = model.to(device)

	prune.global_unstructured(
	   parameters_to_prune,
	   pruning_method=prune.L1Unstructured,
	   amount=frac,
	)

	for turple in parameters_to_prune:
		prune.remove(turple[0], turple[1])

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
	
	#print("amount: 0.9 \n")

	optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
	

	criterion = nn.CrossEntropyLoss()
	
	stat_epoch = 0
# Train and evaluate
	model_ft, hist, loss, epoch = train_model(model, dataloaders_dict, criterion, optimizer_ft, stat_epoch, num_epochs=num_epochs, is_inception=(model_name=="inception"))
	print("amount: "+str(frac)+"\n")
	saved_PATH = "./saved_models/imagenet10_"+model_name+"_wp/"
	torch.save(model_ft, os.path.join(saved_PATH,"wp"+str(frac1)+"_model_"+str(stat_epoch)+"_"+str(epoch)+".pth"))
	torch.save({
			'epoch': epoch,
			'model_state_dict': model_ft.state_dict(),
			'optimizer_state_dict': optimizer_ft.state_dict(),
			'loss': loss,
			}, os.path.join(saved_PATH, "wp"+str(frac1)+"_checkpoint_"+str(stat_epoch)+"_"+str(epoch)+".pth"))

if __name__ == "__main__":
	
	#model_name = input('Model to be tested: ')
	#frac1 = int(input('frac 1-10: '))
	model_name_list = ['vgg11', 'densenet161']
	frac1_list = [9, 9]
	for i in range(len(model_name_list)):
		model_name = model_name_list[i]
		frac1 = frac1_list[i]
		print('Start Weight Pruning imagenet10-'+ model_name+'......')
		main()
