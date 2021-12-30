import numpy as np
from scipy.fftpack import dct, idct
import torch
import random
import csv

#### Retrieve models from txt file: allmodels.txt
def retrieve(file, names):

	model_list=[]

	with open(file,'r') as fl:
		fc = fl.readlines()
		for line in fc:
			current_model = line[:-1]
			model_list.append(current_model)

	cnt = 0
	idx = [0]
	for item in model_list:
		cnt += 1
		if item == '':
			idx = idx + [cnt]

	models = {}
	for i in range(len(idx)):
		if i < len(idx)-1:
			models[names[i]] = model_list[idx[i]:idx[i+1]-1]
		else:
			models[names[i]] = model_list[idx[i]:]

	return models

#### Retrieve cross application models from txt file, cross_application_models.txt
def CAM_retrieve(file, names):

	model_list=[]

	with open(file,'r') as fl:
		fc = fl.readlines()
		for line in fc:
			current_model = line[:-1]
			model_list.append(current_model)

	models = {}
	for i in range(len(names)):
		models[names[i]] = model_list[i]

	return models


### Owner feature (U) generation 
def owner_f(seed, nl, out_rp, option):
	''' The way to generate owner features
		seed: owner's specific id, could be biometrics or national ID or etc.
	'''
	np.random.seed(seed)

	if option == 'gaussian':
		out = np.random.normal(0, 1, size=(nl, out_rp))
	elif option == 'bernoulli':
		rand = np.random.randint(0,2, size=(nl, out_rp))
		out = np.zeros((nl, out_rp))
		for i in range(nl):
			for j in range(out_rp):
				if rand[i][j] == 0:
					out[i][j] = 1/np.sqrt(out_rp)
				else:
					out[i][j] = -1/np.sqrt(out_rp)
	else:
		print('no such options!')

	return out


### adjusted cosine similarity generation. 
def acs(x, y):
	'''Adjust Cosine Similarity correlation
		dx, dy: dispersion of x and y from their corresponding means
	'''
#	 if option == 'rppolarity':
#		 dx = [item-np.mean(x) for item in x]
#		 dy = [item-np.mean(y) for item in y]
#	 elif option == 'wpolarity':
#		 dx = [1 if (item-np.mean(x))>0 else -1 for item in x]
#		 dy = [1 if (item-np.mean(y))>0 else -1 for item in y]
#	 else:
#		 print('Option Wrong!')

	dx = [item-np.mean(x) for item in x]
	dy = [item-np.mean(y) for item in y]

	l = len(dx)
	a = [dx[i]*dy[i] for i in range(l)]
	top = sum(a)
	b = [item**2 for item in dx]
	c = [item**2 for item in dy]
	bottom = np.sqrt(sum(b)*sum(c))
	r = top/bottom
	return r



'''
### Get the weights parameter for cifar10 resent 20 models
def get_weights(name, param, conv, bias):
	if name.split('.')[-1] == 'weight':
		if len(param.shape) == 1:
			bias = torch.cat((bias, param),0)
		else:
			param_mean = torch.mean(param, axis=0)
			conv_flat = param_mean.flatten()
			#conv_flat = param.flatten()
			conv = torch.cat((conv, conv_flat), 0)
		return conv, bias
	else:
		return None


def get_all_weights(model, target=True):
	mD = torch.tensor([]) # multiple Dimension
	sD = torch.tensor([]) # single Dimension
	if target:
		items = list(model.items())
		for item in items:
			out = get_weights(item[0], item[1], mD, sD)
			if out != None:
				mD = out[0]
				sD = out[1]
	else:
		for name, param in model.named_parameters():
			out = get_weights(name, param, mD, sD)
			if out != None:
				mD = out[0]
				sD = out[1]
	return mD, sD
'''


def get_param(model, l, td, flag=True):
	'''
	model: model path
	l: number of layers to extract the weights
	step: this is originally used to faster the DCT computation.
	td: the dimension of DNN model features (weights) from each layer
	flag: the model dataformat is different, need to read load data using different methods show below
	'''
	out = []
	cnt = 0
	if flag:
		items = list(model.items())
		for item in items:
			if item[0].split('.')[-1] == 'weight' and len(item[1].shape) > 1 and cnt < l:
				cnt += 1
				#param = item[1].detach().numpy().flatten()
				param = item[1].cpu().detach().numpy().flatten()
#				 param_select = select(param, step, td)
				param_select = param[:td]
				out = out + list(param_select)
	else:
		for name, param in model.named_parameters():
			if name.split('.')[-1] == 'weight' and len(param.shape) > 1 and cnt < l:
				cnt += 1
				param = param.detach().numpy().flatten()
#				 param_select = select(param, step, td)
				param_select = param[:td]
				out = out + list(param_select)
	return out


def get_param_c10r20_rq(model, l, td):
    '''
    model: model path
    l: number of layers to extract the weights
    step: this is originally used to faster the DCT computation.
    td: the dimension of DNN model features (weights) from each layer
    flag: the model dataformat is different, need to read load data using different methods show below
    '''
    out = []
    cnt = 0

    items = list(model.items())
    state_dict = items[1][1]
    for item in state_dict:
        data = state_dict[item]
        if item.split('.')[-1] == 'weight' and len(data.shape) > 1 and cnt < l:
            cnt += 1
            param = data.detach().cpu().numpy().flatten()
            param_select = param[:td]
            out = out + list(param_select)

    return out



def cal_mu_abs(model, l, nl, flag):
    '''calculate mu_abs of a model
       model: state_dict of a model is ok
    '''
    weights = get_param(model, l, nl//l, flag)
    mu_abs = np.mean([abs(weight) for weight in weights])
    print('The mu_abs of the selected parameter is:{}'.format(mu_abs))




def similarity_sca(target, suspect_models, usr1, usr2, flag1, flag2, nl,l, a, file_name, option = 'rppolarity'):
	'''using side channel attack to obtian the suspect model's parameters
	target: target models
	suspect_models: positive or negative suspect models
	usr: owner feature
	flag1, flag2: adjust the model data format for loading
	nl: the DNN model feaures' dimension
	l: the number of layers to extract the DNN weights/features. 
	a: |weights_recovered-weights_original| < a
	'''
	similarity = []
#	 step = 3000
	td = nl//l # the dimension of output for each layer
	target_model = torch.load(target, map_location='cpu') # type: collections.OrderedDict
	
	
	out = get_param(target_model, l, td, flag1)
	print(len(out))

	
	if option == 'wpolarity': # use weights polarity before random projection
		out_mean = np.round(np.mean(out), 3)
		if np.abs(out_mean) <= 0.005:
			out_mean = 0
		elif np.abs(out_mean - 0.01) < 0.005:
			out_mean = 0.01
		elif np.abs(out_mean + 0.01) < 0.005:
			out_mean = -0.01
		else:
			print('Absolute(mean)> 0.015!')
		out_polarity = [1 if element>out_mean else -1 for element in out]
		out = out_polarity
	 
	template = np.matmul(out, usr1) # numpy.ndarray
#	 t1=datetime.now()
#	 print('fingerprint extraction time ====>', (t1-t0).microseconds)

	for model in suspect_models:
		pm = torch.load(model,map_location='cpu') # pm: positive model
		out_pm = get_param(pm, l, td, flag2)
		
		### In real case, the designer can obtain out_pm using side channel analysis
		### Therefore, the out_pm used for calculation cannot be the exactly the same with the real values.
		out_pm_sca = [out_pm[i] + a*random.choice((-1, 1)) for i in range(len(out_pm))] # fixed
#		 out_pm_sca = [out_pm[i] + random.uniform(-a,a) for i in range(len(out_pm))] # random  
		
		if option == 'wpolarity':
			out_pm_mean = np.round(np.mean(out_pm_sca), 3)
			print('mean_original: {}, mean_sca: {}'.format(out_mean, out_pm_mean))
			out_pm_polarity = [1 if element>out_mean else -1 for element in out_pm_sca]
			out_pm = out_pm_polarity
		else:
			out_pm = out_pm_sca
		
		
		
		try:
			t_pm = np.matmul(out_pm, usr2)
			out =acs(template, t_pm)
		except:
			print('weights length not match!')
			print(model)
			x = len(out_pm)
			t_pm = np.matmul(out_pm, usr2[:x,:])
			out =acs(template[:x], t_pm)
			
		similarity.append(np.round(out, 3))
#		 print('Similarity: {}'.format(out))
	
	with open(file_name, 'a', newline='') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		wr.writerow(similarity)
	
	print('Correlation: {}'.format(similarity))
	
	
	
### fingerprint similarity calculation for cross-applicaiton scenarios, 
### also used to compare two single models, 
def CAM_similarity(source, ft_model, usr1, usr2, flag1, flag2, nl,l,a, file_name=False):
	'''
	source: source model
	ft_model: ft_model is obtained by fine tuning the source model, same dataset
	'''
	similarity = []
	td = nl//l # the dimension of output for each layer
	if isinstance(source, str):   
		source_model = torch.load(source, map_location='cpu') # type: collections.OrderedDict
	else: 
		## this is for imagenet pretrained models downloaded directly from torchvision
		source_model = source
	
	out = get_param(source_model, l, td, flag1)
	if len(out) == 0:
		print('the target model is the cifar10-resnet20 model used in the revised manuscript!')
		out = get_param_c10r20_rq(source_model, l, td)

	
	template = np.matmul(out, usr1) # numpy.ndarray
	
	if isinstance(ft_model, str):   
		pm = torch.load(ft_model, map_location='cpu') # type: collections.OrderedDict
	else: 
		## this is for imagenet pretrained models downloaded directly from torchvision
		pm = ft_model   

	#pm = torch.load(ft_model,map_location='cpu') # pm: positive model
	out_pm = get_param(pm, l, td, flag2)
	if len(out_pm) == 0:
		print('the suspect model is the cifar10-resnet20 model used in the revised manuscript!')
		out_pm = get_param_c10r20_rq(pm, l, td)
	
	out_pm_sca = [out_pm[i] + a*random.choice((-1, 1)) for i in range(len(out_pm))] # fixed
#	 out_pm_sca = [out_pm[i]+random.uniform(-a,a) for i in range(len(out_pm))] # random
	
	out_pm = out_pm_sca

	t_pm = np.matmul(out_pm, usr2)
	out =acs(template, t_pm)
	similarity.append(np.round(out, 3))
	
	if file_name == False:
		pass
	else:
		with open(file_name, 'a', newline='') as f:
			wr = csv.writer(f, quoting=csv.QUOTE_ALL)
			wr.writerow(similarity)
	
	print('Correlation: {}'.format(similarity))







