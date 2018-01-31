import torch
from PIL import Image
from random import shuffle
from os import listdir
import numpy as np
import torchvision
import torchvision.transforms as transforms
from CHAMP.DataTools import Normalize, ContrastNormalized


def LoadData(name,data_path,decorrelate=True,avg_size=(5,5),Grayscale=True,resize=None,GPU=False,dowload=False):
    Composition = list()
    if Grayscale == True :
        Composition.append(transforms.Grayscale())
    if resize is not None :
        Composition.append(transforms.Resize(resize))
    Composition.append(transforms.ToTensor())
    transform = transforms.Compose(Composition)
    if name=='MNIST':
        train_set = torchvision.datasets.MNIST(root=data_path,train=True,transform=transform,download=download)
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=60000,shuffle=True,num_workers=2)
        test_set = torchvision.datasets.MNIST(root=data_path, train=False,transform=transform,download=download)
        test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=10000, shuffle=False,num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'CIFAR':
        train_set = torchvision.datasets.CIFAR10(root=data_path,train=True,transform=transform,download=download)
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=50000,shuffle=True,num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root=data_path,train=False, transform=transform,download=download)
        test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=10000, shuffle=False,num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'STL10':
        train_set = torchvision.datasets.STL10(root=data_path,split='train',transform=transform,download=download)
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=50000,shuffle=True,num_workers=2)
        test_set = torchvision.datasets.STL10(root=data_path,split='test', transform=transform,download=download)
        test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=10000, shuffle=False,num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'Face':
        data_training = LoadFaceDB(data_path,size = (92,92), nb_batch=1, to_shuffle=True)
        data_testing = (data_training[0].clone(),data_training[1].clone())
    if GPU == True :
        data_training = (data_training[0].cuda(),data_training[1].cuda())
        data_testing = (data_testing[0].cuda(),data_testing[1].cuda())
    if decorrelate == True:
        data_training = ContrastNormalized(data_training,avg_size=avg_size,GPU=GPU)
        data_testing = ContrastNormalized(data_testing,avg_size=avg_size,GPU=GPU)
    return (data_training,data_testing)


def LoadFaceDB(path,size = (68,68), nb_batch=1, to_shuffle=True) :
    file_list = list()
    batch_size = 400//nb_batch
    if size is None :
        size = (112,92)
    tensor_image = torch.FloatTensor(nb_batch, batch_size, 1, size[0], size[1])
    tensor_label = torch.LongTensor(nb_batch, batch_size)
    label=0
    for each_dir in listdir(path):
        if each_dir !='.DS_Store':
            for each_file in listdir(path+str(each_dir)):
                tot_dir = str(path)+str(each_dir)+'/'+str(each_file)
                file_list.append((tot_dir,label))
            label+=1
    if to_shuffle == True :
        shuffle(file_list)
    idx = 0
    for i in range(nb_batch):
        for j in range(batch_size):
            image = Image.open(file_list[idx][0])
            image = image.resize(size,Image.ANTIALIAS)
            tensor_image[i,j,0,:,:] = torch.FloatTensor(np.array(image).astype(float))
            tensor_label[i,j] = file_list[idx][1]
            idx+=1
    to_output = (tensor_image.float(),tensor_label)
    return to_output
