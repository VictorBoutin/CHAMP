import torch
from torch.autograd import Variable
from PIL import Image
from random import shuffle
from os import listdir
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import conv2d
from CHAMP.DataTools import Normalize, Decorrelate, ContrastNormalized, zero_one_norm, MakePositive
from CHAMP.DataTools import Decorrelate2





def LoadMNIST(batch_size, data_path='../../DataBase/MNISTtorch', decorrelate=True):
    transform = transforms.Compose([
                 transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root=data_path,train=True,transform=transform,download=False)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=2)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False,transform=transform,download=False)
    #test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=10000, shuffle=False,num_workers=2)
    if decorrelate == True:
        train_data_loader = Decorrelate(train_data_loader)
        test_data_loader = Decorrelate(test_data_loader)

    #return (img_train,label_train), (img_test,label_test)
    return (train_data_loader), (test_data_loader)

def LoadMNIST2(batch_size, data_path='../../DataBase/MNISTtorch', decorrelate=True):
    transform = transforms.Compose([
                 transforms.ToTensor()])
                 #lambda x : Decorrelate2(x)


    train_set = torchvision.datasets.MNIST(root=data_path,train=True,transform=transform,download=False)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=2)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False,transform=transform,download=False)
    #test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=10000, shuffle=False,num_workers=2)

    data_train = (Variable(torch.FloatTensor(60000//batch_size,batch_size,1,28,28)),torch.LongTensor(60000//batch_size,batch_size))
    data_test = (Variable(torch.FloatTensor(1,10000,1,28,28)),torch.LongTensor(1,10000))
    #print(data_train[0].size())

    for idx, each_batch in enumerate(train_data_loader) :
        data_train[0][idx,:,:,:,:]=Variable(each_batch[0])
        data_train[1][idx,:]=each_batch[1]

    for idx,each_batch in enumerate(test_data_loader):
        data_test[0][idx,:,:,:,:]=Variable(each_batch[0])
        data_test[1][idx,:]=each_batch[1]

    if decorrelate == True:
        data_train = Decor(data_train)
        data_test = Decor(data_test)

    return data_train, data_test

def Decor(dataset):
    filt_decor = Variable(torch.Tensor(1,1,3,3))
    filt_decor[0,0,:,:] = torch.FloatTensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
    data_transformed = (Variable(torch.FloatTensor(dataset[0].size())),torch.LongTensor(dataset[1].size()))
    for i, each_set in enumerate(dataset[0]):
        images = each_set
        label = dataset[1][i,:]

        data_transformed[0][i,:,:,:,:] = conv2d(images, filt_decor,padding=1)
        data_transformed[1][i,:] = label
    return data_transformed


def GenerateMask(dico, sigma=0.8, style='Gaussian'):
    dico_size = tuple(dico.size())
    R = dico_size[2]//2
    grid = torch.arange(-1*R,R+1)
    X_grid = grid.unsqueeze(1).expand((dico_size[2],dico_size[3]))
    Y_grid  = torch.t(X_grid)
    radius = torch.sqrt(X_grid**2 + Y_grid**2)
    if style == 'Gaussian':
        mask = torch.exp(-0.5*radius**2/R**2/sigma**2)
    elif style == 'Binary':
        mask = radius < sigma
        mask = mask.type(torch.FloatTensor)
    elif style == 'GaussianBinary':
        mask = torch.exp(-0.5*radius**2/R**2/sigma**2)
        binary_mask = (radius < R+1).type(torch.FloatTensor)
        mask = mask*binary_mask
    elif style == 'Polynomial':
        mask = ((radius-R)/R)*((radius-R)/R)
    elif style == 'Square':
        mask = radius<2
        mask = mask.type(torch.FloatTensor)
        mask[radius>=2]=0.9
    elif style == 'GaussianBinaryMagic':
        mask = torch.exp(-0.5*radius**2/(R+3)**2/sigma**2)
        binary_mask = (radius < R+1).type(torch.FloatTensor)
        mask = mask*binary_mask
    elif style == 'GaussianBinarySuperMagic':
        grid = 0.5*torch.arange(-1*R,R+1)
        X_grid = grid.unsqueeze(1).expand((dico_size[2],dico_size[3]))
        Y_grid  = torch.t(X_grid)
        radius = torch.sqrt(X_grid**2 + Y_grid**2)
        mask = torch.exp(-0.5*radius**2/(R+3)**2/sigma**2)
        binary_mask = (radius < R+1).type(torch.FloatTensor)
        mask = mask*binary_mask
    mask = mask.unsqueeze(0).unsqueeze(1).expand_as(dico)

    return mask


def LoadFaceDB(path,size = (68,68), nb_batch=1, to_shuffle=True, Decorrelated=True, Normalized=False) :
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
    if  Decorrelated == True :
        to_output = ContrastNormalized(to_output)
    #if Normalized == 'ZeroToOne' :
    #    to_output = zero_one_norm(to_output)
    #if Normalized == 'Positive' :
    #    to_output = MakePositive(to_output)
    return to_output
