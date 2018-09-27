import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
import pickle
import cv2
import math
from CHAMP.LowLevel import conv, Normalize, padTensor, RotateDico90
import numpy as np


def ContrastNormalized(data, avg_size=(5, 5), GPU=False):
    #assert avg_size[0]%2 == 1, 'deccorlation filters size should be odd'
    img_size = data[0].size()
    if GPU == True:
        output_tensor = torch.cuda.FloatTensor(img_size)
        to_conv = (torch.ones(avg_size)*1 /
                   (avg_size[0]*avg_size[0])).view(1, 1, avg_size[0], avg_size[1]).cuda()
    else:
        output_tensor = torch.FloatTensor(img_size)
        to_conv = (torch.ones(avg_size)*1 /
                   (avg_size[0]*avg_size[0])).view(1, 1, avg_size[0], avg_size[1])
    for idx_batch, each_batch in enumerate(data[0]):
        padded_tensor = padTensor(each_batch, avg_size[0]//2, mode='reflect')
        convol = conv(padded_tensor, to_conv)
        output_tensor[idx_batch, :, :, :, :] = each_batch - convol
    return output_tensor, data[1]

def LocalContrastNormalization(training_set,sigma=0.5,filter_size=(9,9)):
    data = training_set[0].numpy()
    output = np.zeros_like(data)
    gaussian_window = GenerateMask((1,data.shape[2],filter_size[0],filter_size[1]),sigma=sigma).contiguous()
    somme = torch.sum(gaussian_window.view(1*data.shape[2],filter_size[0]*filter_size[1]),dim=1).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(gaussian_window)
    gaussian_window /= somme
    for i in range(data.shape[0]):
        data_padded = padTensor(training_set[0][i],filter_size[0]//2,mode='reflect')
        te = conv(data_padded, gaussian_window).numpy()
        output[i] = data[i,:,:,:,:]-te
        data_padded = padTensor(torch.FloatTensor(output[i]),filter_size[0]//2,mode='reflect')
        sig = torch.pow(conv(torch.pow(data_padded,2),gaussian_window),1/2)
        mean_sig = torch.mean(sig.view(-1,sig.size()[1]*sig.size()[2]*sig.size()[3]),dim=1)
        mean_sig = mean_sig.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(sig)
        normalized_coeff = torch.max(mean_sig,sig).numpy()
        output[i]/=normalized_coeff
    return (torch.FloatTensor(output),training_set[1]),te,normalized_coeff,gaussian_window

def oldLocalContrastNormalization(training_set,sigma=0.5,filter_size=(9,9)):
    data = training_set[0].numpy()
    output = np.zeros_like(data)
    gaussian_window = GenerateMask((1,data.shape[2],filter_size[0],filter_size[1]),sigma=sigma).contiguous()
    #somme = torch.sum(gaussian_window.view(1*data.shape[2],filter_size[0]*filter_size[1]),dim=1).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(gaussian_window)
    gaussian_window /= torch.sum(gaussian_window.view(-1,data.shape[2]*filter_size[0]*filter_size[1]),dim=1)
    for i in range(data.shape[0]):
        data_padded = padTensor(training_set[0][i],filter_size[0]//2,mode='reflect')
        te = conv(data_padded,gaussian_window)
        output[i] = data[i,:,:,:,:]-te
        data_padded = padTensor(torch.FloatTensor(output[i]),filter_size[0]//2,mode='reflect')
        sig = torch.pow(conv(torch.pow(data_padded,2),gaussian_window),1/2)
        mean_sig = torch.mean(sig.view(-1,sig.size()[1]*sig.size()[2]*sig.size()[3]),dim=1)
        mean_sig = mean_sig.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(sig)
        normalized_coeff = torch.max(mean_sig,sig)
        output[i]/=normalized_coeff
    return (torch.FloatTensor(output),training_set[1]),te,normalized_coeff,gaussian_window


def GenerateGabor(nb_dico, dico_size, sigma=1, lambd=5, gamma=0.5, psi=0, GPU=False):
    dico_size = tuple(dico_size)
    params = {'ksize': dico_size, 'sigma': sigma, 'lambd': lambd,
              'gamma': 0.5, 'psi': psi, 'ktype': cv2.CV_32F}
    if GPU == True:
        dico_gabor = torch.cuda.FloatTensor(nb_dico, 1, dico_size[0], dico_size[1])
        for i in range(nb_dico):
            dico_gabor[i, 0, :, :] = torch.cuda.FloatTensor(
                cv2.getGaborKernel(theta=i*math.pi/nb_dico, **params))
    else:
        dico_gabor = torch.FloatTensor(nb_dico, 1, dico_size[0], dico_size[1])
        for i in range(nb_dico):
            dico_gabor[i, 0, :, :] = torch.FloatTensor(
                cv2.getGaborKernel(theta=i*math.pi/nb_dico, **params))
    dico_gabor = Normalize(dico_gabor)
    return dico_gabor

# def Rebuilt(image,code,dico):
#    all_indice = torch.transpose(code._indices(),0,1)
#    output = torch.zeros(image.size())
#    padding = dico.size()[2]-1
#    for idx, (i,c) in enumerate(zip(all_indice,code._values())):
#        output[i[0],:,i[2]:i[2]+padding+1,i[3]:i[3]+padding+1].add_(c*dico[i[1],:,:,:])
#    return output


def Rebuilt(code, dico_in, idx=None, groups=1, stride=1):
    dico = dico_in.clone()
    if idx is not None:
        dico[idx, :, :, :] = 0
    if groups == 1:
        dico = dico.permute(1, 0, 2, 3)

    dico_rotated = RotateDico90(dico)
    padding = dico.size()[-1]-1
    output = conv(code, dico_rotated, padding=padding, groups=groups, stride=stride)
    return output

def GenerateMask(full_size, sigma=0.8, style='Gaussian',start_R=10):
    dico_size = (full_size[-2],full_size[-1])
    R = dico_size[-1]//2
    if dico_size[-1] % 2 == 1:
        grid = torch.arange(-1*R, R+1)
    else:
        grid = torch.cat((torch.arange(-1*R, 0, 1), torch.arange(1, R+1, 1)), 0)
    # print(grid)
    X_grid = grid.unsqueeze(1).expand((dico_size[-2], dico_size[-1]))
    Y_grid = torch.t(X_grid)
    radius = torch.sqrt((X_grid**2 + Y_grid**2).float())
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
        mask = radius < 2
        mask = mask.type(torch.FloatTensor)
        mask[radius >= 2] = 0.9
    elif style == 'GaussianBinaryMagic':
        mask = torch.exp(-0.5*radius**2/(R+3)**2/sigma**2)
        binary_mask = (radius < R+1).type(torch.FloatTensor)
        mask = mask*binary_mask
    elif style == 'GaussianBinarySuperMagic':
        grid = 0.5*torch.arange(-1*R, R+1)
        X_grid = grid.unsqueeze(1).expand((dico_size[-2], dico_size[-1]))
        Y_grid = torch.t(X_grid)
        radius = torch.sqrt(X_grid**2 + Y_grid**2)
        mask = torch.exp(-0.5*radius**2/(R+3)**2/sigma**2)
        binary_mask = (radius < R+1).type(torch.FloatTensor)
        mask = mask*binary_mask
    elif style == 'Custom':
        grid_or = torch.zeros(dico_size[-1])
        grid = torch.arange(-1*start_R, start_R+1)
        grid_or[0:start_R] = grid[0:start_R]
        grid_or[-start_R:] = grid[-1*start_R:]
        X_grid = grid_or.unsqueeze(1).expand((dico_size[-2], dico_size[-1]))
        Y_grid = torch.t(X_grid)
        radius = torch.sqrt(X_grid**2 + Y_grid**2)
        mask = torch.exp(-0.5*radius**2/R**2/sigma**2)
    else:
        print('style unknown')

    mask = mask.unsqueeze(0).unsqueeze(1).expand(full_size)

    return mask


def FilterInputData(data, sigma=0.8, style='Gaussian', start_R=10):
    Filter = GenerateMask(data[0].size(), sigma=sigma, style=style, start_R=start_R)
    data_filtered = torch.mul(data[0], Filter)
    return (data_filtered, data[1])


def ChangeBatchSize(data, batch_size):
    nb_image = data[0].size()[1]*data[0].size()[0]
    image_size = data[0].size()[2:]
    image = data[0].contiguous().view(nb_image//batch_size, batch_size,
                                      image_size[0], image_size[1], image_size[2])
    label = data[1].contiguous().view(nb_image//batch_size, -1)
    return (image, label)
###


def SaveNetwork(Network, saving_path):
    with open(saving_path, 'wb') as file:
        pickle.dump(Network, file, pickle.HIGHEST_PROTOCOL)
    print('file saved')


def LoadNetwork(loading_path):
    with open(loading_path, 'rb') as file:
        Net = pickle.load(file)
    return Net
