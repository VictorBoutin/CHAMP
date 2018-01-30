import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
import pickle
import cv2
import math
from CHAMP.LowLevel import conv, Normalize, padTensor



def ContrastNormalized(data,avg_size=(5,5),GPU=False):
    assert avg_size[0]%2 == 1, 'deccorlation filters size should be odd'
    img_size = data[0].size()
    if GPU == True :
        output_tensor = torch.cuda.FloatTensor(img_size)
        to_conv = torch.ones(avg_size)*1/(avg_size[0]*avg_size[0])).view(1,1,avg_size[0],avg_size[1].cuda()
    else :
        output_tensor = torch.FloatTensor(img_size)
        to_conv = (torch.ones(avg_size)*1/(avg_size[0]*avg_size[0])).view(1,1,avg_size[0],avg_size[1])
    for idx_batch, each_batch in enumerate(data[0]):
        padded_tensor = padTensor(each_batch,avg_size[0]//2,mode='reflect')
        convol = conv(padded_tensor,to_conv)
        output_tensor[idx_batch,:,:,:,:] = each_batch - convol
    return output_tensor, data[1]

def GenerateGabor(nb_dico, dico_size,sigma=1,lambd=5,gamma=0.5,psi=0):
    dico_size = tuple(dico_size)
    params = {'ksize':dico_size, 'sigma':sigma,'lambd':lambd,
                  'gamma':0.5, 'psi':psi, 'ktype':cv2.CV_32F}

    dico_gabor = torch.Tensor(nb_dico,1,dico_size[0],dico_size[1])
    for i in range(nb_dico):
        dico_gabor[i,0,:,:]=torch.FloatTensor(cv2.getGaborKernel(theta=i*math.pi/nb_dico,**params))
    dico_gabor = Normalize(dico_gabor)
    return dico_gabor

def Rebuilt(image,code,dico):
    all_indice = torch.transpose(code._indices(),0,1)
    output = torch.zeros(image.size())
    padding = dico.size()[2]-1
    for idx, (i,c) in enumerate(zip(all_indice,code._values())):
        output[i[0],:,i[2]:i[2]+padding+1,i[3]:i[3]+padding+1].add_(c*dico[i[1],:,:,:])
    return output

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

def ChangeBatchSize(data, batch_size):
    nb_image = data[0].size()[1]
    image_size=data[0].size()[2:]
    image = data[0].contiguous().view(nb_image//batch_size,batch_size,image_size[0],image_size[1],image_size[2])
    label = data[1].contiguous().view(nb_image//batch_size,-1)
    return (image,label)
###



def SaveNetwork(Network, saving_path):
    with open(saving_path, 'wb') as file:
        pickle.dump(Network, file, pickle.HIGHEST_PROTOCOL)
    print('file saved')

def LoadNetwork(loading_path):
    with open(loading_path, 'rb') as file:
        Net = pickle.load(file)
    return Net
