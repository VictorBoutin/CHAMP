from torch.autograd import Variable
from torch.nn.functional import conv2d,pad, max_pool2d
import torch
import numpy as np

def maxpool2D(data,kernel):
    output = torch.zeros(data[0].size()[0],data[0].size()[1],data[0].size()[2],data[0].size()[3]//kernel,data[0].size()[4]//kernel)
    for i_batch in range(data[0].size()[0]):
        batch = Variable(data[0][i_batch,:,:,:,:])
        output[i_batch,:,:,:,:] = max_pool2d(batch,kernel_size=kernel).data
    return (output,data[1])

def conv(data, filters, padding=0,GPU=False,groups=1,stride=1):
    '''
    Convolution fonction on tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
        * filters <torch.tensor(nb_dico, nb_polarities, w_d, h_d)> :  filters in a tensor format
        * padding <int> or <(int,int,int,int)> : padding of the 2 last dimensions. If this is an <int>
            all the vertical/horizontal & right/left paddings are the same.
    OUTPUT :
        * output <torch.tensor(nb_image, nb_polarities, w+padding-wd+1, h+padding-h_d+1)> : convulution between
        data and filters
    '''
    if GPU:
        filters,data = filters.cuda(), data.cuda()
    filters, data = Variable(filters), Variable(data)
    output = conv2d(data,filters,padding=padding,bias=None,groups=groups,stride=stride)
    return output.data

def realconv(data, filters, padding=0,GPU=False,groups=1,stride=1):
    filt = filters.clone()
    filt = RotateDico90(filt,2)
    filt, data = Variable(filt), Variable(data)
    output = conv2d(data,filt,padding=padding,bias=None,groups=groups,stride=stride)
    return output.data

def conv_valid(data, filters, padding=0,GPU=False,groups=1):
    '''
    Convolution fonction on tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
        * filters <torch.tensor(nb_dico, nb_polarities, w_d, h_d)> :  filters in a tensor format
        * padding <int> or <(int,int,int,int)> : padding of the 2 last dimensions. If this is an <int>
            all the vertical/horizontal & right/left paddings are the same.
    OUTPUT :
        * output <torch.tensor(nb_image, nb_polarities, w+padding-wd+1, h+padding-h_d+1)> : convulution between
        data and filters
    '''
    filters_size = filters.size()[-1]
    if GPU:
        filters,data = filters.cuda(), data.cuda()
    filters, data = Variable(filters), Variable(data)
    pad_size = filters_size//2
    padded_tensor = pad(data,pad=(pad_size, pad_size, pad_size, pad_size),
                      mode='reflect', value=0)
    output = conv2d(padded_tensor,filters,bias=None,groups=groups)
    return output.data

def padTensor(data, padding,mode='constant',value=0):
    '''
    padding function on tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
        * padding <int> : value of the padding at each side of the 2 last dimension of data
        * mode <string> : if 'constant', pad with a constant value (indicated by value), if 'relfect' padding
            are reflexion of the side of the image, if 'replicate' these are the replication
        * value <Float> : value of the padding
    OUTPUT :
        * output <torch.tensor(nb_image, nb_polarities, w+padding, h+padding)> padded version of the input data
    '''
    output = pad(data[:, :, :, :],
                     pad=(padding, padding, padding, padding),
                      mode=mode, value=value)
    return output.data

def Normalize(to_normalize,order=2):
    '''
    L2 Normalize the 3 last dimensions of a tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
    OUTPUT :
        * dico <torch.tensor(nb_image, nb_polarities, w, h)> : normalized version of data
    '''
    size = tuple(to_normalize.size())
    #reshaped = to_normalize.view(size[0]*size[1], size[2]*size[3])
    reshaped = to_normalize.view(size[0], size[1]*size[2]*size[3])
    norm = torch.norm(reshaped, p=order, dim=1)
    norm_int = norm.unsqueeze(1).expand_as(reshaped)
    dico = reshaped.div(norm_int).view(size)
    return dico


def Normalize_last2(to_normalize,order=2):
    '''
    L2 Normalize the 3 last dimensions of a tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
    OUTPUT :
        * dico <torch.tensor(nb_image, nb_polarities, w, h)> : normalized version of data
    '''
    size = tuple(to_normalize.size())
    reshaped = to_normalize.view(size[0]*size[1], size[2]*size[3])
    #reshaped = to_normalize.view(size[0], size[1]*size[2]*size[3])
    norm = torch.norm(reshaped, p=order, dim=1)
    norm_int = norm.unsqueeze(1).expand_as(reshaped)
    dico = reshaped.div(norm_int).view(size)
    return dico

def RotateDico90(dico,k=2):
    dico_rotated = np.rot90(dico.numpy(),k=2,axes=(2, 3)).copy()
    return torch.FloatTensor(dico_rotated)

def PatchExtractor(data,loc_x,loc_y,size):
    das = np.lib.stride_tricks.as_strided(data, (data.shape[2]-size+1, data.shape[3]-size+1, data.shape[0], data.shape[1], size, size), data.strides[-2:] + data.strides)
    #patches = das[(loc_x,loc_y, np.arange(data.shape[0]))]
    #return patches
    return das

def unravel_index(indice,size,GPU=False):
    idx=[]
    for each_size in size[::-1]:
        idx = [indice%each_size] + idx
        indice = indice/each_size
    if GPU==True:
        idx = torch.cat(idx).cuda()
    else :
        idx = torch.cat(idx)
    return idx
