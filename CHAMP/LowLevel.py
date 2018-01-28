from torch.autograd import Variable
from torch.nn.functional import conv2d,pad
import torch

def conv(data, filters, padding=0):
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
    filters, data = Variable(filters), Variable(data)
    output = conv2d(data,filters,padding=padding,bias=None)
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

def Normalize(to_normalize):
    '''
    L2 Normalize the 3 last dimensions of a tensor.
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
    OUTPUT :
        * dico <torch.tensor(nb_image, nb_polarities, w, h)> : normalized version of data
    '''
    size = tuple(to_normalize.size())
    reshaped = to_normalize.view(size[0], size[1]*size[2]*size[3])
    norm = torch.norm(reshaped, p=2, dim=1)
    norm_int = norm.unsqueeze(1).expand_as(reshaped)
    dico = reshaped.div(norm_int).view(size)
    return dico
