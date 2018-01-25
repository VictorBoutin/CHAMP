from torch.autograd import Variable
from torch.nn.functional import conv2d,pad
import torch

def conv(data, filters, padding=0):
    '''
    Convolution fonction on tenseur:
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
        * filters <torch.tensor(nb_dico, nb_polarities, w_d, h_d)> :  filters in a tensor format
        * padding <int> or <(int,int,int,int)> : padding of the 2 last dimensions. If this is an <int>
            all the vertical/horizontal & right/left paddings are the same.
    '''
    filters, data = Variable(filters), Variable(data)
    output = conv2d(data,filters,padding=padding,bias=None)
    return output.data

def padTensor(data, padding,mode='constant',value=0):
    output = pad(data[:, :, :, :],
                     pad=(padding, padding, padding, padding),
                      mode=mode, value=value)
    return output.data

def Normalize(to_normalize):
    size = tuple(to_normalize.size())
    reshaped = to_normalize.view(size[0], size[1]*size[2]*size[3])
    norm = torch.norm(reshaped, p=2, dim=1)
    norm_int = norm.unsqueeze(1).expand_as(reshaped)
    dico = reshaped.div(norm_int).view(size)
    return dico
