import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
import pickle


def convolve2D(data, filters, padding):
    '''
    Convolution fonction on tenseur:
    INPUT :
        * data <torch.tensor(nb_image, nb_polarities, w, h)> : input image in a tensor format
        * filters <torch.tensor(nb_dico, nb_polarities, w_d, h_d)> :  filters in a tensor format
        * padding <int> or <(int,int,int,int)> : padding of the 2 last dimensions. If this is an <int>
            all the vertical/horizontal & right/left paddings are the same.
    '''
    filters, data = Variable(filters), Variable(data)
    output = conv2d(data,filters,padding=padding)
    return output.data


def Normalize(to_normalize):
    size = tuple(to_normalize.size())
    reshaped = to_normalize.contiguous().view(size[0]*size[1], size[2]*size[3])
    #reshaped = to_normalize.view(size[0], size[2]*size[3]) ## Chaged for L2

    try :
        norm = torch.norm(reshaped, p=2, dim=1).detach()
    except:
        norm = torch.norm(reshaped, p=2, dim=1)
    norm_int = norm.unsqueeze(1).expand_as(reshaped)
    dico = reshaped.div(norm_int).view(size)
    return dico

def Decorrelate(dataset):
    filt_decor = Variable(torch.Tensor(1,1,3,3))
    filt_decor[0,0,:,:] = torch.FloatTensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
    to_out = list()
    for i, each_set in enumerate(dataset):
        images, labels = each_set
        images = Variable(images)
        to_out.append((conv2d(images, filt_decor,padding=1),labels))
    return to_out

def Decorrelate2(dataset):
    filt_decor = Variable(torch.Tensor(1,1,3,3))
    filt_decor[0,0,:,:] = torch.FloatTensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
    to_out = torch.FloatTensor(120,500,1,28,28)
    for i, each_set in enumerate(dataset):
        images, labels = each_set
        images = Variable(images)
        a = conv2d(images, filt_decor,padding=1)
        #print(a.size())
        to_out[i,:,:,:,:]=conv2d(images, filt_decor,padding=1).data
    return to_out


def SaveNetwork(Network, saving_path):
    with open(saving_path, 'wb') as file:
        pickle.dump(Network.Layers, file, pickle.HIGHEST_PROTOCOL)
    print('file saved')

def LoadNetwork(loading_path):
    with open(loading_path, 'rb') as file:
        Net = pickle.load(file)
    return Net

def ConstrastNormalized(data,avg_size=(5,5)):
    output_tensor = torch.FloatTensor(data[0].size())
    output_tensor2 = torch.FloatTensor(data[0].size()[0],data[0].size()[1],data[0].size()[2],data[0].size()[3]-4,data[0].size()[4]-4)
    to_conv = torch.ones(avg_size)*1/(avg_size[0]*avg_size[0])
    to_conv2 = Variable(to_conv.view(1,1,avg_size[0],avg_size[1]))
    for idx_batch, each_batch in enumerate(data[0]):
        conv = conv2d(each_batch,to_conv2,padding=2)
        output_tensor[idx_batch,:,:,:,:] = each_batch.data - conv.data
        output_tensor2 = output_tensor[:,:,:,2:-2,2:-2]

    return (Variable(output_tensor2), data[1])

def zero_one_norm(data_input):
    image = data_input[0].data
    output = torch.FloatTensor(image.size())
    for idx_batch, each_batch in enumerate(image):
        each_batch = each_batch.contiguous().view(-1,each_batch.size()[2]*each_batch.size()[3])
        mini,_ = torch.min(each_batch,dim=1)
        mini = mini.unsqueeze(1).expand_as(each_batch)
        each_batch -= mini
        maxi,_= torch.max(each_batch,dim=1)
        maxi = maxi.unsqueeze(1).expand_as(each_batch)
        each_batch /= maxi
        each_batch = each_batch.view(image[0,:,:,:,:].size())
        output[idx_batch,:,:,:,:] = each_batch
    return (Variable(output),data_input[1])

def MakePositive(data_input):
    image = data_input[0].data
    output = torch.FloatTensor(image.size())
    for idx_batch, each_batch in enumerate(image):
        each_batch = each_batch.contiguous().view(-1,each_batch.size()[2]*each_batch.size()[3])
        mini,_ = torch.min(each_batch,dim=1)
        mini = mini.unsqueeze(1).expand_as(each_batch)
        each_batch -= mini
        #maxi,_= torch.max(each_batch,dim=1)
        #maxi = maxi.unsqueeze(1).expand_as(each_batch)
        #each_batch /= maxi
        each_batch = each_batch.view(image[0,:,:,:,:].size())
        output[idx_batch,:,:,:,:] = each_batch
    return (Variable(output),data_input[1])
