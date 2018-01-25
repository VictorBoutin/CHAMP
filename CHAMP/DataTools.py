import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d
import pickle
import cv2
import math
from CHAMP.LowLevel import conv, Normalize


def ContrastNormalized(data,avg_size=(5,5)):
    img_size = data[0].size()
    output_tensor = torch.FloatTensor(img_size)
    padding = avg_size[0]-1
    output_tensor2 = torch.FloatTensor(img_size[0],img_size[1],img_size[2],img_size[3]-padding,img_size[4]-padding)
    to_conv = (torch.ones(avg_size)*1/(avg_size[0]*avg_size[0])).view(1,1,avg_size[0],avg_size[1])
    for idx_batch, each_batch in enumerate(data[0]):
        convol = conv(each_batch,to_conv,padding=padding//2)
        output_tensor[idx_batch,:,:,:,:] = each_batch - convol
        output_tensor2 = output_tensor[:,:,:,padding//2:-padding//2,padding//2:-padding//2]
    return output_tensor2, data[1]

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

###

def Decorrelate(dataset):
    filt_decor = torch.Tensor(1,1,3,3)
    filt_decor[0,0,:,:] = torch.FloatTensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
    to_out = list()
    for i, each_set in enumerate(dataset):
        images, labels = each_set
        images = images
        to_out.append((conv(images, filt_decor,padding=1),labels))
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
