from CHAMP.DataTools import Normalize
from CHAMP.DataLoader import GenerateRound
from torch.nn.functional import conv2d, pad
import torch
import numpy as np
from torch.autograd import Variable
import time
from sklearn.decomposition import PCA

class CHAMP_LayerV2():
    def __init__(self, l0_sparseness=10, sigma=None, nb_dico=30, dico_size=(13,13)
                , verbose=0, doSym=False, FilterStyle='Gaussian'):

        #self.dico_size = tuple((dictionary.size()[2],dictionary.size()[3]))
        #self.nb_dico = dictionary.size()[0]
        #self.dictionary = dictionary
        self.nb_dico = nb_dico
        self.dico_size = dico_size
        self.l0_sparseness = l0_sparseness
        self.verbose = verbose
        self.sigma = sigma
        self.trained = False
        self.type = 'Unsupervised'
        self.doSym = doSym
        self.FilterStyle=FilterStyle
        #self.nb_image = input_image.size()[0]

        #self.dico_size = tuple((dictionary.size()[2],dictionary.size()[3]))
        #self.padding = self.dico_size[0]-1


    def RunLayerV2(self, dataset, dictionary):
        self.dictionary = dictionary
        coding_list = list()
        for i,(image) in enumerate(dataset[0]) :
            label = dataset[1][i]
            #label = dataset[1][i,:]
            code, res, nb_activation, residual_patch = CodingCHAMPV2(image, self.dictionary, \
                                        l0_sparseness=self.l0_sparseness, modulation=None, train=False, doSym=self.doSym)

            #code = torch.sparse.FloatTensor(event[0],event[1], torch.Size([image.size()[0],self.nb_dico,image.size()[2],image.size()[3]]))

            coding_list.append((code,label))
        return coding_list

    def TrainLayerV2(self, data_set, eta=0.05, nb_epoch=2, eta_homeo=0.01):
        self.nb_epoch = nb_epoch
        self.eta = eta
        self.eta_homeo = eta_homeo

        self.dictionary = np.random.rand(self.nb_dico,1,self.dico_size[0],self.dico_size[1])
        self.dictionary = Variable(torch.FloatTensor(self.dictionary))
        if self.sigma is not None :
            mask = GenerateRound(self.dictionary,sigma=self.sigma, style=self.FilterStyle)
        else :
            mask = Variable(torch.ones(self.dictionary.size()))
        #print(mask)
        activation = torch.zeros(self.nb_dico)
        self.res_list = list()
        modulation = torch.ones(self.nb_dico)
        dico = self.dictionary
        for i_epoch in range(self.nb_epoch):
            tic = time.time()
            for i,image in enumerate(data_set[0]) :
                code, res, nb_activation, residual_patch = CodingCHAMPV2(image, dico, \
                                            l0_sparseness=self.l0_sparseness, modulation=modulation,train=True, \
                                            doSym=self.doSym,sigma=self.sigma, style=self.FilterStyle)
                #print(residual_patch)
                residual_patch = torch.mul(residual_patch,mask.data)
                dico.data.add_(residual_patch.mul_(self.eta).view(dico.size()))
                #dico = torch.mul(dico, mask)
                dico = Normalize(dico)
                self.res_list.append(res)
                activation += nb_activation
                if eta_homeo is not None :
                    modulation = UpdateModulation(modulation, activation, eta_homeo=self.eta_homeo)
            if self.verbose != 0 and (i_epoch+1)%50==0 :
                print('epoch {0} done in {1:.2f} s'.format(i_epoch, time.time()-tic))
        self.dictionary = dico
        self.activation = activation
        self.trained=True
        return self.dictionary

    def TrainLayer0(self, data_set, eta=0.05, nb_epoch=2, eta_homeo=0.01):
        self.nb_epoch = nb_epoch
        self.eta = eta
        self.eta_homeo = eta_homeo

        self.dictionary = np.random.rand(self.nb_dico,1,self.dico_size[0],self.dico_size[1])
        self.dictionary = Variable(torch.FloatTensor(self.dictionary))
        if self.sigma is not None :
            mask = GenerateRound(self.dictionary,sigma=self.sigma, style=self.FilterStyle)
        else :
            mask = Variable(torch.ones(self.dictionary.size()))
        activation = torch.zeros(self.nb_dico)
        self.res_list = list()
        modulation = torch.ones(self.nb_dico)
        dico = self.dictionary
        for i_epoch in range(self.nb_epoch):
            tic = time.time()
            for i,wrapped_data in enumerate(data_set) :
                image,label = wrapped_data
                if i == 0:
                    print(image.size())
                    print(label.size())
                code, res, nb_activation, residual_patch = CodingCHAMP(image, dico, \
                                            l0_sparseness=self.l0_sparseness, modulation=modulation,train=True, doSym=self.doSym)
                ### Modification avec Laurent
                residual = torch.mul(residual,mask)
                dico.data.add_(residual_patch.mul_(self.eta).view(dico.size()))

                #dico = torch.mul(dico,mask)
                dico = Normalize(dico)
                self.res_list.append(res)
                activation += nb_activation
                if eta_homeo is not None :
                    modulation = UpdateModulation(modulation, activation, eta_homeo=self.eta_homeo)
            if self.verbose != 0 and i_epoch%50==0:
                print('epoch {0} done in {1:.2f} s'.format(i_epoch, time.time()-tic))
        self.dictionary = dico
        self.activation = activation
        self.trained=True
        return self.dictionary


def UpdateModulation( Modulation, activation, eta_homeo):
    target = torch.mean(activation)
    tau = - (torch.max(activation)-target)/np.log(0.2)
    modulation_function = torch.exp( (1-eta_homeo)*torch.log(Modulation) - eta_homeo*((activation-target)/tau))
    return modulation_function


def CodingCHAMPV2(image_input, dictionary, l0_sparseness=2, \
                modulation=None, verbose=0,train=True, doSym=False,sigma=None,style='Gaussian'):
    nb_image = image_input.size()[0]
    image_size = image_input.size()[2]
    #print("taille de l'image",image_input.size())
    #print("taille du dico",dictionary.size())
    dico_shape = tuple((dictionary.size()[2:4])) ## size (h_d,w_d)
    nb_dico = dictionary.size()[0]
    ## Could be once for all to speed up
    if sigma is not None :
        mask_int = GenerateRound(dictionary,sigma=sigma, style=style)
        #mask_int = Normalize(mask_int)
        mask = mask_int[0,0,:,:]
    else :
        #mask = Variable(torch.ones(dictionary.size()[2],dictionary.size()[3]))
        mask = Variable(torch.ones(dictionary.size()))
    #dictionary = torch.mul(dictionary,mask)

    #dictionary = torch.mul(dictionary,mask)
    dictionary = Normalize(dictionary)

    padding = dico_shape[0]-1
    X_conv = conv2d(dictionary,dictionary,bias=None,padding=padding).data

    Conv0 = conv2d(image_input,dictionary,bias=None) ## size(nb_image,nb_dico,h_i-h_d+1,w_i-w_d+1)
    #print(Conv0.size())
    Conv_padded = pad(Conv0[:,:,:,:],pad=(padding,padding,padding,padding),mode='constant')
    Conv_size = tuple(Conv0.size())
    Conv = Conv0.data.view(-1,Conv_size[1]*Conv_size[2]*Conv_size[3])
    Sparse_code_addr = torch.zeros(4, nb_image * l0_sparseness).long()
    Sparse_code_coeff = torch.zeros(nb_image * l0_sparseness)

    residual_image = image_input.data.clone()
    residual_patch = torch.zeros(nb_dico,1,dictionary.size()[2],dictionary.size()[3])
    nb_activation = torch.zeros(nb_dico)
    if modulation is None:
        modulation = torch.ones(nb_dico)
    Mod = modulation.unsqueeze(1).unsqueeze(2).expand_as(Conv0[0,:,:,:])
    Mod = Mod.contiguous().view(Conv[0].size())
    idx = 0
    for i_m in range(nb_image):
        #Conv_one_image = Conv0[im,:,:,:] ## size(nb_dico,h_i-h_d+1,w_i-w_d+1)
        Conv_one_image = Conv[i_m,:] ##size(nb_dico*(h_i-h_d+1)*(w_i-w_d+1))
        coeff_memory = torch.zeros(Conv_one_image.size())

        for i_l0 in range(l0_sparseness):

            ConvMod = Conv_one_image*Mod
            #print('before normalization',ConvMod.size())
            #m_value, m_ind = torch.max(Conv_one_image,0)
            if doSym == True :
                _, m_ind = torch.max(torch.abs(ConvMod),0)
            elif doSym == 'norm':
                ConvMod = Normalize(ConvMod.view(1,Conv_size[1],Conv_size[2],Conv_size[3])).view(-1)
                _, m_ind = torch.max(torch.abs(ConvMod),0)
                #print('after normalization',ConvMod.size())
            else :
                _, m_ind = torch.max(ConvMod,0)

            indice = np.unravel_index(int(m_ind.numpy()),Conv_size)
            m_value = Conv_one_image[m_ind]
            c_ind = m_value/X_conv[indice[1],indice[1],dico_shape[0]-1,dico_shape[1]-1]
            coeff_memory[m_ind] += c_ind
            Sparse_code_addr[:,idx] = torch.LongTensor([int(i_m),int(indice[1]),int(indice[2]),int(indice[3])])
            Sparse_code_coeff[idx] = float(coeff_memory[m_ind].numpy())
            Conv_padded[i_m,:,indice[2]+padding-padding:indice[2]+padding+padding+1,indice[3]+padding-padding:indice[3]+padding+padding+1].\
            data.add_(-c_ind*X_conv[indice[1],:,:])
            Conv_int = Conv_padded[i_m,:,padding:-padding,padding:-padding].contiguous()
            Conv_one_image = Conv_int.data.view(-1)
            if train == True:
                #residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1] = torch.add(\
                #torch.mul(residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1],mask.data)\
                #,-c_ind*dictionary[indice[1],0,:,:].data)
                residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1].add_(-c_ind*dictionary[indice[1],0,:,:].data)
                residual_patch[indice[1],0,:,:].add_(residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1])
                nb_activation[indice[1]] +=1
            idx += 1
    if train == True:
        res = torch.mean(torch.norm(residual_image.view(nb_image,image_size*image_size),p=2,dim=1))
        mean = residual_patch / nb_activation.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(residual_patch)
        residual_patch[mean==mean] = mean[mean==mean]
    else :
        res=0
    code = torch.sparse.FloatTensor(Sparse_code_addr, Sparse_code_coeff, torch.Size([nb_image,nb_dico,Conv_size[2],Conv_size[3]]))
    return code, res, nb_activation, residual_patch
