from CHAMP.DataTools import Normalize
from CHAMP.DataLoader import GenerateRound
from torch.nn.functional import conv2d, pad
import torch
import numpy as np
from torch.autograd import Variable
import time

'''
Class defining a CHAMP Layer (A Layer of Convolutional Matching Pursuit)
Initialization parameters :
    * nb_dico (int) : number of filters in the dictionary
    * l0_sparseness (int) : number of activation per image
    * dico_size (int,int) : height and widht of the filters in the dictionary
    * sigma (float) : parameter to set the mask's gaussian volatility. If sigma is None, the masks are not applied
    * verboe (int) : control the verbosity. Could be superior to 1 to monitor the training
    * FilterStyle (string) : Define the type of mask to apply
    * MaskMode (string) : Option to control where the mask are applied
'''

class CHAMP_Layer():
    def __init__(self, l0_sparseness=10, sigma=None, nb_dico=30, dico_size=(13,13)
                , verbose=0, doSym=False, FilterStyle='Gaussian',MaskMod='Residual'\
                ,learning='Hebbian'):

        self.nb_dico = nb_dico
        self.dico_size = dico_size
        self.l0_sparseness = l0_sparseness
        self.verbose = verbose
        self.sigma = sigma
        self.trained = False
        self.type = 'Unsupervised'
        self.doSym = doSym
        self.FilterStyle=FilterStyle
        self.MaskMod = MaskMod
        self.learning = learning


    def RunLayer(self, dataset, dictionary):
        self.dictionary = dictionary
        coding_list = list()
        for i,(image) in enumerate(dataset[0]) :
            label = dataset[1][i]
            code, res, nb_activation, residual_patch = self.CodingCHAMP(image, self.dictionary, \
                                        l0_sparseness=self.l0_sparseness, modulation=None, train=False, doSym=self.doSym)
            coding_list.append((code,label))
        return coding_list

    def TrainLayer(self, data_set, eta=0.05, nb_epoch=2, eta_homeo=0.01, nb_record=5,dico_init=None):
        self.nb_epoch = nb_epoch
        self.eta = eta
        self.eta_homeo = eta_homeo
        if dico_init is None :
            self.dictionary = np.random.rand(self.nb_dico,1,self.dico_size[0],self.dico_size[1])
            self.dictionary = Normalize(Variable(torch.FloatTensor(self.dictionary)))
        else :
            self.dictionary = Normalize(dico_init)
        if self.sigma is not None :
            mask = GenerateRound(self.dictionary,sigma=self.sigma, style=self.FilterStyle)

        else :
            mask = Variable(torch.ones(self.dictionary.size()))
        #mask = GenerateRound(self.dictionary,sigma=0.8, style='Gaussian')
        #self.dictionary = Normalize(self.dictionary*mask)

        self.activation = torch.zeros(self.nb_dico)
        self.res_list = list()
        self.gradient = list()
        modulation = torch.ones(self.nb_dico)
        tic = time.time()
        nb_applied_dico = 10
        first_epoch = 1
        for i_epoch in range(self.nb_epoch):
            for i,image in enumerate(data_set[0]) :
                code, res, nb_activation, residual_patch,reconstructed_image, residual_image = self.CodingCHAMP(image, self.dictionary, \
                                            l0_sparseness=self.l0_sparseness, modulation=modulation,train=True, \
                                            doSym=self.doSym,sigma=self.sigma, style=self.FilterStyle)
                if self.learning == 'Hebbian':
                    self.dictionary.data.add_(residual_patch.mul_(self.eta).view(self.dictionary.size()))
                    self.dictionary = Normalize(self.dictionary)

                self.res_list.append(res)
                self.activation += nb_activation
                if eta_homeo is not None :
                    modulation = self.UpdateModulation(modulation, self.activation, eta_homeo=self.eta_homeo)
            if self.verbose != 0 :
                if ((i_epoch+1)%(self.nb_epoch//self.verbose)) == 0:
                    timing = time.time()-tic
                    print('epoch {0} - {1} done in {2}m{3}s'.format(first_epoch, i_epoch+1, int(timing//60),int(timing%60)))
                    tic, first_epoch = time.time(),  i_epoch+1
        self.dictionary = torch.mul(self.dictionary, mask)
        self.trained=True
        return self.dictionary, reconstructed_image, residual_image


    def UpdateModulation(self, Modulation, activation, eta_homeo):
        target = torch.mean(activation)
        tau = - (torch.max(activation)-target)/np.log(0.2)
        modulation_function = torch.exp( (1-eta_homeo)*torch.log(Modulation) - eta_homeo*((activation-target)/tau))
        return modulation_function


    def CodingCHAMP(self, image_input, dictionary, l0_sparseness=2, \
                modulation=None, verbose=0,train=True, doSym=False,sigma=None,style='Gaussian'):
        nb_image = image_input.size()[0]
        image_size = image_input.size()[2]
        dico_shape = tuple((dictionary.size()[2],dictionary.size()[3]))
        nb_dico = dictionary.size()[0]
        ## Could be once for all to speed up
        if sigma is not None and self.MaskMod=='Residual':
            mask = GenerateRound(dictionary,sigma=sigma, style=style)
        else :
            mask = Variable(torch.ones(dictionary.size()[2],dictionary.size()[3]))

        padding = dico_shape[0]-1
        X_conv = conv2d(dictionary,dictionary,bias=None,padding=padding).data
        Conv0 = conv2d(image_input,dictionary,bias=None)
        Conv_padded = pad(Conv0[:,:,:,:],pad=(padding,padding,padding,padding),mode='constant')
        Conv_size = tuple(Conv0.size())
        Conv = Conv0.data.view(-1,Conv_size[1]*Conv_size[2]*Conv_size[3])
        Sparse_code_addr = torch.zeros(4, nb_image * l0_sparseness).long()
        Sparse_code_coeff = torch.zeros(nb_image * l0_sparseness)

        residual_image = image_input.data.clone()
        reconstructed_image = torch.zeros(image_input.size())
        residual_patch = torch.zeros(nb_dico,1,dictionary.size()[2],dictionary.size()[3])
        nb_activation = torch.zeros(nb_dico)
        if modulation is None:
            modulation = torch.ones(nb_dico)
        Mod = modulation.unsqueeze(1).unsqueeze(2).expand_as(Conv0[0,:,:,:])
        Mod = Mod.contiguous().view(Conv[0].size())
        idx = 0
        for i_m in range(nb_image):
            Conv_one_image = Conv[i_m,:]
            coeff_memory = torch.zeros(Conv_one_image.size())

            for i_l0 in range(l0_sparseness):

                ConvMod = Conv_one_image*Mod
                if doSym == True :
                    _, m_ind = torch.max(torch.abs(ConvMod),0)
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
                    reconstructed_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1].add_(c_ind*dictionary[indice[1],0,:,:].data)
                    if self.MaskMod == 'Residual':
                        patch_before = residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1]
                        residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1].add_(-c_ind*dictionary[indice[1],0,:,:].data)
                        residual_patch[indice[1],0,:,:] += (patch_before*mask.data[0,0,:,:] - c_ind*dictionary[indice[1],0,:,:].data)
                        #patch = residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1].clone()
                        #residual_patch[indice[1],0,:,:] += (patch + c_ind*dictionary[indice[1],0,:,:].data)*mask.data[0,0,:,:] - c_ind*dictionary[indice[1],0,:,:].data
                    else :
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
        return code, res, nb_activation, residual_patch, reconstructed_image,residual_image

'''
def RunLayer0(self, dataset, dictionary):
    self.dictionary = dictionary
    coding_list = list()
    for i,wrapped_data in enumerate(dataset) :
        image,label = wrapped_data
        #label = dataset[1][i,:]
        code, res, nb_activation, residual_patch = CodingCHAMP(image, self.dictionary, \
                                    l0_sparseness=self.l0_sparseness, modulation=None, train=True, doSym=self.doSym)

        #code = torch.sparse.FloatTensor(event[0],event[1], torch.Size([image.size()[0],self.nb_dico,image.size()[2],image.size()[3]]))
        coding_list.append((code,label))
    return coding_list
'''
'''
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
        if self.verbose != 0:
            print('epoch {0} done in {1:.2f} s'.format(i_epoch, time.time()-tic))
    self.dictionary = dico
    self.activation = activation
    self.trained=True
    return self.dictionary
'''
