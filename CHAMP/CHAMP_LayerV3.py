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

        self.nb_dico = nb_dico
        self.dico_size = dico_size
        self.l0_sparseness = l0_sparseness
        self.verbose = verbose
        self.sigma = sigma
        self.trained = False
        self.type = 'Unsupervised'
        self.doSym = doSym
        self.FilterStyle=FilterStyle

    def RunLayer(self, dataset, dictionary):
        self.dictionary = dictionary
        coding_list = list()
        for i,(image) in enumerate(dataset[0]) :
            label = dataset[1][i]
            #label = dataset[1][i,:]
            code, res, nb_activation, residual_patch = CodingCHAMP(image, self.dictionary, \
                                        l0_sparseness=self.l0_sparseness, modulation=None, train=False, doSym=self.doSym)

            #code = torch.sparse.FloatTensor(event[0],event[1], torch.Size([image.size()[0],self.nb_dico,image.size()[2],image.size()[3]]))

            coding_list.append((code,label))
        return coding_list

    def TrainLayer_V2(self, data_set, eta=0.05, nb_epoch=2, eta_homeo=0.01):
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
            for i,image in enumerate(data_set[0]) :
                resi = torch.zeros(self.nb_dico,1,self.dico_size[0],self.dico_size[1])
                code, res, nb_activation, residual_patch, selected_patch, te = CodingCHAMP_V2(image, dico, \
                                            l0_sparseness=self.l0_sparseness, modulation=modulation,train=True, doSym=self.doSym)
                for i_dico in range(self.nb_dico):
                    addr_dico = te[:,te[1,:]==i_dico]
                    all_dico = selected_patch[addr_dico[0],addr_dico[1],:,:].view(-1,self.dico_size[0]*self.dico_size[1]).numpy()
                    pca = PCA(n_components=1)
                    pca.fit(all_dico)
                    inter = torch.FloatTensor(pca.components_[0,:].reshape((self.dico_size[0],self.dico_size[1])))
                    resi[i_dico,0,:,:] = inter
                #dico.data.add_(resi.mul_(self.eta).view(dico.size()))
                dico.data.add_(resi)
                    #print(inter.size())
                #dico.data.add_(residual_patch.mul_(self.eta).view(dico.size()))
                dico = torch.mul(dico,mask)
                #dico = Normalize(dico)
                self.res_list.append(res)
                activation += nb_activation
                if eta_homeo is not None :
                    modulation = UpdateModulation(modulation, activation, eta_homeo=self.eta_homeo)
                if i == 1:
                    break
            if self.verbose != 0 and i_epoch%50==0:
                print('epoch {0} done in {1:.2f} s'.format(i_epoch, time.time()-tic))
        self.dictionary = dico
        self.activation = activation
        self.trained=True
        return self.dictionary

def CodingCHAMP_V2(image_input, dictionary, l0_sparseness=2, \
                modulation=None, verbose=0,train=True, doSym=False):
    nb_image = image_input.size()[0]
    image_size = image_input.size()[2]
    dico_shape = tuple((dictionary.size()[2],dictionary.size()[3]))
    nb_dico = dictionary.size()[0]

    #dictionary = Normalize(dictionary)
    padding = dico_shape[0]-1
    X_conv = conv2d(dictionary,dictionary,bias=None,padding=padding).data

    Conv0 = conv2d(image_input,dictionary,bias=None)
    #print('Xconv : ',X_conv)
    #print('Conv0 : ',Conv0)
    Conv_padded = pad(Conv0[:,:,:,:],pad=(padding,padding,padding,padding),mode='constant')
    Conv_size = tuple(Conv0.size())
    Conv = Conv0.data.view(-1,Conv_size[1]*Conv_size[2]*Conv_size[3])
    enum_l0=torch.arange(0,l0_sparseness).long()
    Sparse_code_addr = torch.zeros(4, nb_image * l0_sparseness).long()
    Sparse_code_coeff = torch.zeros(nb_image * l0_sparseness)

    residual_image = image_input.data.clone()
    selected_patch = torch.zeros(nb_image,nb_dico,dictionary.size()[2],dictionary.size()[3])
    residual_patch = torch.zeros(nb_dico,1,dictionary.size()[2],dictionary.size()[3])
    te = np.zeros((2,nb_image * l0_sparseness)).astype(int)
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
            #m_value, m_ind = torch.max(Conv_one_image,0)
            if doSym == True :
                _, m_ind = torch.max(torch.abs(ConvMod),0)
            else :
                _, m_ind = torch.max(ConvMod,0)

            indice = np.unravel_index(int(m_ind.numpy()),Conv_size)
            m_value = Conv_one_image[m_ind]
            c_ind = m_value/X_conv[indice[1],indice[1],dico_shape[0]-1,dico_shape[1]-1]
            coeff_memory[m_ind] += c_ind
            Sparse_code_addr[:,idx] = torch.LongTensor([int(i_m),int(indice[1]),int(indice[2]),int(indice[3])])
            #Sparse_code_coeff[idx] = float(c_ind.numpy())
            Sparse_code_coeff[idx] = float(coeff_memory[m_ind].numpy())


            Conv_padded[i_m,:,indice[2]+padding-padding:indice[2]+padding+padding+1,indice[3]+padding-padding:indice[3]+padding+padding+1].\
            data.add_(-c_ind*X_conv[indice[1],:,:])
            Conv_int = Conv_padded[i_m,:,padding:-padding,padding:-padding].contiguous()
            Conv_one_image = Conv_int.data.view(-1)
            if train == True:
                #selected_patch[,i_m,:,:] = residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1]]
                #selected_patch[i_m,indice[1],:,:] = residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1]
                residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1].add_(-c_ind*dictionary[indice[1],0,:,:].data)
                residual_patch[indice[1],0,:,:].add_(residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1])
                selected_patch[i_m,indice[1],:,:] = residual_image[i_m,0,indice[2]:indice[2]+padding+1,indice[3]:indice[3]+padding+1]
                te[0, idx] = i_m
                te[1, idx] = indice[1]
                nb_activation[indice[1]] +=1
            idx += 1
    if train == True:
        res = torch.mean(torch.norm(residual_image.view(nb_image,image_size*image_size),p=2,dim=1))
        mean = residual_patch / nb_activation.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(residual_patch)
        residual_patch[mean==mean] = mean[mean==mean]
    else :
        res=0
    code = torch.sparse.FloatTensor(Sparse_code_addr, Sparse_code_coeff, torch.Size([nb_image,nb_dico,Conv_size[2],Conv_size[3]]))
    return code, res, nb_activation, residual_patch, selected_patch, te
    #return (Sparse_code_addr, Sparse_code_coeff), res, nb_activation, residual_patch
