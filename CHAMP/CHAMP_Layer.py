from CHAMP.DataTools import GenerateMask
from CHAMP.LowLevel import conv, Normalize, padTensor, unravel_index
from torch.nn.functional import conv2d, pad
import torch
import numpy as np
from torch.autograd import Variable
import time
from CHAMP.LowLevel import PatchExtractor, RotateDico90
from CHAMP.Monitor import DisplayDico
class CHAMP_Layer_np:
    def __init__(self, l0_sparseness=10, nb_dico=30, dico_size=(13, 13),
                     verbose=0, doSym=False, alpha=None, mask=None,stride=1):

        self.nb_dico = nb_dico
        self.dico_size = dico_size
        self.l0_sparseness = l0_sparseness
        self.verbose = verbose
        self.trained = False
        self.type = 'Unsupervised'
        self.doSym = doSym
        self.mask = mask
        self.alpha = alpha
        self.stride = stride

    def RunLayer(self, dataset, dictionary=None):
        if dictionary is not None :
            self.dictionary = dictionary
        code_size = (dataset[0].size()[-2]-self.dico_size[0]+1,dataset[0].size()[-1]-self.dico_size[1]+1)
        code = np.zeros((dataset[0].size()[0],dataset[0].size()[1],self.nb_dico,code_size[0],code_size[1]))
        for i, (image) in enumerate(dataset[0]):
            return_fn = ConvMP_np(image, self.dictionary, l0_sparseness=self.l0_sparseness,
                                    modulation=None, train=False, doSym=self.doSym, alpha=self.alpha)
            code_one_batch, activation = return_fn
            code[i,:,:,:,:] = code_one_batch
        return (torch.FloatTensor(code),dataset[1])


    def TrainLayer(self, data_set, eta=0.05, nb_epoch=2, eta_homeo=None, nb_record=4,dico_init=None,seed=None):
        self.nb_epoch = nb_epoch
        self.eta = eta
        self.eta_homeo = eta_homeo

        nb_channel = data_set[0].size()[2]
        if seed is not None :
            np.random.seed(seed)
        if dico_init is None :
            self.dictionary = np.random.rand(self.nb_dico, nb_channel, self.dico_size[0], self.dico_size[1])
        else :
            self.dictionary = dico_init
        if self.mask is None:
            self.mask = np.ones_like(self.dictionary)
        else :
            self.mask = self.mask.numpy()
        #self.dictionary = self.dictionary*self.mask
        self.dictionary = Normalize(torch.FloatTensor(self.dictionary))

        self.res_list = list()
        conv_size = data_set[0].size(-1)-self.dico_size[0]+1
        if self.eta_homeo is not None :
            modulation = np.ones(self.nb_dico)
        else :
            modulation = None
        tic = time.time()
        first_epoch = 1
        for i_epoch in range(self.nb_epoch):
            when = np.zeros((self.nb_dico,self.l0_sparseness))
            self.activation = np.zeros(self.nb_dico)
            for idx_batch, batch  in enumerate(data_set[0]):
                return_fn = ConvMP_np(batch, self.dictionary,
                                l0_sparseness=self.l0_sparseness, modulation=modulation, verbose=self.verbose, train=True,
                                doSym=self.doSym, mask=self.mask, alpha=self.alpha,stride=self.stride,when=when)
                self.residual_image, self.code, res, nb_activation,self.where,self.when,self.how = return_fn
                dictionary,self.patches = learn(self.code, self.dictionary, self.residual_image,self.eta)
                #dictionary = learn2(self.code, self.dictionary, self.residual_image,self.eta)
                self.dictionary = torch.FloatTensor(dictionary)
                self.dictionary = Normalize(self.dictionary)
                self.res_list.append(res)
                self.activation += nb_activation
                if self.eta_homeo is not None :
                     modulation = UpdateModulation(modulation, self.activation, eta_homeo=self.eta_homeo)
            if self.verbose != 0:
                if ((i_epoch + 1) % (self.nb_epoch // self.verbose)) == 0:
                    timing = time.time() - tic
                    print('epoch {0} - {1} done in {2}m{3}s'.format(first_epoch,
                                                                    i_epoch + 1, int(timing // 60), int(timing % 60)))
                    tic, first_epoch = time.time(),  i_epoch + 1
            #DisplayDico(self.dictionary)
        return self.dictionary

def UpdateModulation( Modulation, activation, eta_homeo,how=None):
    target = np.mean(activation)
    tau = - (np.max(activation)-target)/np.log(0.1)
    modulation_function = np.exp( (1-eta_homeo)*np.log(Modulation) - eta_homeo*((activation-target)/tau))
    return modulation_function

def ConvMP_np(image_input, dictionary, l0_sparseness=2,
                modulation=None, verbose=0, train=True, doSym='pos', mask=None,\
                alpha=None,stride=1,when=None):
    nb_image = image_input.size()[0]
    image_size = image_input.size()[2]
    dico_shape = tuple((dictionary.size()[1],dictionary.size()[2], dictionary.size()[3]))
    nb_dico = dictionary.size()[0]
    padding = dico_shape[2] - stride
    tic = time.time()
    if mask is None :
        mask = np.ones(dictionary.size())
    X_conv = conv(dictionary, dictionary, padding=padding,stride=stride)
    X_conv_size = X_conv.size()[-2:]
    I_conv = conv(image_input, dictionary*torch.FloatTensor(mask),stride=stride)
    I_conv_padded = padTensor(I_conv, padding=X_conv_size[0]//2)
    Conv_size = tuple(I_conv.size())
    I_conv_ravel = I_conv.numpy().reshape(-1, Conv_size[1] * Conv_size[2] * Conv_size[3])
    X_conv = X_conv.numpy()
    I_conv_padded = I_conv_padded.numpy()
    code = np.zeros((nb_image, nb_dico, Conv_size[2], Conv_size[3]))
    activation = np.zeros(nb_dico)
    dico = dictionary.numpy()
    where = np.zeros((nb_dico,Conv_size[-2],Conv_size[-1]))
    how = np.zeros((nb_dico))
    #if modulation is None:
    #    modulation = np.ones(nb_dico)
    if modulation is not None :
        Mod = modulation[:,np.newaxis,np.newaxis]*np.ones((Conv_size[1], Conv_size[2], Conv_size[3]))
        Mod = Mod.reshape(Conv_size[1]* Conv_size[2]*Conv_size[3])
    if train == True :
        residual_image = image_input.clone().numpy()
    for i_m in range(nb_image):
        Conv_one_image = I_conv_ravel[i_m,:]
        for i_l0 in range(l0_sparseness):
            if modulation is None :
                Conv_Mod = Conv_one_image
            else :
                Conv_Mod = Conv_one_image*Mod
            m_ind = np.argmax(Conv_Mod,axis=0)
            #m_ind = np.argmax(Conv_one_image,axis=0)
            m_value = Conv_one_image[m_ind]
            indice = np.unravel_index(m_ind, Conv_size)
            c_ind = m_value/X_conv[indice[1],indice[1],X_conv_size[0]//2, X_conv_size[1]//2]
            if alpha is not None :
                c_ind = alpha*c_ind
            code[i_m, indice[1], indice[2], indice[3]] += c_ind
            I_conv_padded[i_m, :, indice[2]:indice[2] + X_conv_size[0], indice[3]:indice[3] + X_conv_size[1]]+= -c_ind * X_conv[indice[1], :, :, :]
            Conv_one_image = I_conv_padded[i_m, :, X_conv_size[0]//2:-(X_conv_size[0]//2), X_conv_size[1]//2:-(X_conv_size[1]//2)].reshape(-1)
            activation[indice[1]]+=1
            how[indice[1]]+=c_ind
            if when is not None :
                when[indice[1],i_l0] += 1
            if train == True :
                residual_image[i_m, :, indice[2]*stride:indice[2]*stride + dico_shape[1], indice[3]*stride:indice[3]*stride + dico_shape[2]] -= c_ind * dico[indice[1], :, :, :]
                where[indice[1],indice[2],indice[3]]+=1
    activation[activation==0]=1
    if train == True:
        res = torch.mean(torch.norm(torch.FloatTensor(residual_image).view(nb_image, -1),p=2,dim=1))
        to_return = (residual_image, code, res, activation, where, when, how)
    else :
        to_return = (code,activation)
    return to_return

def learn(code,dictionary,residual,eta):
    nb_dico = dictionary.size()[0]
    dico_size = (dictionary.size()[1],dictionary.size()[2],dictionary.size()[3])
    for idx_dico in range(nb_dico):
        #print(code)
        mask = code[:,idx_dico,:,:]>0

        loc_image,loc_line,loc_col = np.where(mask)
        #print(loc_image,loc_line,loc_col)
        if len(loc_image) != 0:
            patches = np.zeros((len(loc_image),dico_size[0],dico_size[1],dico_size[2]))
            act_c = code[:,idx_dico,:,:][mask]
            #print(act_c)
            for idx in range(len(loc_image)) :
                patches[idx,:,:,:] = residual[loc_image[idx],:,loc_line[idx]:loc_line[idx]+dico_size[1],loc_col[idx]:loc_col[idx]+dico_size[2]]
            to_add = np.mean(patches,axis = 0)
            dictionary[idx_dico,:,:,:].add_(eta*torch.FloatTensor(to_add))
    dictionary = Normalize(dictionary)
    return dictionary, patches#,to_add
'''
def learn2(code,dictionary,residual,eta):
    nb_dico = dictionary.size()[0]
    dico_size = (dictionary.size()[1],dictionary.size()[2],dictionary.size()[3])
    for idx_dico in range(nb_dico):
        #print(code)
        mask = code[:,idx_dico,:,:]>0

        loc_image,loc_line,loc_col = np.where(mask)
        if len(loc_image) != 0:
            patches = np.zeros((dico_size[0],dico_size[1],dico_size[2]))
            act_c = code[:,idx_dico,:,:][mask]
            for idx in range(len(loc_image)) :
                patches[:,:,:] += residual[loc_image[idx],:,loc_line[idx]:loc_line[idx]+dico_size[1],loc_col[idx]:loc_col[idx]+dico_size[2]]
                #to_add = np.mean(patches,axis = 0)
            patches[:,:,:]/
            #to_add = patches/len(loc_image)
            dictionary[idx_dico,:,:,:].add_(eta*torch.FloatTensor(to_add))
    dictionary = Normalize(dictionary)
    return dictionary
'''
