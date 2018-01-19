import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def DisplayDico(dico):
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05)
    dico_size = tuple(dico.size()[2:])
    nb_dico = dico.size()[0]
    out_dico = dico.data.numpy().reshape(-1,dico_size[0],dico_size[1])
    fig = plt.figure(figsize=(10,(nb_dico//10+1)), subplotpars=subplotpars)
    for i, each_filter in enumerate(out_dico):
        ax = fig.add_subplot(nb_dico//10+1,10,i+1)
        ax.imshow(each_filter, cmap='gray')
        ax.set_xticks(())
        ax.set_yticks(())

def DisplayConvergenceCHAMP(ClusterLayer,to_display=['error'],eta=None,eta_homeo=None):
    '''
    Function to display the monitored variable during the training
    INPUT :
        + ClusterLayer : (<list>) of (<object Cluster>) holding the cluster object for each layer
        + to_display : (<list>) of (<string>) to indicate which monitoring variable to display. 'error' will plot the L2 error,
            'histo' will the activated cluster histogram
        + eta : (<float>) to add eta in the name of the graph
        + eta_homeo : (<float>) to add eta_homeo in the name of the graph
    '''
    if type(ClusterLayer) is not list:
        ClusterLayer = [ClusterLayer]
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    fig = plt.figure(figsize=(10,2*len(ClusterLayer)),subplotpars=subplotpars)
    location = 1

    for idx,each_Layer in enumerate(ClusterLayer) :
        nb_dico = each_Layer.nb_dico
        for idx_type, each_type in enumerate(to_display):
            each_type = str(each_type)
            ax = fig.add_subplot(len(ClusterLayer),len(to_display),location)
            #max_x = each_Layer.record[each_type].shape[0]*each_Layer.record_each
            #ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
            if each_type=='error':
                to_plot = plt.plot(each_Layer.res_list)
                if (eta is not None) and (eta_homeo is not None) :
                    ax.set_title('Convergence Layer {0} with eta : {1} and eta_homeo : {2}'.format(idx+1,eta,eta_homeo),fontsize= 8)
                else :
                    ax.set_title('Convergence Layer {0}'.format(idx+1),fontsize= 8)
            elif each_type=='histo':

                to_plot = plt.bar(np.arange(nb_dico),each_Layer.activation,\
                width=np.diff(np.arange(nb_dico+1)), ec="k", align="edge")
                if (eta is not None) and (eta_homeo is not None) :
                    ax.set_title('Histogram of activation at Layer {0} with eta : {1} and eta_homeo : {2}'.format(idx+1,eta,eta_homeo),fontsize= 8)
                else :
                    ax.set_title('Histogram of activation at Layer {0}'.format(idx+1),fontsize= 8)
            location +=1

def DisplayConvergenceClassif(ClusterLayer,to_display=['error']):
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    fig = plt.figure(figsize=(10,2),subplotpars=subplotpars)
    location = 1
    for idx_type, each_type in enumerate(to_display):
        each_type = str(each_type)
        ax = fig.add_subplot(1,len(to_display),location)
        #max_x = each_Layer.record[each_type].shape[0]*each_Layer.record_each
        #ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
        if each_type=='error':
            to_plot = plt.plot(ClusterLayer.res)
            ax.set_title('Classification Layer : {0}'.format(each_type),fontsize= 8)
        if each_type=='accu':
            to_plot = plt.plot(ClusterLayer.accuracy_list)
            ax.set_title('Classification Layer : {0}'.format(each_type),fontsize= 8)


        location +=1
