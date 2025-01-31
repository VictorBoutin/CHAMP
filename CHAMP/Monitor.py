import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def DisplayDico(dico, figsize=8, fig=None, ax=None):
    subplotpars = matplotlib.figure.SubplotParams(
        left=0.1, right=1., bottom=0., top=1., wspace=0.05, hspace=0.05)
    try:
        if dico.type() in ['torch.FloatTensor', 'torch.LongTensor']:
            dico = dico.numpy()
    except:
        pass
    dico_size = tuple(dico.shape[2:])
    nb_dico = dico.shape[0]
    nb_channel = dico.shape[1]
    if nb_channel != 1:
        raise NameError('Do Not Support Multi Channel Plot')
    out_dico = dico.reshape(-1, dico_size[0], dico_size[1])
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize/nb_dico), subplotpars=subplotpars)
        #fig = plt.figure(figsize=(15, 15))
    if ax is None:
        # ax = fig.add_subplot(111)
        ax = []
        fig, ax = plt.subplots(1, nb_dico, figsize=(figsize, figsize/nb_dico))#, subplotpars=subplotpars)


    for i, each_filter in enumerate(out_dico):
        # print(each_filter.shape)
        #each_filter /= np.abs(each_filter).max()
        #cmax = np   .max(each_filter)
        #cmin = np.min(each_filter)
        #ax_ = fig.add_subplot(nb_dico//10+1, 10, i+1)
        cmax = np.abs(each_filter.max())
        ax[i].imshow(each_filter, cmap='gray', vmin=-cmax, vmax=cmax)
        ax[i].set_xticks(())
        ax[i].set_yticks(())

    return fig, ax


def DisplayWhere(where, figsize=8, fig=None, ax=None):
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05)
    try:
        if where.type() in ['torch.FloatTensor', 'torch.LongTensor']:
            dico = where.numpy()
    except:
        pass
    where_size = tuple(where.shape[-2:])
    nb_map = where.shape[0]
    fig = plt.figure(figsize=(figsize, (nb_map//figsize+1)), subplotpars=subplotpars)
    for i in range(nb_map):
        ax = fig.add_subplot(nb_map//10+1, 10, i+1)
        cmax = np.abs(where[i, :, :].max())
        ax.imshow(where[i, :, :], vmin=0, vmax=cmax)
        ax.set_xticks(())
        ax.set_yticks(())
    return fig, ax

def DisplayConvergenceCHAMP(ClusterLayer, to_display=['error'], figsize=4, eta=None, eta_homeo=None, color='black', fig=None, ax=None):
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
    if fig is None:
        subplotpars = matplotlib.figure.SubplotParams(
            left=0.15, right=1., bottom=0.15, top=1., wspace=0.1, hspace=0.2)
        fig = plt.figure(figsize=(figsize, figsize/2.6180), subplotpars=subplotpars)
        #fig = plt.figure(figsize=(15, 15))
    if ax is None:
        ax = fig.add_subplot(111)

    location = 1

    for idx, each_Layer in enumerate(ClusterLayer):
        nb_dico = each_Layer.nb_dico
        for idx_type, each_type in enumerate(to_display):
            each_type = str(each_type)
            #ax = fig.add_subplot(len(ClusterLayer), len(to_display), location)
            #max_x = each_Layer.record[each_type].shape[0]*each_Layer.record_each
            # ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
            if each_type == 'error':
                to_plot = plt.plot(each_Layer.res_list)
                if (eta is not None) and (eta_homeo is not None):
                    ax.set_title('Convergence Layer {0} with eta : {1} and eta_homeo : {2}'.format(
                        idx+1, eta, eta_homeo), fontsize=8)
                else:
                    ax.set_title('Convergence Layer {0}'.format(idx+1), fontsize=8)
            elif each_type == 'histo':
                to_plot = plt.bar(np.arange(nb_dico), each_Layer.activation,
                                  width=np.diff(np.arange(nb_dico+1)), fc=color, align="edge")
                # if (eta is not None) and (eta_homeo is not None):
                #     ax.set_title('Histogram of activation at Layer {0} with eta : {1} and eta_homeo : {2}'.format(
                #         idx+1, eta, eta_homeo), fontsize=8)
                # else:
                #     ax.set_title('Histogram of activation at Layer {0}'.format(idx+1), fontsize=8)
            location += 1
    return fig, ax


def DisplayConvergenceClassif(ClusterLayer, to_display=['error'], fig=None, ax=None):
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    fig = plt.figure(figsize=(10, 2), subplotpars=subplotpars)
    location = 1
    for idx_type, each_type in enumerate(to_display):
        each_type = str(each_type)
        ax = fig.add_subplot(1, len(to_display), location)
        #max_x = each_Layer.record[each_type].shape[0]*each_Layer.record_each
        # ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
        if each_type == 'error':
            to_plot = plt.plot(ClusterLayer.loss_list)
            ax.set_title('Classification Layer : {0}'.format(each_type), fontsize=8)
        if each_type == 'accu':
            to_plot = plt.plot(ClusterLayer.accuracy_list)
            ax.set_title('Classification Layer : {0}'.format(each_type), fontsize=8)

        location += 1
    return fig, ax


def DisplayCV(ClusterLayer, to_display=['error'], title=None, fig=None, ax=None):
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.1, hspace=0.2)
    fig = plt.figure(figsize=(10, 2), subplotpars=subplotpars)
    location = 1
    if title is not None:
        title = title
    else:
        title == ''
    for idx_type, each_type in enumerate(to_display):
        each_type = str(each_type)
        ax = fig.add_subplot(1, len(to_display), location)
        #max_x = each_Layer.record[each_type].shape[0]*each_Layer.record_each
        # ax.set_xticks([0,roundup(max_x/3,each_Layer.record_each),roundup(2*max_x/3,each_Layer.record_each)])
        if each_type == 'error':
            to_plot = plt.plot(ClusterLayer[0])
            ax.set_title(str(title) + ' Classification Layer : {0}'.format(each_type), fontsize=8)
        if each_type == 'accu':
            to_plot = plt.plot(ClusterLayer[1])
            ax.set_title(str(title) + ' Classification Layer : {0}'.format(each_type), fontsize=8)

        location += 1
    return fig, ax
