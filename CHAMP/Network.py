

class Network(object):
    def __init__(self, Layers, verbose=0):
        self.Layers = Layers
        self.verbose = verbose

    def TrainNetwork(self,training_set, testing_set, eta=0.05 , eta_homeo=0.01,\
                    epoch_unsupervised=2, epoch_cl=100, batch_size_cl=300,lr=5e-3):
        #event_i = event
        idx_Layer = 0
        LayerList = list()
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'Unsupervised':
                 dico = each_Layer.TrainLayer(training_set,eta=eta, eta_homeo=eta_homeo,nb_epoch=epoch_unsupervised)
                 training_set = (training_set[0].view(60000//batch_size_cl,batch_size_cl,1,28,28),training_set[1].view(60000//batch_size_cl,batch_size_cl))
                 output_training = each_Layer.RunLayer(training_set,dico)
                 output_testing = each_Layer.RunLayer(testing_set,dico)
            elif each_Layer.type == 'Classification':
                net_trained = each_Layer.train_classif(output_training, nb_epoch=epoch_cl, \
                        data_test_loader = output_testing,lr=lr)
        return dico
