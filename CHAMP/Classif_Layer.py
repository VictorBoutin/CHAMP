import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Classif_Layer(nn.Module):

    def __init__(self, nb_dico, size_image,verbose):
        super(Classif_Layer, self).__init__()
        self.nb_dico = nb_dico
        self.size_image = size_image
        self.type = 'Classification'
        self.verbose = verbose
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(self.nb_dico*self.size_image[0]*self.size_image[1], M)
        #self.fc2 = nn.Linear(M, 10)
        self.fc1 = nn.Linear(self.nb_dico*self.size_image[0]*self.size_image[1], 10)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1,self.nb_dico*self.size_image[0]*self.size_image[1])
        x = F.softmax(self.fc1(x))
        #x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train_classif(self, data_train_loader, nb_epoch=5, data_test_loader=None, lr=0.1):

        criterion = nn.CrossEntropyLoss()
        self.res = list()
        self.accuracy_list = list()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        #nb_batch = input_coding.size()[0]//batch_size
        for epoch in range(nb_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, each_batch in enumerate(data_train_loader):
                data, target = each_batch
                #data, target = each_batch
                #target = data_train_loader[1][i,:]
                batch_size = data.size()[0]
                #input_image = input_coding[i*batch_size:(i+1)*batch_size,:,:,:]
                #each_label = label[i*batch_size:(i+1)*batch_size]

                # wrap them in Variable
                if type(data) is torch.sparse.FloatTensor:
                    inputs, labels = Variable(data.to_dense()), Variable(target)
                else :
                    inputs, labels = Variable(data.data), Variable(target)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

                #if i % 10 == 9:
                #    self.res.append(running_loss/ (i*batch_size))
            self.res.append(running_loss/batch_size)
            if data_test_loader is not None:
                if type(data) is torch.sparse.FloatTensor:
                    output_testing = self.forward(Variable(data_test_loader[0][0].to_dense()))
                else :
                    output_testing = self.forward(Variable(data_test_loader[0][0].data))
                _, predicted = torch.max(output_testing.data, 1)
                #total += output_testing[0][1].size(0)
                correct = (predicted == data_test_loader[0][1]).sum()
                accuracy = correct/data_test_loader[0][1].size(0)
                if self.verbose != 0:
                    print('accuracy : {0:.2f} %'.format(accuracy*100))
                self.accuracy_list.append(accuracy)
            if self.verbose != 0:
                print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss/batch_size))
                    #running_loss = 0.0
        return self
