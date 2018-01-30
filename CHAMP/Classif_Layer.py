import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Classif_Layer(nn.Module):

    def __init__(self, nb_dico, size_image, nb_categories, verbose=0,GPU=False):
        super(Classif_Layer, self).__init__()
        self.nb_dico = nb_dico
        self.size_image = size_image
        self.type = 'Classification'
        self.verbose = verbose
        self.fc1 = nn.Linear(self.nb_dico*self.size_image[0]*self.size_image[1], nb_categories)
        self.GPU=GPU


    def forward(self, x):
        x = x.view(-1,self.nb_dico*self.size_image[0]*self.size_image[1])
        x = F.softmax(self.fc1(x),dim=1)
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
        for epoch in range(nb_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            #data,target =
            for i, each_batch in enumerate(data_train_loader[0]):
                if self.GPU:
                    data, target = each_batch.cuda(), data_train_loader[1][i,:].cuda()
                else :
                    data, target = each_batch, data_train_loader[1][i,:]
                #data, target = each_batch
                #target = data_train_loader[1][i,:]
                batch_size = data.size()[0]
                #input_image = input_coding[i*batch_size:(i+1)*batch_size,:,:,:]
                #each_label = label[i*batch_size:(i+1)*batch_size]

                # wrap them in Variable
                if type(data) is torch.sparse.FloatTensor:
                    inputs, labels = Variable(data.to_dense()), Variable(target)
                else :
                    inputs, labels = Variable(data.contiguous()), Variable(target)

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
                if self.GPU :
                    if type(data) is torch.sparse.FloatTensor:
                        output_testing = self.forward(Variable(data_test_loader[0][0].to_dense().cuda()))
                    else :
                        output_testing = self.forward(Variable(data_test_loader[0][0].contiguous().cuda()))
                else :
                    if type(data) is torch.sparse.FloatTensor:
                        output_testing = self.forward(Variable(data_test_loader[0][0].to_dense()))
                    else :
                        output_testing = self.forward(Variable(data_test_loader[0][0].contiguous()))
                _, predicted = torch.max(output_testing.data, 1)
                #total += output_testing[0][1].size(0)
                correct = (predicted == data_test_loader[1][0,:]).sum()
                accuracy = correct/data_test_loader[1].size()[1]
                if self.verbose != 0:
                    print('accuracy : {0:.2f} %'.format(accuracy*100))
                self.accuracy_list.append(accuracy)
            if self.verbose != 0:
                print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss/batch_size))
                    #running_loss = 0.0
        return self
