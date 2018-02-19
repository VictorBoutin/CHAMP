import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Classif_Layer(nn.Module):

    def __init__(self, nb_dico, size_image, nb_categories, verbose=0,GPU=False,loss='CE',init='random'):
        super(Classif_Layer, self).__init__()
        self.nb_dico = nb_dico
        self.size_image = size_image
        self.type = 'Classification'
        self.verbose = verbose
        self.norma = nn.BatchNorm2d(nb_dico)
        self.fc1 = nn.Linear(self.nb_dico*self.size_image[0]*self.size_image[1], nb_categories)

        if init == 'zero':
            self.fc1.weight.data.fill_(0)
        self.GPU=GPU

        if loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss == 'NNL':
            self.criterion = nn.NLLLoss()
        if self.GPU :
            self.cuda()
    def forward(self, x):
        x = self.norma(x)
        x = x.view(-1,self.nb_dico*self.size_image[0]*self.size_image[1])
        x = F.softmax(self.fc1(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train1epoch(self,data_train_loader,lr=0.1,momentum=0.9, weight_decay=0,op='SGD'):
        self.train()
        #criterion = nn.CrossEntropyLoss()
        if op == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
        elif op == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif op == 'AdamSparse':
            optimizer = optim.SparseAdam(self.parameters(), lr=lr)
        for i, each_batch in enumerate(data_train_loader[0]):
            if self.GPU:
                data, target = each_batch.cuda(), data_train_loader[1][i,:].cuda()
            else :
                data, target = each_batch, data_train_loader[1][i,:]
            data,target = Variable(data), Variable(target)
            optimizer.zero_grad()

            outputs = self(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            optimizer.step()
        return loss.data[0]

    def TrainClassifier(self,data_train_loader, nb_epoch=5, data_test_loader=None, lr=0.1,momentum=0.5, weight_decay=0,op='SGD'):
        self.loss_list = []
        self.accuracy_list = []
        for epoch in range(nb_epoch):
            loss = self.train1epoch(data_train_loader,lr=lr,momentum=momentum,weight_decay=weight_decay,op=op)
            if data_test_loader is None:
                data_test_loader = data_train_loader
            accuracy = self.test(data_test_loader)
            self.accuracy_list.append(accuracy)
            self.loss_list.append(loss)
            if self.verbose != 0:
                if ((epoch + 1) % (nb_epoch // self.verbose)) == 0:
                    print('accuracy : {0:.2f} %, loss : {1:4f}'.format(accuracy,loss))

    def test(self, data_test_loader):
        self.eval()
        if self.GPU == True:
            data,target = data_test_loader[0][0].cuda(),data_test_loader[1][0].cuda()
        else :
            data,target = data_test_loader[0][0],data_test_loader[1][0]
        data,target = Variable(data),Variable(target)
        output = self.forward(data)
        prediction = output.data.max(1,keepdim=True)[1]
        correct = prediction.eq(target.data.view_as(prediction)).cpu().sum()
        accuracy = 100 * correct / data.size()[0]
        return accuracy

class Classif_Layer_1CONV(nn.Module):
    def __init__(self, nb_dico,size_image, nb_categories, depth_L1=10, verbose=0,GPU=False,loss='CE',init='random',kernel_size=5):
        super(Classif_Layer_1CONV, self).__init__()
        self.nb_dico = nb_dico
        self.size_image = size_image
        self.type = 'Classification'
        self.verbose = verbose
        self.depth_L1 = depth_L1
        self.norma = nn.BatchNorm2d(nb_dico)
        self.conv1 = nn.Conv2d(self.nb_dico, self.depth_L1, kernel_size=kernel_size)
        self.size_conv1 = size_image[0]-kernel_size+1
        self.fc1 = nn.Linear(self.depth_L1*self.size_conv1*self.size_conv1, nb_categories)

        if init == 'zero':
            self.fc1.weight.data.fill_(0)
        self.GPU=GPU

        if loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss == 'NNL':
            self.criterion = nn.NLLLoss()
        if self.GPU :
            self.cuda()

    def forward(self, x):
        x = self.norma(x)
        x = self.conv1(x)
        x = x.view(-1,self.depth_L1*self.size_conv1*self.size_conv1)
        x = F.softmax(self.fc1(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train1epoch(self,data_train_loader,lr=0.1,momentum=0.9,op='SGD'):
        self.train()
        #criterion = nn.CrossEntropyLoss()
        if op == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif op == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif op == 'AdamSparse':
            optimizer = optim.SparseAdam(self.parameters(), lr=lr)
        for i, each_batch in enumerate(data_train_loader[0]):
            if self.GPU:
                data, target = each_batch.cuda(), data_train_loader[1][i,:].cuda()
            else :
                data, target = each_batch, data_train_loader[1][i,:]
            data,target = Variable(data), Variable(target)
            optimizer.zero_grad()

            outputs = self(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            optimizer.step()
        return loss.data[0]

    def TrainClassifier(self,data_train_loader, nb_epoch=5, data_test_loader=None, lr=0.1,momentum=0.5,op='SGD'):
        self.loss_list = []
        self.accuracy_list = []
        for epoch in range(nb_epoch):
            loss = self.train1epoch(data_train_loader,lr=lr,momentum=momentum,op=op)
            if data_test_loader is None:
                data_test_loader = data_train_loader
            accuracy = self.test(data_test_loader)
            self.accuracy_list.append(accuracy)
            self.loss_list.append(loss)
            if self.verbose != 0:
                if ((epoch + 1) % (nb_epoch // self.verbose)) == 0:
                    print('accuracy : {0:.2f} %, loss : {1:4f}'.format(accuracy,loss))

    def test(self, data_test_loader):
        self.eval()
        if self.GPU == True:
            data,target = data_test_loader[0][0].cuda(),data_test_loader[1][0].cuda()
        else :
            data,target = data_test_loader[0][0],data_test_loader[1][0]
        data,target = Variable(data),Variable(target)
        output = self.forward(data)
        prediction = output.data.max(1,keepdim=True)[1]
        correct = prediction.eq(target.data.view_as(prediction)).cpu().sum()
        accuracy = 100 * correct / data.size()[0]
        return accuracy

class Classif_Layer_Relu(nn.Module):

    def __init__(self, nb_dico, size_image, nb_categories, verbose=0,GPU=False,loss='CE',init='random'):
        super(Classif_Layer_Relu, self).__init__()
        self.nb_dico = nb_dico
        self.size_image = size_image
        self.type = 'Classification'
        self.verbose = verbose
        self.norma = nn.BatchNorm2d(nb_dico)
        self.fc1 = nn.Linear(self.nb_dico*self.size_image[0]*self.size_image[1], nb_categories)

        if init == 'zero':
            self.fc1.weight.data.fill_(0)
        self.GPU=GPU

        if loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss == 'NNL':
            self.criterion = nn.NLLLoss()
        if self.GPU :
            self.cuda()
    def forward(self, x):
        x = F.relu(x)
        x = self.norma(x)
        x = x.view(-1,self.nb_dico*self.size_image[0]*self.size_image[1])
        x = F.softmax(self.fc1(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train1epoch(self,data_train_loader,lr=0.1,momentum=0.9, weight_decay=0,op='SGD'):
        self.train()
        #criterion = nn.CrossEntropyLoss()
        if op == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
        elif op == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif op == 'AdamSparse':
            optimizer = optim.SparseAdam(self.parameters(), lr=lr)
        for i, each_batch in enumerate(data_train_loader[0]):
            if self.GPU:
                data, target = each_batch.cuda(), data_train_loader[1][i,:].cuda()
            else :
                data, target = each_batch, data_train_loader[1][i,:]
            data,target = Variable(data), Variable(target)
            optimizer.zero_grad()

            outputs = self(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            optimizer.step()
        return loss.data[0]

    def TrainClassifier(self,data_train_loader, nb_epoch=5, data_test_loader=None, lr=0.1,momentum=0.5, weight_decay=0,op='SGD'):
        self.loss_list = []
        self.accuracy_list = []
        for epoch in range(nb_epoch):
            loss = self.train1epoch(data_train_loader,lr=lr,momentum=momentum,weight_decay=weight_decay,op=op)
            if data_test_loader is None:
                data_test_loader = data_train_loader
            accuracy = self.test(data_test_loader)
            self.accuracy_list.append(accuracy)
            self.loss_list.append(loss)
            if self.verbose != 0:
                if ((epoch + 1) % (nb_epoch // self.verbose)) == 0:
                    print('accuracy : {0:.2f} %, loss : {1:4f}'.format(accuracy,loss))

    def test(self, data_test_loader):
        self.eval()
        if self.GPU == True:
            data,target = data_test_loader[0][0].cuda(),data_test_loader[1][0].cuda()
        else :
            data,target = data_test_loader[0][0],data_test_loader[1][0]
        data,target = Variable(data),Variable(target)
        output = self.forward(data)
        prediction = output.data.max(1,keepdim=True)[1]
        correct = prediction.eq(target.data.view_as(prediction)).cpu().sum()
        accuracy = 100 * correct / data.size()[0]
        return accuracy
