import torch
from PIL import Image
from random import shuffle
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from CHAMP.DataTools import Normalize, ContrastNormalized


def LoadData(name, data_path, decorrelate=False, avg_size=(5, 5), Grayscale=True, resize=None, GPU=False, download=False):
    Composition = list()
    if Grayscale == True:
        Composition.append(transforms.Grayscale())
    if resize is not None:
        Composition.append(transforms.Resize(resize))

    Composition.append(transforms.ToTensor())
    # Composition.append(transforms.Normalize((0,),(1,)))

    transform = transforms.Compose(Composition)
    if name == 'MNIST':
        train_set = torchvision.datasets.MNIST(
            root=data_path, train=True, transform=transform, download=download)
        train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=60000, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.MNIST(
            root=data_path, train=False, transform=transform, download=download)
        test_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=10000, shuffle=False, num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),
                         data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),
                        data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'CIFAR':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, transform=transform, download=download)
        train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=50000, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(
            root=data_path, train=False, transform=transform, download=download)
        test_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=10000, shuffle=False, num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),
                         data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),
                        data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'STL10':
        train_set = torchvision.datasets.STL10(
            root=data_path, split='train', transform=transform, download=download)
        train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=50000, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.STL10(
            root=data_path, split='test', transform=transform, download=download)
        test_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=10000, shuffle=False, num_workers=2)
        data_training = list(train_data_loader)
        data_training = (data_training[0][0].unsqueeze(0).contiguous(),
                         data_training[0][1].unsqueeze(0).contiguous())
        data_testing = list(test_data_loader)
        data_testing = (data_testing[0][0].unsqueeze(0).contiguous(),
                        data_testing[0][1].unsqueeze(0).contiguous())
    if name == 'Face':
        if resize is None:
            resize = (65, 65)
        data_training = LoadFaceDB(data_path, size=resize, to_shuffle=True)
        data_testing = (data_training[0].clone(), data_training[1].clone())

    if GPU == True:
        data_training = (data_training[0].cuda(), data_training[1].cuda())
        data_testing = (data_testing[0].cuda(), data_testing[1].cuda())
    if decorrelate == True:
        data_training = ContrastNormalized(data_training, avg_size=avg_size, GPU=GPU)
        data_testing = ContrastNormalized(data_testing, avg_size=avg_size, GPU=GPU)

    return (data_training, data_testing)
    # return (train_data_loader,test_data_loader)


def LoadFaceDB(path, size=(68, 68), to_shuffle=True):
    file_list = list()
    batch_size = 400
    if size is None:
        size = (112, 92)
    tensor_image = torch.FloatTensor(1, batch_size, 1, size[0], size[1])
    tensor_label = torch.LongTensor(1, batch_size)
    label = 0
    for each_dir in os.listdir(path):
        if each_dir != '.DS_Store':
            try:
                for each_file in os.listdir(os.path.join(path, str(each_dir))):
                    tot_dir = os.path.join(str(path), str(each_dir), str(each_file))
                    file_list.append((tot_dir, label))
                label += 1
            except:
                pass
    if to_shuffle == True:
        shuffle(file_list)
    idx = 0
    for j in range(batch_size):
        image = Image.open(file_list[idx][0])
        image = image.resize(size, Image.ANTIALIAS)
        tensor_image[0, j, 0, :, :] = torch.FloatTensor(np.array(image).astype(float))
        tensor_label[0, j] = file_list[idx][1]
        idx += 1
    to_output = (tensor_image.float(), tensor_label)
    return to_output


def LoadCaltech101(path, size=(100, 100), to_shuffle=True, test_per_category=5, item=None):
    file_list_training = list()
    file_list_testing = list()
    #training_data = torch.FloatTensor(1,24,1,size[0],size[1])
    #training_label = torch.FloatTensor(1,24,1,size[0],size[1])
    #testing_data = torch.FloatTensor(1,nb_training,1,size[0],size[1])
    #testing_label = torch.FloatTensor(1,nb_training,1,size[0],size[1])
    if item is None:
        all_dir = os.listdir(path)
    else:
        all_dir = item
    label = 0
    for each_dir in all_dir:
        if each_dir != '.DS_Store':
            for each_file in os.listdir(os.path.join(path, str(each_dir)))[0:-test_per_category]:
                tot_dir = os.path.join(str(path), str(each_dir), str(each_file))
                file_list_training.append((tot_dir, label))
            for each_file in os.listdir(os.path.join(path, str(each_dir)))[-test_per_category:]:
                tot_dir = os.path.join(str(path), str(each_dir), str(each_file))
                file_list_testing.append((tot_dir, label))
            label += 1
    if to_shuffle == True:
        shuffle(file_list_training)
        shuffle(file_list_testing)

    idx = 0
    training_data = torch.FloatTensor(1, len(file_list_training), 1, size[0], size[1])
    training_label = torch.FloatTensor(1, len(file_list_training))
    testing_data = torch.FloatTensor(1, len(file_list_testing), 1, size[0], size[1])
    testing_label = torch.FloatTensor(1, len(file_list_testing))
    for idx, data_id in enumerate(file_list_training):
        image = Image.open(data_id[0]).convert('L')
        image = image.resize(size, Image.ANTIALIAS)
        training_data[0, idx, 0, :, :] = torch.FloatTensor(np.array(image).astype(float))
        training_label[0, idx] = data_id[1]
    for idx, data_id in enumerate(file_list_testing):
        image = Image.open(data_id[0]).convert('L')
        image = image.resize(size, Image.ANTIALIAS)
        testing_data[0, idx, 0, :, :] = torch.FloatTensor(np.array(image).astype(float))
        testing_label[0, idx] = data_id[1]
    return (training_data, training_label), (testing_data, testing_label)
