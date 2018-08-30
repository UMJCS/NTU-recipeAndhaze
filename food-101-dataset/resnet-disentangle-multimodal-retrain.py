import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time

import torch
import torch.utils.data
import torch.nn.parallel as para
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = os.getcwd()+'/'#'/home/lily/Desktop/food/' #/home/FoodRecog/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder='food-101/images/'#'ready_chinese_food'#scaled_images ready_chinese_food
image_path = os.path.join(root_path,image_folder,'/')

file_path = root_path#os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'foodtype_vectors.txt')#os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'train.txt')
validation_data_path = os.path.join(file_path, 'val.txt')
test_data_path = os.path.join(file_path, 'test.txt')

result_path = root_path + 'results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

INGREDIENTS = []
INGREfile = open('foodtype_vectors.txt')
INGREdata = INGREfile.readlines()

for i in range(0,len(INGREdata),7):
    lines = INGREdata[i:i+7]
    outline = []
    for line in lines:
        for ele in line:
            if ele.isdigit():
                outline.append(int(ele))
    INGREDIENTS.append(outline)

# —Create dataset———————————————————————————————————————————————————————————————————————————————————————————————————————
def default_loader(path):
    img_path = root_path + image_folder + path

    jpgfile = Image.open(img_path).convert('RGB')

    return jpgfile


class FoodData(torch.utils.data.Dataset):
    def __init__(self, train_data=False, test_data=False, transform=None,
                 loader=default_loader):

        # load image paths / label file
        if train_data:
            with io.open(train_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            # # train_load = 'train_label.mat'
            # labels = matio.loadmat(train_load)
            file1 = io.open('train_label.txt', encoding='utf-8')
            labels = [int(i[:-1]) for i in file1.readlines()]

            # matio.savemat(path,['train_label',label])
            # ingredients = matio.loadmat(file_path + 'ingredient_train_feature.mat')['ingredient_train_feature'][0:66071,:]

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            file4 = io.open('val_label.txt', encoding='utf-8')
            labels1 = [int(i[:-1]) for i in file4.readlines()]
            # labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']
            # ingredients1 = matio.loadmat(file_path + 'ingredient_validation_feature.mat')['ingredient_validation_feature'][0:11016,:]
            #
            path_to_images = path_to_images + path_to_images1
            labels.extend(labels1)
            # labels = np.concatenate([labels,labels1],1)[0,:]
            # #ingredients = np.concatenate([ingredients, ingredients1], 0)
        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')

            file2 = io.open('test_label.txt', encoding='utf-8')
            labels = [int(i[:-1]) for i in file2.readlines()]

            # ingredients = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature'][0:33154,:]
        else:
            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            file3 = io.open('val_label.txt', encoding='utf-8')
            labels = [int(i[:-1]) for i in file3.readlines()]

        #ingredients = ingredients.astype(np.float32)
        self.path_to_images = path_to_images
        self.labels = labels
        #self.ingredients = ingredients
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        # get label
        label = self.labels[index]
        # get ingredients 353-D vector
        #ingredient = self.ingredients[index, :]
        return img, label#, ingredient



    def __len__(self):
        return len(self.path_to_images)


# —Model———————————————————————————————————————————————————————————————————————————————————————————————————————
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeBasicBlock, self).__init__()
        self.conv1 = Deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = Deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Deconv_Bottleneck(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeBottleneck, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Deconv_Bottleneck(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.ConvTranspose2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=101):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 256-512
        self.avgpooling = nn.AvgPool2d(image_size[0] // (2 ** 5), stride=1)
        # get latent representation
        latent_len = 500
        self.latent = nn.Linear(512 * block.expansion, latent_len)

        # classifier_v
        self.classifier1 = nn.Linear(blk_len, num_classes)

        # define resnet decoder
        self.latent_re = nn.Linear(latent_len, 512 * block.expansion)
        self.layer5 = self._make_Delayer(DeBottleneck, 256, layers[3], stride=2)  # 512-256
        self.layer6 = self._make_Delayer(DeBottleneck, 128, layers[3], stride=2)  # 256-128
        self.layer7 = self._make_Delayer(DeBottleneck, 64, layers[3], stride=2)  # 128-64
        self.layer8 = self._make_Delayer(DeBottleneck, 64, layers[3], stride=1)  # 64-64
        # self.deconv9 = nn.ConvTranspose2d(64* block.expansion, 3, kernel_size=7, stride=2, padding=3, output_padding=1,
        #                                  bias=False)
        self.deconv9 = nn.ConvTranspose2d(64 * block.expansion, 64, kernel_size=1, bias=False)
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.deconv10 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1,
                                           bias=False)
        self.sigmoid = nn.Sigmoid()

        # define ingredient encoder for 353 input features
        num_ingre_feature = 227
        self.nn1 = nn.Linear(num_ingre_feature, num_ingre_feature)
        self.nn2 = nn.Linear(num_ingre_feature, latent_len)
        # classifier_T
        # self.classifier2 = nn.Linear(blk_len, num_classes)
        # define ingredient decoder
        self.nn3 = nn.Linear(latent_len, num_ingre_feature)
        self.nn4 = nn.Linear(num_ingre_feature, num_ingre_feature)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_Delayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, output_padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encoder_V(self, x):  # convolve images
        x = self.conv1(x)  # ／2
        x = self.bn1(x)
        x = self.relu(x)
        [x, a] = self.maxpool(x)  # ／2
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)  # ／2
        # print(x.shape)
        x = self.layer3(x)  # ／2
        # print(x.shape)
        x = self.layer4(x)  # ／2
        # print(x.shape)
        x = self.avgpooling(x)  # (1x1)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x_latent = self.latent(x)

        return x_latent, a

    def decoder_V(self, x_latent, a):
        x = self.latent_re(x_latent)
        x = x.view(x.shape[0], 512 * 4, 1, 1)
        # print(x.shape)
        x = F.upsample(x, scale_factor=image_size[0] // (2 ** 5), mode='nearest')
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        x = self.layer6(x)
        # print(x.shape)
        x = self.layer7(x)
        # print(x.shape)
        x = self.layer8(x)
        # print(x.shape)
        x = self.deconv9(x)
        x = self.unmaxpool(x, a)
        # print(x.shape)
        x = self.deconv10(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x

    def forward(self, x, y):  # x:image y:ingredient

        x_latent, a = self.encoder_V(x)
        predicts_V = self.classifier1(x_latent[:, 0:blk_len])
        x = self.decoder_V(x_latent, a)

        y = self.relu(self.nn1(y))
        y_latent = self.nn2(y)
        predicts_T = self.classifier1(y_latent[:, 0:blk_len])
        y_re = self.relu(self.nn3(y_latent))
        y_re = self.nn4(y_re)

        return predicts_V, predicts_T, x, y_re, x_latent, y_latent


# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
#         model_dict = model.state_dict()
#
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#
#     return model

#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model
#
#

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model
#


# —Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
EPOCHS = 18
learning_rate = 5e-6
blk_len = 300

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 6, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

train_loader = torch.utils.data.DataLoader(
    FoodData(train_data=True, test_data=False,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    FoodData(train_data=False, test_data=True,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
#model = resnet50(pretrained=True)
model = torch.load('exp13_ingredient_1e-05_8.pth')
if CUDA:
	model = model.cuda()
	model = nn.DataParallel(model)
    
if CUDA==0:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # .module.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
if CUDA:
    criterion = criterion.cuda()


def loss_function(predicts_V, predicts_T, labels, data, ingredients, x, y, x_latent, y_latent):
    # normalization = nn.Softmax()
    CE_V = criterion(predicts_V, labels ) * 18
    CE_T = criterion(predicts_T, labels )
    RE_V = torch.sum((data - x) ** 2) * (1e-5)
    RE_T = torch.sum((ingredients - y) ** 2) * (1e-2)
    AE = torch.sum((x_latent[:, 0:blk_len] - y_latent[:, 0:blk_len]) ** 2) * (
        1e-3)  # error in aligning latent content features of x_latent and y_latent

    # CE_T = criterion(predicts_T, labels - 1)
    # RE_latent = torch.sum((image_latent-ingredient_latent)**2)
    # print('Loss ====> CE_V: {} | RE_latent: {} |'.format(CE_V,RE))
    # print('Loss ====> CE_V: {} | CE_T: {} | RE_latent: {} |'.format(CE_V, CE_T,RE_latent))
    return CE_V, CE_T, RE_V, RE_T, AE  # +CE_T#+0.01*RE_latent


def top_match(predicts, labels):
    sorted_predicts = predicts.cpu().data.numpy().argsort()
    top1_labels = sorted_predicts[:, -1:][:, 0]
    match = float(sum(top1_labels == (labels)))

    top5_labels = sorted_predicts[:, -5:]
    hit = 0
    for i in range(0, labels.size(0)):
        hit += (labels[i]) in top5_labels[i, :]

    return match, hit


def train(epoch):
    # toggle model to train mode
    print('Training starts..')
    model.train()
    train_loss = 0
    top1_accuracy_total_V = 0
    top5_accuracy_total_V = 0
    top1_accuracy_total_T = 0
    top5_accuracy_total_T = 0
    total_time = time.time()

    for batch_idx, (data, labels) in enumerate(
            train_loader):  # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if batch_idx == 1:
        # break
        # print('batch %',batch_idx)         #---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        #qqqqq = labels[0]
        ingredients = []
        for element in labels:
            #kkk = INGREDIENTS[int(element)]
            ingredients.append(INGREDIENTS[int(element)])
        ingredients = np.array(ingredients,dtype='uint8')
        ingredients = torch.from_numpy(ingredients).float()
        #kkkkk = type(ingredients)
        start_time = time.time()
        data = Variable(data)
        ingredients = Variable(ingredients)
        if CUDA:
            data = data.cuda()
            labels = labels.cuda()
            ingredients = ingredients.cuda()
        optimizer.zero_grad()

        # obtain output from model
        predicts_V, predicts_T, x, y, x_latent, y_latent = model(data, ingredients)

        # calculate scalar loss
        CE_V, CE_T, RE_V, RE_T, AE = loss_function(predicts_V, predicts_T, labels, data, ingredients, x, y, x_latent,
                                                   y_latent)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss = CE_V + CE_T + RE_V + RE_T + AE
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        # compute accuracy
        matches_V, hits_V = top_match(predicts_V, labels)
        matches_T, hits_T = top_match(predicts_T, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        top1_accuracy_total_T += matches_T
        top1_accuracy_cur_T = matches_T / float(labels.size(0))
        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        top5_accuracy_total_T += hits_T
        top5_accuracy_cur_T = hits_T / float(labels.size(0))

        if epoch == 1 and batch_idx == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | RE_T: {:.4f} | AE: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Top1_Accuracy_T:{} | Top5_Accuracy_T:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, RE_T.data, AE.data, top1_accuracy_cur_V,
                    top5_accuracy_cur_V, top1_accuracy_cur_T, top5_accuracy_cur_T,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | RE_T: {:.4f} | AE: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Top1_Accuracy_T:{} | Top5_Accuracy_T:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, RE_T.data, AE.data, top1_accuracy_cur_V,
                    top5_accuracy_cur_V, top1_accuracy_cur_T, top5_accuracy_cur_T,
                    round((time.time() - start_time) * LOG_INTERVAL, 4),
                    round((time.time() - total_time), 4)))

    # ---------------------------------------------------------------------------------------------------------------------------------
    # for effective code debugging
    # break
    # ---------------------------------------------------------------------------------------------------------------------------------
    print(
        '====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Average Top1_Accuracy_T:{} | Average Top5_Accuracy_T:{} | Time:{}'.format(
            epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset),
                   top5_accuracy_total_V / len(train_loader.dataset), top1_accuracy_total_T / len(train_loader.dataset),
                   top5_accuracy_total_T / len(train_loader.dataset), round((time.time() - total_time), 4)))
    if epoch!=0 and epoch%2==0:
        torch.save(model, 'exp14_retrain_ingredient_'+ str(learning_rate)+'_' + str(epoch) +'.pth')
        print("model saved")
    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        # print('write in-epoch loss at epoch {} | batch {}'.format(epoch,batch_idx))
        file.write('%f\n' % (train_loss / len(train_loader)))


def test(epoch):
    # toggle model to test / inference mode
    print('testing starts..')
    model.eval()
    top1_accuracy_total_V = 0
    top1_accuracy_total_T = 0
    top5_accuracy_total_V = 0
    top5_accuracy_total_T = 0
    total_time = time.time()

    # each data is of BATCH_SIZE (default 128) samples
    for test_batch_idx, (data, labels) in enumerate(test_loader):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if test_batch_idx == 1:
        # break
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------
        ingredients = []
        for element in labels:
            #kkk = INGREDIENTS[int(element)]
            ingredients.append(INGREDIENTS[int(element)])
        ingredients = np.array(ingredients,dtype='uint8')
        ingredients = torch.from_numpy(ingredients).float()
        start_time = time.time()

        # we're only going to infer, so no autograd at all required
        with torch.no_grad():
            data = Variable(data)
            ingredients = Variable(ingredients)

        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()
            ingredients = ingredients.cuda()
            labels = labels.cuda()

        predicts_V, predicts_T, x, _, _, _ = model(data, ingredients)

        # compute accuracy
        matches_V, hits_V = top_match(predicts_V, labels)
        matches_T, hits_T = top_match(predicts_T, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        top1_accuracy_total_T += matches_T
        top1_accuracy_cur_T = matches_T / float(labels.size(0))
        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        top5_accuracy_total_T += hits_T
        top5_accuracy_cur_T = hits_T / float(labels.size(0))

        print(
            'Testing batch: {} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Top1_Accuracy_T:{} | Top5_Accuracy_T:{} | Time:{} | Total_Time:{}'.format(
                test_batch_idx, top1_accuracy_cur_V, top5_accuracy_cur_V, top1_accuracy_cur_T, top5_accuracy_cur_T,
                round((time.time() - start_time), 4),
                round((time.time() - total_time), 4)))

    print(
        '====> Test set: Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Average Top1_Accuracy_T:{} | Average Top5_Accuracy_T:{} | Total Time:{}'.format(
            top1_accuracy_total_V / len(test_loader.dataset), top5_accuracy_total_V / len(test_loader.dataset),
            top1_accuracy_total_T / len(test_loader.dataset), top5_accuracy_total_T / len(test_loader.dataset),
            round((time.time() - total_time), 4)))

    # save testing performance per epoch
    with io.open(result_path + 'test_accuracy.txt', 'a', encoding='utf-8') as file:
        file.write('%f ' % (top1_accuracy_total_V / len(test_loader.dataset)))
        file.write('%f ' % (top5_accuracy_total_V / len(test_loader.dataset)))
        file.write('%f ' % (top1_accuracy_total_T / len(test_loader.dataset)))
        file.write('%f\n ' % (top5_accuracy_total_T / len(test_loader.dataset)))


def tester():
    # create reconstruction tensor
    comparison = torch.zeros(64, 3, 256, 256)
    if CUDA:
        comparison = comparison.cuda()
    trans = transforms.ToTensor()
    # process test images one by one
    for i in range(0, 32):
        # read the image tensor
        test_img = trans(Image.open(test_path + str(i) + '.jpg').convert('RGB')).view(1, 3, 256, 256)
        if CUDA:
            test_img = test_img.cuda()
        # obtain latent space
        x_latent, a = model.module.encoder_V(test_img)  # .module.encoder_V(test_img)
        # produce reconstructed img
        recon = model.module.decoder_V(x_latent, a)  # .module.decoder_V(x_latent, a)
        # save the original and reconstructed imgs to tensor
        # each row shows the 8 images
        # with right below them the reconstructed output
        row = (i + 1) // 8  # the row index for this img
        mod = (i + 1) % 8
        if not mod:
            row -= 1
            col = 8
        else:
            col = (i + 1) % 8

        row *= 2

        comparison[row * 8 + col - 1, :] = test_img
        comparison[(row + 1) * 8 + col - 1, :] = recon
        del test_img
        del recon

        # produce synthetic images with varied latent values
        # modify latent vector
        rng = np.arange(-5, 5 + (10 / 8), (10 / 8))  # set variable varies in [-3,3] with step 1
        num_features = len(rng)  # change the first seven features
        batch_image = batch_img_producer(x_latent, rng, num_features)
        # produce reconstructed images
        recon_img = model.module.decoder_V(batch_image, a.repeat(len(rng) * num_features, 1, 1,
                                                                 1))  # .module.decoder_V(batch_image, a.repeat(len(rng) * num_features, 1, 1, 1))
        del batch_image
        save_image(recon_img, result_path + 'synthetic_test_' + str(i) + '.jpg', nrow=len(rng))
        del recon_img

        # save results to the folder, note that should save to the shared folder in docker image, and view files in local folder
    print('Generating images..')
    save_image(comparison.data, result_path + 'reconstruction_test.jpg', nrow=8)
    del comparison
    torch.cuda.empty_cache()

    # process train images one by one
    # create reconstruction tensor
    comparison = torch.zeros(64, 3, 256, 256)
    if CUDA:
        comparison = comparison.cuda()

    for i in range(0, 32):
        # read the image tensor
        test_img = trans(Image.open(train_path + str(i) + '.jpg').convert('RGB')).view(1, 3, 256, 256)
        if CUDA:
            test_img = test_img.cuda()
        # obtain latent space
        x_latent, a = model.module.encoder_V(test_img)  # .module.encoder_V(test_img)
        # produce reconstructed img
        recon = model.module.decoder_V(x_latent, a)  # .module.decoder_V(x_latent, a)
        # save the original and reconstructed imgs to tensor
        # each row shows the 8 images
        # with right below them the reconstructed output
        row = (i + 1) // 8  # the row index for this img
        mod = (i + 1) % 8
        if not mod:
            row -= 1
            col = 8
        else:
            col = (i + 1) % 8
        row *= 2

        comparison[row * 8 + col - 1, :] = test_img
        comparison[(row + 1) * 8 + col - 1, :] = recon
        del test_img
        del recon

        # produce synthetic images with varied latent values
        # modify latent vector
        rng = np.arange(-10, 10 + 2, 3)  # set variable varies in [-3,3] with step 1
        num_features = len(rng)  # change the first seven features
        batch_image = batch_img_producer(x_latent, rng, num_features)
        # produce reconstructed images
        recon_img = model.module.decoder_V(batch_image, a.repeat(len(rng) * num_features, 1, 1,
                                                                 1))  # .module.decoder_V(batch_image, a.repeat(len(rng) * num_features, 1, 1, 1))
        del batch_image
        save_image(recon_img, result_path + 'synthetic_train_' + str(i) + '.jpg', nrow=len(rng))
        del recon_img
        # save results to the folder, note that should save to the shared folder in docker image, and view files in local folder
    print('Generating images..')
    save_image(comparison.data, result_path + 'reconstruction_train.jpg', nrow=8)


def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr

    lr = init_lr * 0.1  # drop to 0.1*init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def batch_img_producer(x_latent, rng, num_feature_groups):
    # duplicate latent vector with individually changed feature value
    batch_img = x_latent.repeat(len(rng) * num_feature_groups, 1)
    feature_group_size = x_latent.shape[1] // num_feature_groups
    rng = torch.tensor(rng).view(-1, 1)
    batch_rng = rng.repeat(1, feature_group_size)

    for i in range(0, num_feature_groups):
        if i == num_feature_groups - 1:
            batch_img[(i * len(rng)):((i + 1) * len(rng)), i * feature_group_size:] = rng.repeat(1, x_latent.shape[
                1] - i * feature_group_size)[:]
            break

        batch_img[i * len(rng):(i + 1) * len(rng), i * feature_group_size:(i + 1) * feature_group_size] = batch_rng[
                                                                                                          :]  # vary the values of i-th feature group for i-th image batch (in total num_features batches)

    return batch_img


for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, 8)
    # print(learning_rate)
    train(epoch)
    test(epoch)

# torch.save(model.state_dict(), result_path+'model.pt')

# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
# On the last epoch, test performance on the 32 test samples
# tester()


