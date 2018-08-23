import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import re

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
root_path = os.getcwd() + '/'#'/home/lily/Desktop/food/' #/home/FoodRecog/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder='food-101/images/'#'ready_chinese_food'#scaled_images ready_chinese_food
image_path = os.path.join(root_path,image_folder)

file_path = root_path#os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = ('/ingredients-101/annotations/ingredients_simplified.txt')#os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'train.txt')
validation_data_path = os.path.join(file_path, 'val.txt')
test_data_path = os.path.join(file_path, 'test.txt')

result_path = root_path + 'results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


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
            file1 = io.open('train_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file1.readlines()]
            # ingredients = matio.loadmat(file_path + 'ingredient_train_feature.mat')['ingredient_train_feature'][0:66071,
            #               :]

            # with io.open(validation_data_path, encoding='utf-8') as file:
            #     path_to_images1 = file.read().split('\n')
            # file3 = io.open('val_label.txt',encoding='utf-8')
            # labels = [int(i[:-1]) for i in file3.readlines()]
            # # ingredients1 = matio.loadmat(file_path + 'ingredient_validation_feature.mat')[
            # #                    'ingredient_validation_feature'][0:11016, :]

            # path_to_images = path_to_images + path_to_images1
            # labels = np.concatenate([labels, labels1], 1)[0, :]
            #ingredients = np.concatenate([ingredients, ingredients1], 0)

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            file4 = io.open('val_label.txt',encoding='utf-8')
            labels1 = [int(i[:-1]) for i in file4.readlines()]
            #labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']
            #ingredients1 = matio.loadmat(file_path + 'ingredient_validation_feature.mat')['ingredient_validation_feature'][0:11016,:]
            #
            path_to_images = path_to_images+path_to_images1
            labels.extend(labels1)

        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            file2 = io.open('test_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file2.readlines()]
            # ingredients = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature'][0:33154,
            #               :]
        else:
            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            file3 = io.open('val_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file3.readlines()]
        # else:
        #     with io.open(validation_data_path, encoding='utf-8') as file:
        #         path_to_images = file.read().split('\n')
        #     labels = matio.loadmat(file_path + 'validation_label.mat')['validation_label'][0, :]
        #     ingredients = matio.loadmat(file_path + 'ingredient_validation_feature.mat')[
        #         'ingredient_validation_feature']

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
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=101, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * ((image_size[0] // (2 ** 5))**2), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not re.match(k, 'classifier.0.weight') and not re.match(k, 'classifier.6.weight') and not re.match(k, 'classifier.6.bias')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not re.match(k, 'classifier.0.weight') and not re.match(k, 'classifier.6.weight') and not re.match(k, 'classifier.6.bias')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model



# decoder network for image
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.latent_re = nn.Sequential(
            nn.Linear(4096, 512 * ((image_size[0] // (2 ** 5)) ** 2)),
            nn.ReLU(True),
            nn.Dropout(),
        )

        batchNorm_momentum = 0.1

        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv54d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn54d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv44d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn44d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv34d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn34d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x5p = self.latent_re(x)
        x5p = x5p.view(x.size(0), 512, image_size[0] // (2 ** 5), image_size[0] // (2 ** 5))

        # Stage 5d
        x5d = self.upsample5(x5p)
        x54d = F.relu(self.bn54d(self.conv54d(x5d)))
        x53d = F.relu(self.bn53d(self.conv53d(x54d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.upsample4(x51d)
        x44d = F.relu(self.bn44d(self.conv44d(x4d)))
        x43d = F.relu(self.bn43d(self.conv43d(x44d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.upsample3(x41d)
        x34d = F.relu(self.bn34d(self.conv34d(x3d)))
        x33d = F.relu(self.bn33d(self.conv33d(x34d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.upsample2(x31d)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.upsample1(x21d)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        x_recon = self.sigmoid(x11d)
        return x_recon


# # encoder network for textual channel
# class encoder_t(nn.Module):
#     def __init__(self):
#         super(encoder_t, self).__init__()
#
#         batchNorm_momentum = 0.1
#
#         self.nn1 = nn.Linear(353, 4096)
#         self.bn1 = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)
#         self.nn2 = nn.Linear(4096, 4096)
#         self.bn2 = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)
#
#     def forward(self, y):
#         # compute latent vectors
#         y = F.relu(self.bn1(self.nn1(y)))
#         y_latent = F.relu(self.bn2(self.nn2(y)))
#         return y_latent
#
#
# # decoder network for textual channel
# class decoder_t(nn.Module):
#     def __init__(self):
#         super(decoder_t, self).__init__()
#
#         batchNorm_momentum = 0.1
#
#         self.nn2d = nn.Linear(4096, 4096)
#         self.bn2d = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)
#
#         self.nn1d = nn.Linear(4096, 353)
#
#     def forward(self, y):  # x:image, y:ingredient
#         # compute reconstructed vectors
#         y = F.relu(self.bn2d(self.nn2d(y)))
#         y_recon = self.nn1d(y)
#
#         return y_recon


# entire model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn(pretrained=True)
        self.decoder = SegNet()

        # network for ingredient channel
        # encoder
        #self.encoder_t = encoder_t()
        #self.decoder_t = decoder_t()

        # classifier
        self.classifier = nn.Linear(blk_len, 101)

        # high performance computing

    #        if CUDA:
    #           self.classifier = nn.DataParallel(self.classifier).cuda()#.to(0)
    #          self.encoder = nn.DataParallel(self.encoder).cuda()  # device_ids=[0,1,2]
    #         self.decoder = nn.DataParallel(self.decoder).cuda()
    #        self.encoder_t = nn.DataParallel(self.encoder_t).cuda()  # device_ids=[0,1,2]
    #       self.decoder_t = nn.DataParallel(self.decoder_t).cuda()

    def forward(self, x):  # x:image, y:ingredient
        # compute image latent vectors & recons
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)

        # compute predicts
        predicts = self.classifier(x_latent[:, 0:blk_len])

        # compute ingredient vectors
        #y_latent = self.encoder_t(y)
        #predicts_t = self.classifier(y_latent[:, 0:blk_len])
        #y_recon = self.decoder_t(y_latent)

        return predicts, x_recon, x_latent[:, 0:blk_len]#,predicts_t,#y_recon, , y_latent[:, 0:blk_len]


# —Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
EPOCHS = 6
learning_rate = 1e-4
blk_len = 1536

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

train_loader = torch.utils.data.DataLoader(
    FoodData(train_data=True, test_data=False,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    FoodData(train_data=False, test_data=True,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
model = MyModel()
if CUDA:
    model.cuda()
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
if CUDA:
    criterion.cuda()


def loss_function(predicts_V, labels, data, x_recon):
    CE_V = criterion(predicts_V, labels) * 20
    RE_V = torch.sum((data - x_recon) ** 2) * (1e-5)
    return CE_V, RE_V


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

    total_time = time.time()

    for batch_idx, (data, labels) in enumerate(train_loader):
        #  if batch_idx == 1:
        #      break
        #print("Round start")
        start_time = time.time()
        data = Variable(data)
        #ingredients = Variable(ingredients)
        if CUDA:
            data = data.cuda()
            labels = labels.cuda()
            #ingredients = ingredients.cuda()
        optimizer.zero_grad()

        # obtain output from model
        predicts_V,x_recon,x_latent = model(data)
        #print("model end")

        # data = data.cuda(1)
        # labels = labels.cuda(1)
        # #ingredients = ingredients.cuda(1)
        # predicts_V = predicts_V.cuda(1)
        # #predicts_T = predicts_T.cuda(1)
        # x_recon = x_recon.cuda(1)
        # #y_recon = y_recon.cuda(1)
        # x_latent = x_latent.cuda(1)
        #y_latent = y_latent.cuda(1)
        # calculate scalar loss
        CE_V, RE_V = loss_function(predicts_V,labels,data,x_recon)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss = CE_V + RE_V
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        # compute accuracy
        matches_V, hits_V = top_match(predicts_V, labels)
        #matches_T, hits_T = top_match(predicts_T, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        #top1_accuracy_total_T += matches_T
        #top1_accuracy_cur_T = matches_T / float(labels.size(0))
        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        #top5_accuracy_total_T += hits_T
        #top5_accuracy_cur_T = hits_T / float(labels.size(0))

        if epoch == 1 and batch_idx == 0:
            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                file.write('%f\n' % train_loss)
            
        if batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data, top1_accuracy_cur_V, top5_accuracy_cur_V,
                    round((time.time() - start_time) * LOG_INTERVAL, 4),
                    round((time.time() - total_time), 4)))

    print(
        '====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Time:{}'.format(
            epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset),
                   top5_accuracy_total_V / len(train_loader.dataset),
            round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        file.write('%f\n' % (train_loss / len(train_loader)))


def test(epoch):
    # toggle model to test / inference mode
    print('testing starts..')
    model.eval()
    top1_accuracy_total_V = 0
    top5_accuracy_total_V = 0
    total_time = time.time()

    # each data is of BATCH_SIZE (default 128) samples
    for test_batch_idx, (data, labels) in enumerate(test_loader):
        # if test_batch_idx == 1:
        #   break
        start_time = time.time()

        # we're only going to infer, so no autograd at all required
        with torch.no_grad():
            data = Variable(data)
            #ingredients = Variable(ingredients)

        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()
            #ingredients = ingredients.cuda()

        predicts_V = model(data)

        # compute accuracy
        matches_V, hits_V = top_match(predicts_V, labels)
        #matches_T, hits_T = top_match(predicts_T, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        #top1_accuracy_total_T += matches_T
        #top1_accuracy_cur_T = matches_T / float(labels.size(0))
        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        #top5_accuracy_total_T += hits_T
        #top5_accuracy_cur_T = hits_T / float(labels.size(0))

        print(
            'Testing batch: {} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                test_batch_idx, top1_accuracy_cur_V, top5_accuracy_cur_V,
                round((time.time() - start_time), 4), round((time.time() - total_time), 4)))

    print(
        '====> Test set: Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Total Time:{}'.format(
            top1_accuracy_total_V / len(test_loader.dataset), top5_accuracy_total_V / len(test_loader.dataset),
            round((time.time() - total_time), 4)))
    if epoch!=0 and epoch%2 ==0:
        model_save_name = 'exp11_vgg_'+ str(learning_rate) + '_'+ str(epoch) + '.pkl'
        torch.save(model, model_save_name)
        #torch.save(model.state_dict(),model_save_name)
        print("model saved!")
    # save testing performance per epoch
    with io.open(result_path + 'test_accuracy.txt', 'a', encoding='utf-8') as file:
        file.write('%f ' % (top1_accuracy_total_V / len(test_loader.dataset)))
        file.write('%f\n ' % (top5_accuracy_total_V / len(test_loader.dataset)))



def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr

    lr = init_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, 4)
    print(learning_rate)
    train(epoch)
    test(epoch)
