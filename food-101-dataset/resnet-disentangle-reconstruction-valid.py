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


#—Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = os.getcwd()+'/'#'/home/lily/Desktop/food/' #/home/FoodRecog/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder='food-101/images/'#'ready_chinese_food'#scaled_images ready_chinese_food
image_path = os.path.join(root_path,image_folder,'/')

file_path = root_path#os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = ('/ingredients-101/annotations/ingredients_simplified.txt')#os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'train.txt')
validation_data_path = os.path.join(file_path, 'val.txt')
test_data_path = os.path.join(file_path, 'test.txt')

result_path = root_path + 'results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

#—Create dataset———————————————————————————————————————————————————————————————————————————————————————————————————————
def default_loader(path):
    img_path = root_path + image_folder + path

    jpgfile = Image.open(img_path).convert('RGB')

    return jpgfile

class FoodData(torch.utils.data.Dataset):

    def __init__(self, train_data = True, test_data = False,transform=None,
                 loader=default_loader):

        #load image paths / label file
        if train_data:
            with io.open(train_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            # # train_load = 'train_label.mat'
            # labels = matio.loadmat(train_load)
            file1 = io.open('train_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file1.readlines()]

            #matio.savemat(path,['train_label',label])
            #ingredients = matio.loadmat(file_path + 'ingredient_train_feature.mat')['ingredient_train_feature'][0:66071,:]

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            file4 = io.open('val_label.txt',encoding='utf-8')
            labels1 = [int(i[:-1]) for i in file4.readlines()]
            #labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']
            #ingredients1 = matio.loadmat(file_path + 'ingredient_validation_feature.mat')['ingredient_validation_feature'][0:11016,:]
            #
            path_to_images =path_to_images+path_to_images1
            labels.extend(labels1)
            #labels = np.concatenate([labels,labels1],1)[0,:]
            # #ingredients = np.concatenate([ingredients, ingredients1], 0)
        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')

            file2 = io.open('test_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file2.readlines()]


            #ingredients = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature'][0:33154,:]
        else:
            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            file3 = io.open('val_label.txt',encoding='utf-8')
            labels = [int(i[:-1]) for i in file3.readlines()]

            #ingredients = matio.loadmat(file_path + 'ingredient_validation_feature.mat')['ingredient_validation_feature']

        #ingredients = ingredients.astype(np.float32)
        self.path_to_images = path_to_images
        self.labels = labels
        #self.ingredients=ingredients
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        #get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        #get label
        label = self.labels[index]
        # get ingredients 353-D vector
        #ingredient = self.ingredients[index,:]
        return img,label#, ingredient

    def __len__(self):
        return len(self.path_to_images)


#—Model———————————————————————————————————————————————————————————————————————————————————————————————————————
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
                     padding=1, output_padding = 1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=172):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        #define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0]) #64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#256-512
        self.avgpooling = nn.AvgPool2d(image_size[0] // (2 ** 5), stride=1)
        # get latent representation
        latent_len = 200
        self.latent = nn.Linear(512 * block.expansion, latent_len)

        #classifier
        self.classifier1 = nn.Linear(latent_len, num_classes)

        # define resnet decoder
        self.latent_re = nn.Linear(latent_len,512)
        self.layer5 = self._make_Delayer(DeBasicBlock, 256, layers[3], stride=2)#512-256
        self.layer6 = self._make_Delayer(DeBasicBlock, 128, layers[3], stride=2)  # 256-128
        self.layer7 = self._make_Delayer(DeBasicBlock, 64, layers[3], stride=2)  # 128-64
        self.layer8 = self._make_Delayer(DeBasicBlock, 64, layers[3], stride=1)  # 64-64
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.deconv9 = nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=2,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def forward(self, x): #x:image y:ingredient
        #print(x.shape)
        x = self.conv1(x) #／2
        x = self.bn1(x)
        x = self.relu(x)
        [x,a] = self.maxpool(x) #／2
        #print(x.shape)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x) #／2
        #print(x.shape)
        x = self.layer3(x) #／2
        #print(x.shape)
        x = self.layer4(x) #／2
        #print(x.shape)
        x = self.avgpooling(x) #(1x1)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        x_latent= self.latent(x)
        #print(x_latent.shape)
        predicts = self.classifier1(x_latent)

        x = self.latent_re(x_latent)
        x = x.view(x.shape[0], 512, 1, 1)
        #print(x.shape)
        x = F.upsample(x, scale_factor=image_size[0] // (2 ** 5), mode='nearest')
        #print(x.shape)
        x = self.layer5(x)
        #print(x.shape)
        x = self.layer6(x)
        #print(x.shape)
        x = self.layer7(x)
        #print(x.shape)
        x = self.layer8(x)
        #print(x.shape)
        x = self.unmaxpool(x,a)
        #print(x.shape)
        x = self.deconv9(x)
        #print(x.shape)
        x = self.sigmoid(x)
        #print("end model")
        return predicts,x



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)    
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



#—Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
#Image Info
no_of_channels = 3
image_size = [256,256]#[64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1#True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 50
EPOCHS = 10
learning_rate=0.0001

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

train_loader = torch.utils.data.DataLoader(
    FoodData(train_data = True, test_data = False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    FoodData(train_data = False, test_data = True,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


#  —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
model = resnet18(pretrained=True)
#model = model.load_state_dict(torch.load(model_file))
if CUDA:
    model.cuda()
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion =nn.CrossEntropyLoss()
if CUDA:
    criterion.cuda()

def loss_function(predicts_V,labels,data,x): #-> Variable:
    CE_V = criterion(predicts_V, labels)* 8
    RE = torch.sum((data-x)**2)*(1e-5)
    #CE_T = criterion(predicts_T, labels - 1)
    #RE_latent = torch.sum((image_latent-ingredient_latent)**2)
    #print('Loss ====> CE_V: {} | RE: {} | Loss: {}'.format(CE_V, RE,CE_V+RE))
    #print('Loss ====> CE_V: {} | CE_T: {} | RE_latent: {} |'.format(CE_V, CE_T,RE_latent))
    return CE_V,RE#+CE_T#+0.01*RE_latent

def top_match(predicts, labels):
    sorted_predicts = predicts.cpu().data.numpy().argsort()
    top1_labels = sorted_predicts[:,-1:][:,0]
    match = float(sum(top1_labels == (labels)))
    
    top5_labels = sorted_predicts[:, -5:]
    hit = 0
    for i in range(0,labels.size(0)):
        hit+= (labels[i]) in top5_labels[i,:]
    
    
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
        #print("<------Round start----->")
      #  if batch_idx == 1:
      #      break
        start_time = time.time()
        data = Variable(data)
        #ingredients = Variable(ingredients)
        if CUDA:
            data = data.cuda()
            labels = labels.cuda()
            #ingredients = ingredients.cuda()
        optimizer.zero_grad()

        # obtain output from model
        predicts_V,x = model(data)
        
        # calculate scalar loss
        CE_V, RE = loss_function(predicts_V, labels, data, x)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        #print("CE_V: "+str(CE_V))
        #print("RE: "+str(RE))


        loss = CE_V + RE
        loss.backward()
        #print(" End backward !!!!!!!!")
        train_loss += loss.data
        optimizer.step()
        #compute accuracy
        matches_V,hits_V = top_match(predicts_V,labels)
        #matches_T,hits_T = top_match(predicts_T,labels)
            #top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        #top1_accuracy_total_T += matches_T
        #top1_accuracy_cur_T = matches_T / float(labels.size(0))
            #top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        #top5_accuracy_total_T += hits_T
        #top5_accuracy_cur_T = hits_T / float(labels.size(0))
        #print("end accuracy")
        if epoch == 1 and batch_idx == 0:
            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data, top1_accuracy_cur_V, top5_accuracy_cur_V, round((time.time() - start_time) * LOG_INTERVAL, 4),
                round((time.time() - total_time), 4)))


    print('====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Time:{}'.format(
        epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset), top5_accuracy_total_V / len(train_loader.dataset),
        round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write('%f\n' % (train_loss / len(train_loader)))

def test(epoch):
    # toggle model to test / inference mode
    print('testing starts..')
    model.eval()
    top1_accuracy_total_V = 0
    #top1_accuracy_total_T = 0
    top5_accuracy_total_V = 0
    #top5_accuracy_total_T = 0
    total_time = time.time()

    # each data is of BATCH_SIZE (default 128) samples
    for test_batch_idx, (data, labels) in enumerate(test_loader):
        #if test_batch_idx == 1:
         #   break
        start_time = time.time()


        # we're only going to infer, so no autograd at all required
        with torch.no_grad():
            data = Variable(data)
            #ingredients= Variable(ingredients)
            
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()
            #ingredients = ingredients.cuda()
            
        predicts_V,_= model(data)


        #compute accuracy
        matches_V,hits_V = top_match(predicts_V,labels)
        #matches_T,hits_T = top_match(predicts_T,labels)
            #top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))
        #top1_accuracy_total_T += matches_T
        #top1_accuracy_cur_T = matches_T / float(labels.size(0))
            #top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))
        #top5_accuracy_total_T += hits_T
        #top5_accuracy_cur_T = hits_T / float(labels.size(0))

        print('Testing batch: {} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
            test_batch_idx, top1_accuracy_cur_V, top5_accuracy_cur_V, round((time.time() - start_time), 4), round((time.time() - total_time), 4)))
 


    print('====> Test set: Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Total Time:{}'.format(top1_accuracy_total_V / len(test_loader.dataset), top5_accuracy_total_V / len(test_loader.dataset), round((time.time() - total_time),4)))
    if epoch!=0 and epoch%2==0:
    	model_save_name = 'exp9_resnet_'+ str(learning_rate) + '_'+ str(epoch) + '.pkl' 
    	torch.save(model,model_save_name)
    	print("model saved!")
    #save testing performance per epoch
    with io.open(result_path + 'test_accuracy.txt', 'a', encoding='utf-8') as file:
        file.write('%f ' %(top1_accuracy_total_V / len(test_loader.dataset)))
        file.write('%f/n ' %(top5_accuracy_total_V / len(test_loader.dataset)))
        #file.write('%f ' %(top1_accuracy_total_T / len(test_loader.dataset)))
        #file.write('%f\n' %(top5_accuracy_total_T / len(test_loader.dataset)))

def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr

    lr = init_lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# def batch_img_producer(x_latent, rng, num_feature_groups):
#     # duplicate latent vector with individually changed feature value
#     batch_img = x_latent.repeat(len(rng) * num_feature_groups, 1)
#     feature_group_size = x_latent.shape[1] // num_feature_groups
#     rng = torch.tensor(rng).view(-1, 1)
#     batch_rng = rng.repeat(1,feature_group_size)
#
#     for i in range(0, num_feature_groups):
#         if i == num_feature_groups - 1:
#             batch_img[(i * len(rng)):((i + 1) * len(rng)), i * feature_group_size:] = rng.repeat(1, x_latent.shape[
#                 1] - i * feature_group_size)[:]
#             break
#
#         a = i * len(rng)
#         b = (i + 1) * len(rng)
#         c = i * feature_group_size
#         d = (i + 1) * feature_group_size
#         e = batch_rng[:]
#         batch_img[a:b, c:d] = e  # vary the values of i-th feature group for i-th image batch (in total num_features batches)
#
#     return batch_img
# a = torch.randn(1, 5)
# rng = np.arange(-1, 1, 1) #set variable varies in [-3,3] with step 1
# num_feature_groups = len(rng)
# batch_image = batch_img_producer(a,rng,num_feature_groups)

for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer,learning_rate,epoch,3)
    print(learning_rate)
    train(epoch)
    test(epoch)
