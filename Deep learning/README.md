# Pytorch with Modal Arts CNN开荒笔记

目标：在表情识别数据集上迁移学习resnet18，保存模型

## 引入库

使用CNN做图像处理一般使用的库：

```python
import os
import tarfile

import pandas as pd
import numpy as np

from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```



## 数据读取（OBS）

Modal Arts从OBS中获取数据，文件数不宜太多。载入数据集使用的方法：

1. 把数据集打成tar包上传到OBS

2. 使用Sync OBS同步notebook和数据集，notebook的当前目录(.)会载入数据集

   > notebook的运行目录不是它的存储目录。`.`目录下是默认没有文件的，使用`Sync OBS`同步的数据会被挂载到`.`目录下。

3. 使用`os`, `tarfile`包解压缩文件，工具方法：

   ```python
   def untar(tar_file, ext_path):
       if os.path.exists(ext_path):
           os.removedirs(ext_path)
       os.makedirs(ext_path)
       t = tarfile.open(tar_file)
       t.extractall(path=ext_path)
   ```

4. 一般的查看图像的方法

   ```python
   def show_img(idx, df):
       if idx < 0 or idx > 19107:
           print("idx error")
       with Image.open(os.path.join(image_path, df.loc[idx].pic_name)) as img:
           plt.imshow(img)
           plt.axis('off')
           print("label : " + str(df.loc[idx].label))
   ```

   

## 定义数据集（Dataset）

### 介绍

`torch.utils.data.Dataset`是一个PyTorch用来表示数据集的抽象类。我们用这个类来处理自己的数据集的时候必须继承`Dataset`,然后重写下面的函数：

1. `__len__(self)`: 使得`len(dataset)`返回数据集的大小；
2. `__getitem__(self, idx)`：使得支持`dataset[i]`能够返回第i个数据样本这样的下标操作。
3. `__init__(self, *args)`：完成数据的读取和预处理

### 实例

```python
class ERDataset(Dataset):
    """expression-recognition dataset"""
    def __init__(self, label_file, image_path, transform=None):
        self.df = pd.read_csv(label_file)
        self.image_path = image_path
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_file = os.path.join(self.image_path,
                                self.df.iloc[idx].pic_name)
        image = io.imread(img_file)
        label = self.df.iloc[idx].label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```


### 注意：

1. 使用`PIL.Image.open`来打开图像
2. 返回一个tuple

## 数据预处理（Transform）

使用`torchvision.transforms`包中提供的函数进行变换。它们的参数都是一个图片(Image对象或ndarray或tensor)

`Resize((h,w))`：调整大小

`ToTensor()`：把`PIL.Image`对象变成tensor

`transforms.Normalize((0.1307,), (0.3081,))`：归一化。之前需要ToTensor。

### 一、 裁剪——Crop

#### 1.随机裁剪：transforms.RandomCrop

`class torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')` 

功能：依据给定的size随机裁剪 参数：

1.  size- (sequence or int)，若为sequence,则为(h,w)，若为int，则(size,size) 
2. padding-(sequence or int, optional)，此参数是设置填充多少个pixel。 当为int时，图像上下左右均填充int个，例如padding=4，则上下左右均填充4个pixel，若为32*32，则会变成40*40。 当为sequence时，若有2个数，则第一个数表示左右扩充多少，第二个数表示上下的。当有4个数时，则为左，上，右，下。
3. fill- (int or tuple) 填充的值是什么（仅当填充模式为constant时有用）。int时，各通道均填充该值，当长度为3的tuple时，表示RGB通道需要填充的值。 
4. padding_mode- 填充模式，这里提供了4种填充模式，1.constant，常量。2.edge 按照图片边缘的像素值来填充。3.reflect，暂不了解。 4. symmetric，暂不了解。

#### 2.中心裁剪：transforms.CenterCrop

class torchvision.transforms.CenterCrop(size) 功能：依据给定的size从中心裁剪 参数： size- (sequence or int)，若为sequence,则为(h,w)，若为int，则(size,size)

#### 3.随机长宽比裁剪 transforms.RandomResizedCrop

class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2) 功能：随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size 参数： size- 输出的分辨率 scale- 随机crop的大小区间，如scale=(0.08, 1.0)，表示随机crop出来的图片会在的0.08倍至1倍之间。 ratio- 随机长宽比设置 interpolation- 插值的方法，默认为双线性插值(PIL.Image.BILINEAR)

#### 4.上下左右中心裁剪：transforms.FiveCrop

class torchvision.transforms.FiveCrop(size) 功能：对图片进行上下左右以及中心裁剪，获得5张图片，返回一个4D-tensor 参数： size- (sequence or int)，若为sequence,则为(h,w)，若为int，则(size,size)

#### 5.上下左右中心裁剪后翻转: transforms.TenCrop

class torchvision.transforms.TenCrop(size, vertical_flip=False) 功能：对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得10张图片，返回一个4D-tensor。 参数： size- (sequence or int)，若为sequence,则为(h,w)，若为int，则(size,size) vertical_flip (bool) - 是否垂直翻转，默认为flase，即默认为水平翻转

### 二、翻转和旋转——Flip and Rotation

#### 6.依概率p水平翻转transforms.RandomHorizontalFlip

class torchvision.transforms.RandomHorizontalFlip(p=0.5) 功能：依据概率p对PIL图片进行水平翻转 参数： p- 概率，默认值为0.5

#### 7.依概率p垂直翻转transforms.RandomVerticalFlip

class torchvision.transforms.RandomVerticalFlip(p=0.5) 功能：依据概率p对PIL图片进行垂直翻转 参数： p- 概率，默认值为0.5

#### 8.随机旋转：transforms.RandomRotation

class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None) 功能：依degrees随机旋转一定角度 参数： degress- (sequence or float or int) ，若为单个数，如 30，则表示在（-30，+30）之间随机旋转 若为sequence，如(30，60)，则表示在30-60度之间随机旋转 resample- 重采样方法选择，可选 PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC，默认为最近邻 expand- ? center- 可选为中心旋转还是左上角旋转

### 三、图像变换

#### 9.resize：transforms.Resize

class torchvision.transforms.Resize(size, interpolation=2) 功能：重置图像分辨率 参数： size- If size is an int, if height > width, then image will be rescaled to (size * height / width, size)，所以建议size设定为h*w interpolation- 插值方法选择，默认为PIL.Image.BILINEAR

#### 10.标准化：transforms.Normalize

class torchvision.transforms.Normalize(mean, std) 功能：对数据按通道进行标准化，即先减均值，再除以标准差，注意是 h*w*c

#### 11.转为tensor：transforms.ToTensor

class torchvision.transforms.ToTensor 功能：将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。

#### 12.填充：transforms.Pad

class torchvision.transforms.Pad(padding, fill=0, padding_mode='constant') 功能：对图像进行填充 参数： padding-(sequence or int, optional)，此参数是设置填充多少个pixel。 当为int时，图像上下左右均填充int个，例如padding=4，则上下左右均填充4个pixel，若为32*32，则会变成40*40。 当为sequence时，若有2个数，则第一个数表示左右扩充多少，第二个数表示上下的。当有4个数时，则为左，上，右，下。 fill- (int or tuple) 填充的值是什么（仅当填充模式为constant时有用）。int时，各通道均填充该值，当长度为3的tuple时，表示RGB通道需要填充的值。 padding_mode- 填充模式，这里提供了4种填充模式，1.constant，常量。2.edge 按照图片边缘的像素值来填充。3.reflect，？ 4. symmetric，？

#### 13.修改亮度、对比度和饱和度：transforms.ColorJitter

class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) 功能：修改修改亮度、对比度和饱和度

#### 14.转灰度图：transforms.Grayscale

class torchvision.transforms.Grayscale(num_output_channels=1) 功能：将图片转换为灰度图 参数： num_output_channels- (int) ，当为1时，正常的灰度图，当为3时， 3 channel with r == g == b

#### 15.线性变换：transforms.LinearTransformation()

class torchvision.transforms.LinearTransformation(transformation_matrix) 功能：对矩阵做线性变化，可用于白化处理！ whitening: zero-center the data, compute the data covariance matrix 参数： transformation_matrix (Tensor) – tensor [D x D], D = C x H x W

#### 16.仿射变换：transforms.RandomAffine

class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0) 功能：仿射变换

#### 17.依概率p转为灰度图：transforms.RandomGrayscale

class torchvision.transforms.RandomGrayscale(p=0.1) 功能：依概率p将图片转换为灰度图，若通道数为3，则3 channel with r == g == b

#### 18.将数据转换为PILImage：transforms.ToPILImage

class torchvision.transforms.ToPILImage(mode=None) 功能：将tensor 或者 ndarray的数据转换为 PIL Image 类型数据 参数： mode- 为None时，为1通道， mode=3通道默认转换为RGB，4通道默认转换为RGBA

#### 19.transforms.Lambda

Apply a user-defined lambda as a transform. 暂不了解，待补充。

### 四、对transforms操作，使数据增强更灵活

PyTorch不仅可设置对图片的操作，还可以对这些操作进行随机选择、组合

#### 20.transforms.RandomChoice(transforms)

功能：从给定的一系列transforms中选一个进行操作，randomly picked from a list

#### 21.transforms.RandomApply(transforms, p=0.5)

功能：给一个transform加上概率，以一定的概率执行该操作

#### 22.transforms.RandomOrder

功能：将transforms中的操作顺序随机打乱

也可以自己实现一些变换。比如dataset中同时存在图像和label，可以如下面来只对图像进行变换。

```python
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```

## 加载数据（Data Loader）

`torch.utils.data.DataLoader`

### 功能

- 按照`batch_size`获得批量数据；
- 打乱数据顺序；
- 用多线程`multiprocessing`来加载数据；

### 参数

第一个参数传入实例化的`dataset`对象

第二个参数传入`batch_size`，表示每个batch包含多少个数据。

第三个参数传入`shuffle`，布尔型变量，表示是否打乱。

第四个参数传入`num_workers`表示使用几个线程来加载数据，`-1`是全部。

> batch的选择：1、2的幂，2、一般几十到几百，小一点收敛快，3、二阶优化算法使用大batch，几千上万都可以。

```python
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
```

## 建立模型

### 自建模型

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 构造函数中定义模型的层次
    def forward(self, x):
        # 定义前向传播

model = MyNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model, device, loader, optimizer, n_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
    	data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.some_loss_function(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
```

#### nn

`nn.Conv2d(in_c, out_c, size)`：输入通道数，输出通道数，卷积核大小

> 卷积后大小：new_size = (old_size - kernel_size + 2 * padding) / stride + 1

`nn.Linear(in,out)`：输入向量维度，输出向量维度

#### F

`F.relu(x)`

`F.sigmoid(x)`

`F.max_pool2d(x, sizex, sizey)`：池化核的size

`F.log_softmax(x, dim)`：对应dim上进行softmax计算

`F.nll_loss(output, target)`两者的size不同：output:(n_batch, c)，target:(n_batch)；output是一个onehot，target数字是下标。

#### tensor

`out.view(*size)`：展开为对应size，维度为-1则自动计算。

## 迁移学习

1. 获取模型

```python
# 下载预训练模型
res18 = models.resnet18(pretrained = True)
# 冻结所有参数
for param in res18.parameters():
    param.requires_grad = False
```

2. 简单的层修改

```python
#调用模型
model = models.resnet50(pretrained=True)
#提取fc层中固定的参数
fc_features = model.fc.in_features
#修改类别为9
model.fc = nn.Linear(fc_features, 9)
```

3. 修改网络结构，需要参数覆盖

```python
# 定义mocel
class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        #去掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)  
        self.Linear_layer = nn.Linear(2048, 8)

#加载model
resnet50 = models.resnet50(pretrained=True)
cnn = CNN(Bottleneck, [3, 4, 6, 3])
#读取参数
pretrained_dict = resnet50.state_dict()
model_dict = cnn.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
cnn.load_state_dict(model_dict)
# print(resnet50)
print(cnn)
```

