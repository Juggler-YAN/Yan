# FCN（全卷积网络）———— 语义分割

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt

from PIL import Image

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size, voc_dir):
    """加载VOC语义分割数据集"""
    num_workers = 4
    train_data = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_data = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_data, test_data

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    acc_nums = 0
    nums = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            acc_nums = acc_nums + accuracy(net(X), y)
            nums = nums + y.numel()
    return acc_nums / nums

def predict(img):
    X = test_data.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

if __name__ == '__main__':

    # args
    batch_size = 32
    crop_size = (320, 480)
    num_epochs = 10
    lr = 0.001
    wd = 1e-3
    devices = try_all_gpus()
    voc_dir = './data/VOCdevkit/VOC2012'

    # data
    train_data, test_data = load_data_voc(batch_size, crop_size, voc_dir)

    # net
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=64, padding=16, stride=32))
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # loss
    def loss(inputs, targets):
        return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

    # opti
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    # train
    train_loss_arr = []
    test_acc_arr = []
    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()
        train_loss = 0
        train_nums = 0
        for X, y in train_data:
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.sum().backward()
            trainer.step()
            train_loss = train_loss + float(l.sum())
            train_nums = train_nums + y.shape[0]
        train_loss = train_loss / train_nums
        test_acc = evaluate_accuracy(net, test_data)
        print(f'{epoch + 1}: train_loss = {train_loss}, test_acc = {test_acc}')
        train_loss_arr.append(train_loss)
        test_acc_arr.append(test_acc)

    # plot
    plt.figure(1, figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss_arr, label='train_loss')
    plt.title("loss")
    plt.legend()
    plt.subplot(122)
    plt.plot(test_acc_arr, label='test_acc')
    plt.title("acc")
    plt.legend()
    # plt.show()
    plt.savefig("1.png")

    # predict
    test_images, test_labels = read_voc_images(voc_dir, False)
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[0], *crop_rect)
    pred = label2image(predict(X))
    imgs = [X.permute(1,2,0), pred.cpu(), torchvision.transforms.functional.crop(
                test_labels[0], *crop_rect).permute(1,2,0)]
    plt.figure(2)
    plt.imshow(imgs[0])
    plt.savefig("2.png")
    plt.figure(3)
    plt.imshow(imgs[1])
    plt.savefig("3.png")
    plt.figure(4)
    plt.imshow(imgs[2])
    plt.savefig("4.png")