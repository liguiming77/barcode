# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式

import os
import sys
# from tools import AddPepperNoise
from tools import predict_transform,clc_total_correct_predict

from yuvdataset import traincropset
from fnet import barmodel
root_path=os.path.abspath('.')
sys.path.append(root_path)
confidence = 0.5
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

warnings.filterwarnings("ignore")

resume = False
# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8 #mobilenetv2=40*4*6  newmobilenetv2=int(40*4*3*1.5) newefficientnet=40*4

# batch_size = 40*4*2 #mobilevit
# Number of epochs to train for
EPOCH = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
# feature_extract = True
feature_extract = False
# 超参数设置
pre_epoch = 0  # 定义已经遍历数据集的次数


def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


input_size = 50 #224#380

net = barmodel(1280,num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# nn.init.zeros_(net)
# 读取网络信息
#net.load_state_dict(torch.load('./model/net_007.pth'))

# Send the model to GPU
# net = net.to(device)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        # AddPepperNoise(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),

    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: traincropset(stage=x,transform=data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2,
                                   collate_fn=my_collate_fn) for x in ['train', 'val']} #mobilenetv2

# dataloaders_dict = {
#     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=18,
#                                    collate_fn=my_collate_fn) for x in ['train', 'val']}  #mobilevit

# b = image_datasets['train'].class_to_idx
# print(b)
# assert 1>2
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints')  # 输出结果保存路径

args = parser.parse_args()
params_to_update = net.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print("\t", name)


def main():
    ii = 0
    LR = 1e-3  # 学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")  # 定义遍历数据集的次数


    # criterion
    # criterion = LabelSmoothSoftmaxCE()
    ##new add
    criterion = torch.nn.BCELoss().to(device)
    ## end add

    # optimizer
    optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)



    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

    with open("log/acc.txt", "w",encoding='utf-8') as f:
        with open("log/log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step(epoch)

                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(dataloaders_dict['train'], 0):
                    # 准备数据
                    length = len(dataloaders_dict['train'])

                    input1,input2,target = data  # torch.Size([160, 3, 224, 224])
                    target = torch.eye(num_classes)[target]
                    input1,input2,target = input1.to(device),input2.to(device),target.to(device)

                    # print(target.shape)
                    # assert 1>2

                    # 训练
                    optimizer.zero_grad()
                    # forward + backward
                    output = net(input1,input2)
                    loss = criterion(output, target)

                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    # _, predicted = torch.max(output.data, 1)
                    predicts = predict_transform(outputs=output, confidence=confidence)
                    total += target.size(0)
                    correct += clc_total_correct_predict(predicts=predicts, labels=target)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                             100. * float(correct) / float(total)))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()
                    ## del data start
                    del data
                    # del input
                    del target
                    del output
                    del loss
                    del predicts
                    ## del data end

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in dataloaders_dict['val']:
                        net.eval()
                        images1, images2,labels = data
                        labels = torch.eye(num_classes)[labels]
                        images1, images2,labels = images1.to(device), images2.to(device),labels.to(device)
                        outputs = net(images1,images2)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        predicted = predict_transform(outputs=outputs, confidence=confidence)
                        total += labels.size(0)
                        correct += clc_total_correct_predict(predicts=predicted, labels=labels)
                    print('测试分类准确率为：%.3f%%' % (100. * float(correct) / float(total)))
                    acc = 100. * float(correct) / float(total)
                    scheduler.step(acc)

                    # 将每次测试结果实时写入acc.txt文件中
                    if ((epoch + 1) % 1 == 0):
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("log/best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == "__main__":
    main()
