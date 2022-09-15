# -*- coding: utf-8 -*-
from torchvision import transforms
import torch
from fnet import barmodel
from PIL import Image
from tools import default_loader
from glob import glob
input_size = 50
newsize=(50,50)
label_num =43
labels_ids = [str(w) for w in range(label_num)]
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%*'
id2chars = {idx:w for idx,w in enumerate(chars)}


net = barmodel(in_features=1280, out_features=2,pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
checkpoint=torch.load('model/net_050.pth')
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
net.eval()
val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
# label2tensor = {}
labels_tensors_merge = None
res = []
for id in labels_ids:
    file = 'labels/'+id+'.png'
    res.append(val_transform(default_loader(file)))
    labels_tensors_merge = torch.stack(res, dim=0)
def predict(pic_path):
    pic_pil = default_loader(pic_path,is_observ=True)
    pic_tensors = [val_transform(pic_pil)]*label_num
    pics_tensors_merge = torch.stack(pic_tensors, dim=0)
    with torch.no_grad():
        predict = net(pics_tensors_merge,labels_tensors_merge)

        predict = predict[:,1]
        print(predict)
        # predict[1]=0.0
        max_idx = torch.argmax(predict)
        char = id2chars[int(max_idx)]
        print(char)
        return char
        # predict = predict.cpu().numpy()

#
# from mdataset import evalset
# from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
#         transforms.RandomHorizontalFlip(),
#         # AddPepperNoise(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#
#     ]),
#
#     'val': transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
# }
#
#
#
# def clc_total_correct_predict(predicts, labels):
#     """
#     统计模型所有标签全部预测正确的个数
#     :param predicts: 预测的标签[N.C]
#     :param labels: 正确的标签[N.C]
#     :return: 标签全部预测正确的个数
#     """
#     total_correct = 0
#
#     dim0 = labels.size(0)
#     for i in range(dim0):
#         if torch.equal(predicts[i], labels[i]):
#             total_correct += 1.0
#         else:
#             pass
#     return total_correct
# def predict_transform(outputs, confidence):
#     """
#     将模型的输出经过阈值筛选变化为0或者1
#     :param outputs: 模型的输出值[N,C]
#     :param confidence: 阈值，大于该阈值的值为1,小于为0
#     :return: 最终的预测值[N,C]
#     """
#     # gt,将大于confidence变为True,否则为false,然后经过float()变为1和0
#     predicts = outputs.gt(confidence).float()
#
#     return predicts
#
#
# def my_collate_fn(batch):
#     '''
#     batch中每个元素形如(data, label)
#     '''
#     # 过滤为None的数据
#     batch = list(filter(lambda x: x[0] is not None, batch))
#     if len(batch) == 0: return torch.Tensor()
#     return default_collate(batch)  # 用默认方式拼接过滤后的batch数据
#
#
# image_datasets = evalset(stage='val',transform=data_transforms['val'])
# # Create training and validation dataloaders
# dataloaders_dict =  torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=2,
#                                    collate_fn=my_collate_fn)
#
# f = open('out.txt','w',encoding='utf-8')
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for data in dataloaders_dict:
#         net.eval()
#         images1, images2, labels,fn,fn_out = data
#         labels = torch.eye(2)[labels]
#         images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
#         outputs = net(images1, images2)
#         # 取得分最高的那个类 (outputs.data的索引号)
#         predicted = predict_transform(outputs=outputs, confidence=0.5)
#         f.write(str(outputs) + '\n')
#         f.write(str(fn)+'\n')
#         f.write(str(fn_out)+'\n')
#         f.write(str(labels)+'\n')
#         f.write(str(predicted)+'\n')
#         f.write('*****************\n')
#
#         total += labels.size(0)
#         correct += clc_total_correct_predict(predicts=predicted, labels=labels)
#     print('测试分类准确率为：%.3f%%' % (100. * float(correct) / float(total)))
#     acc = 100. * float(correct) / float(total)
# f.close()
# chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
# #2:A,3:B,4:1,5:2,6:+,7:%
# %+21BA

a3=predict('6/10/image_4_1_240_320/3.png')
a4=predict('6/10/image_4_1_240_320/4.png')
a5=predict('6/10/image_4_1_240_320/5.png')
a6=predict('6/10/image_4_1_240_320/6.png')
a7=predict('6/10/image_4_1_240_320/7.png')
a8=predict('6/10/image_4_1_240_320/8.png')

a=6
m=0
if a3=='%':
    m+=1
if a4=='+':
    m+=1
if a5=='2':
    m+=1
if a6=='1':
    m+=1
if a7=='B':
    m+=1
if a8=='A':
    m+=1
print(float(m)/float(a))


