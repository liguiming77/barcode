from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np

from glob import glob
from numpy import random
from tools import base_loader

def list_sub_list(list1,list2):
    return [w for w in list1 if w not in list2]
def list_sub_xing_check(list1):
    return  [w for w in list1 if '/1.png' not in w and '/2.png' not in w and '/9.png' not in w]

def vag_split(alldata,raito = 0.1):
    random.shuffle(alldata)
    size = len(alldata)
    test_size = int(raito*size)
    test_data = alldata[0:test_size]
    train_data = alldata[test_size:]
    return train_data,test_data


def sc_split(alldata,raito = 0.1):
    # test_data = glob('6/5/*/*png')+glob('6/6/*/*png')+glob('6/7/*/*png')+glob('6/8/*/*png')+glob('6/9/*/*png')+glob('6/10/*/*png')

    alldata = list_sub_xing_check(alldata)
    random.shuffle(alldata)
    size = len(alldata)
    test_split_size = int(size * raito)
    test_data = alldata[0:test_split_size]
    train_data = alldata[test_split_size:]
    return train_data,test_data

def short2picpaths(short='6 - 4 - 0'):
    numbs = short.split('-')
    numbs = [w.strip() for w in numbs]
    picpaths = glob(numbs[0]+'/'+ numbs[1]+ '/'+'image_'+numbs[2]+'_1_240_320/*png')
    picpaths = list_sub_xing_check(picpaths)
    return picpaths
def huanghe_split(alldata,raito = 1.0):
    huanghe_fails = short2picpaths('6 - 4 - 0')+short2picpaths('6 - 4 - 2')+short2picpaths('6 - 4 - 4')+short2picpaths('6 - 5 - 2')+short2picpaths('6 - 5 - 4')+short2picpaths('6 - 6 - 0')+short2picpaths('6 - 7 - 5')
    test_data = list_sub_xing_check(huanghe_fails)
    #6 - 4 - 0
    #6 - 4 - 2
    #6 - 4 - 4
    #6 - 5 - 2
    #6 - 5 - 4
    #6 - 6 - 0
    #6 - 7 - 5
    alldata = list_sub_xing_check(alldata)
    random.shuffle(test_data)
    train_data = list_sub_list(alldata,test_data)
    size = len(test_data)
    test_split_size = int(size*raito)
    test_data = test_data[0:test_split_size]
    train_data = train_data + test_data[test_split_size:]
    return train_data,test_data

datas = glob('/Users/liguiming/Desktop/work/移远/sample/5littler/decoder6/*.yuv')
random.shuffle(datas)

train_data,test_data = vag_split(datas)
# _,test_data = huanghe_split(datas)

# train_data,test_data = vag_split(datas)
# train_data,test_data = sc_split(datas)
## label regular 字符值_序号_宽_高
# label_data = '6/label.txt'
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%*'
tag2chars = {idx:w for idx,w in enumerate(chars)}
chars2tag = { char:tag for tag,char in tag2chars.items()}
chars_no = len(chars)
# newsize=(50,50) # w,h
# min_area = 400
input_size = 50

# label_map = {}
# # with open(label_data,'r',encoding='utf-8') as f:
# #     labels = f.readlines()
# labels = labels[0].strip()
# labels = '**'+labels+'*'
def picname2tag(picname='1/32_956_27_50.yuv'):
    name = picname.split('/')[-1].split('.')[0]
    pices = name.split('_')
    assert len(pices)==4
    tag = int(pices[0])
    w = int(pices[2])
    h = int(pices[3])
    return tag,w,h

train_transforms= transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def gene_idx(size,exclude=0):
    while True:
        idx = random.randint(0,size)
        if idx != exclude:
            return idx


enhance_for_img2_transforms = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)## new add
class traincropset(Dataset):
    def __init__(self, stage = None,transform = None,loader=base_loader,enhance_for_img2_transforms=enhance_for_img2_transforms):
        self.transform = transform
        self.loader = loader
        if stage=='train':
            self.file = train_data
            # self.labels_file = labels_file
            # self.target = train_target
            self.enhance_for_img2_transforms=enhance_for_img2_transforms
        else:
            self.file = test_data
            # self.labels_file = labels_file
            # self.target = test_target
            self.enhance_for_img2_transforms=None
        self.size = len(self.file)
    def __getitem__(self, index):## same is 0 else 1
        fn = self.file[index]
        tag,w,h = picname2tag(fn)
        # label = fn.split('/')[-1].split('_')[0]
        fn_pos = 'labels/'+str(tag)+'.png'
        idx_passes = [w for w in range(chars_no) if w != tag]
        np.random.shuffle(idx_passes)
        idx_pass = idx_passes[0]
        fn_pass = 'labels/'+str(idx_pass)+'.png'
        if index%4 == 0 :
            target = 1

            img1 = self.loader(fn,w=w,h=h,is_observ=True)
            img2 = self.loader(fn_pos)

        else:
            target = 0
            img1 = self.loader(fn,w=w,h=h,is_observ=True)
            img2 = self.loader(fn_pass,w=w,h=h)

        if self.enhance_for_img2_transforms:
            img1 = self.enhance_for_img2_transforms(img1)
            img2 = self.enhance_for_img2_transforms(img2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1,img2, target
    def __len__(self):
        return len(self.file)
