from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np

from glob import glob
from numpy import random
from tools import default_loader
# from tools import CustomRandomResizedCrop
# base_path = '/mdata/data2/data2/pics/'
# badslice_file = glob('badslice/*png')*100
# labels_file = glob('labels/*png')
# random.shuffle(badslice_file)
# train_target = random.randint(0,2,size=len(train_file))
# test_file = glob( base_path+ 'val/饮料瓶/*1.jpg')*5 + glob( base_path+ 'val/玻璃制品/*1.jpg')*10 + glob( base_path+ 'val/其他非回收/*1.jpg')*10 + glob( base_path+ 'val/疑似非回收/*1.jpg')*2 + glob( base_path+ 'val/医疗废品/*1.jpg')*1000 + glob( base_path+ 'val/有害垃圾/*1.jpg')*100 + glob( base_path+ 'val/402/*1.jpg')*100 + glob( base_path+ 'val/干湿垃圾/*1.jpg')*5 + glob( base_path+ 'val/摄像头装反/*1.jpg')*10 + glob( base_path+ 'val/其他塑料/*1.jpg') + glob( base_path+ 'val/建筑垃圾/*1.jpg')*20 + glob( base_path+ 'val/黑屏/*1.jpg')*20 + glob( base_path+ 'val/违法物品/*1.jpg')*10000 + glob( base_path+ 'val/其他织物/*1.jpg') + glob( base_path+ 'val/其他金属/*1.jpg')*10 + glob( base_path+ 'val/感光度低/*1.jpg')*10 + glob( base_path+ 'val/纸类/*1.jpg')+glob( base_path+ 'val/绿屏/*1.jpg')*10 + glob( base_path+ 'val/鞋子/*1.jpg')*30 + glob( base_path+ 'val/可回收物/*1.jpg')*5 + glob( base_path+ 'val/小家电/*1.jpg')*10 + glob( base_path+ 'val/易拉罐/*1.jpg')*100 + glob( base_path+ 'val/角度需调整/*1.jpg')*10 + glob( base_path+ 'val/危险物品/*1.jpg')*500 + glob( base_path+ 'val/403/*1.jpg')*500 + glob( base_path+ 'val/其他/*1.jpg')*50 + glob( base_path+ 'val/208/*1.jpg')*100 + glob( base_path+ 'val/碎玻璃/*1.jpg')*1000 + glob( base_path+ 'val/衣服/*1.jpg')*500
# test_target = random.randint(0,2,size=len(test_file))
# name2tag={'208': 0, '402': 1, '403': 2, '其他': 3, '其他塑料': 4, '其他织物': 5, '其他金属': 6, '其他非回收': 7, '医疗废品': 8, '危险物品': 9, '可回收物': 10, '小家电': 11, '干湿垃圾': 12, '建筑垃圾': 13, '感光度低': 14, '摄像头装反': 15, '易拉罐': 16, '有害垃圾': 17, '玻璃制品': 18, '疑似非回收': 19, '碎玻璃': 20, '纸类': 21, '绿屏': 22, '衣服': 23, '角度需调整': 24, '违法物品': 25, '鞋子': 26, '饮料瓶': 27, '黑屏': 28}
# tag2name = {v:k for k,v in name2tag.items()}
def list_sub_list(list1,list2):
    return [w for w in list1 if w not in list2]
def list_sub_xing_check(list1):
    return  [w for w in list1 if '/1.png' not in w and '/2.png' not in w and '/9.png' not in w]

def vag_split(alldata,raito = 1.0):
    # test_data = glob('6/6/*/*png')+glob('6/7/*/*png')+glob('6/8/*/*png')+glob('6/9/*/*png')+glob('6/10/*/*png')
    test_data =  glob('6/10/*/*png')
    alldata = list_sub_xing_check(alldata)
    test_data = list_sub_xing_check(test_data)

    random.shuffle(test_data)
    train_data = list_sub_list(alldata,test_data)
    size = len(test_data)
    test_split_size = int(size*raito)
    test_data = test_data[0:test_split_size]
    train_data = train_data + test_data[test_split_size:]
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

datas = glob('6/*/*/*.png')
random.shuffle(datas)

train_data,_ = vag_split(datas)
_,test_data = huanghe_split(datas)

# train_data,test_data = vag_split(datas)
# train_data,test_data = sc_split(datas)

label_data = '6/label.txt'
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
tag2chars = {idx:w for idx,w in enumerate(chars)}
chars2tag = { char:tag for tag,char in tag2chars.items()}
# newsize=(50,50) # w,h
# min_area = 400
input_size = 50

label_map = {}
with open(label_data,'r',encoding='utf-8') as f:
    labels = f.readlines()
labels = labels[0].strip()
labels = '**'+labels+'*'
def picname2tag(picname='1/1.png'):
    name = picname.split('/')[-1].split('.')[0]
    idx_name = int(name)
    label = labels[idx_name-1]
    if label == '*':
        assert 1>2
    tag = chars2tag[label]
    return tag

train_tag_set = [10,11,1,2,41,42]

# SCALE=(0.05,0.5)#(0.005,0.05) #(0.002,0.1)
# w_div_h = 0.2
# RATIO=(w_div_h,1/w_div_h)


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
#
# def default_loader(path,is_observ=False):
#     img_pil =  Image.open(path).convert("RGB")
#     if not is_observ:
#         img_pil = img_pil.crop( (0,0,img_pil.width, int( (img_pil.height*2)/3 )  )  )
#     # img_pil = Image.open(path).convert("L")
#     # img_np = np.asarray(img_pil)
#     # img_np = np.resize(img_np,newsize)
#     # img_np = np.expand_dims(img_np,-1) ## 增加一个纬度
#
#     img_pil = img_pil.resize(newsize)
#     # img_tensor = preprocess(img_pil)
#     if is_observ:
#         img_pil=img_pil.transpose(Image.ROTATE_270) ## totate 270
#     return img_pil

# def default_loader(path):
#     img_pil = Image.open(path).convert("L")
#     img_np = np.asarray(img_pil)
#     img_np = np.resize(img_np,newsize)
#     img_np = np.expand_dims(img_np,-1) ## 增加一个纬度
#     return img_np



def gene_idx(size,exclude=0):
    while True:
        idx = random.randint(0,size)
        if idx != exclude:
            return idx


enhance_for_img2_transforms = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)## new add
# train_crop_transform = CustomRandomResizedCrop(size=(newsize[1],newsize[0]),scale=SCALE,ratio=RATIO )
class traincropset(Dataset):
    def __init__(self, stage = None,transform = None,loader=default_loader,enhance_for_img2_transforms=enhance_for_img2_transforms):
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
        tag = picname2tag(fn)
        # label = fn.split('/')[-1].split('_')[0]
        fn_pos = 'labels/'+str(tag)+'.png'
        idx_passes = [w for w in range(43) if w != tag]
        np.random.shuffle(idx_passes)
        idx_pass = idx_passes[0]
        fn_pass = 'labels/'+str(idx_pass)+'.png'
        idx_passes_fakes = [w for w in train_tag_set if w != tag]
        np.random.shuffle(idx_passes_fakes)
        idx_passes_fake = idx_passes_fakes[0]
        fn_pass_fake = 'labels/' + str(idx_passes_fake) + '.png'
        if index%4 == 0 :
            target = 1
            img1 = self.loader(fn,is_observ=True)
            img2 = self.loader(fn_pos)
        elif index%4 == 1 or index%4 == 2:
            target = 0
            img1 = self.loader(fn,is_observ=True)
            img2 = self.loader(fn_pass_fake)
        else:
            target = 0
            img1 = self.loader(fn,is_observ=True)
            img2 = self.loader(fn_pass)

        if self.enhance_for_img2_transforms:
            img1 = self.enhance_for_img2_transforms(img1)
            img2 = self.enhance_for_img2_transforms(img2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1,img2, target
    def __len__(self):
        return len(self.file)

class evalset(Dataset):
    def __init__(self, stage = None,transform = None,loader=default_loader,enhance_for_img2_transforms=enhance_for_img2_transforms):
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
        tag = picname2tag(fn)
        # label = fn.split('/')[-1].split('_')[0]
        fn_pos = 'labels/'+str(tag)+'.png'
        idx_passes = [w for w in range(43) if w != tag]
        np.random.shuffle(idx_passes)
        idx_pass = idx_passes[0]
        fn_pass = 'labels/'+str(idx_pass)+'.png'
        idx_passes_fakes = [w for w in train_tag_set if w != tag]
        np.random.shuffle(idx_passes_fakes)
        idx_passes_fake = idx_passes_fakes[0]
        fn_pass_fake = 'labels/' + str(idx_passes_fake) + '.png'
        fn_out = ''
        if index%2 == 0:
            target = 1
            img1 = self.loader(fn)
            img2 = self.loader(fn_pos)
            fn_out = fn_pos
        elif index%4 == 1:
            target = 0
            img1 = self.loader(fn)
            img2 = self.loader(fn_pass_fake)
            fn_out = fn_pass_fake
        else:
            target = 0
            img1 = self.loader(fn)
            img2 = self.loader(fn_pass)
            fn_out = fn_pass

        if self.enhance_for_img2_transforms:
            img1 = self.enhance_for_img2_transforms(img1)
            img2 = self.enhance_for_img2_transforms(img2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1,img2, target,fn,fn_out
    def __len__(self):
        return len(self.file)
# filter_size_w_h = (640*0.05,480*0.05)
# class cropset(Dataset):
#     def __init__(self, img_src = None,img_dst = None,crop_transform = None,loader=default_loader,transform=None,times=500):
#         #定义好 image 的路径
#         self.img_src = img_src
#         self.img_dst = img_dst
#         self.crop_transform = crop_transform
#         self.loader=loader
#         self.transform = transform
#         self.times = times
#
#     def __getitem__(self, index):## same is 0 else 1
#         img_src = self.loader(self.img_src)
#         img_dst = self.loader(self.img_dst)
#         img_crop,zone =  self.crop_transform(img_dst) #zone [i, j, h, w] ## y1,x1,h,w
#         img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_crop)), axis=0)))
#         img = self.transform(img)
#         del img_src
#         del img_dst
#         del img_crop
#         return img,zone  #zone [i, j, h, w] ## y1,x1,h,w
#
#     def __len__(self):
#         return self.times
# sample_train = []
# sample_val = []
# for _ in range(60):
#     for index,v in enumerate(train_data):
#         fn = train_data[index]
#         tag = picname2tag(fn)
#         # label = fn.split('/')[-1].split('_')[0]
#         fn_pos = 'labels/'+str(tag)+'.png'
#         idx_passes = [w for w in range(43) if w != tag]
#         np.random.shuffle(idx_passes)
#         idx_pass = idx_passes[0]
#         fn_pass = 'labels/'+str(idx_pass)+'.png'
#         idx_passes_fakes = [w for w in train_tag_set if w != tag]
#         np.random.shuffle(idx_passes_fakes)
#         idx_passes_fake = idx_passes_fakes[0]
#         fn_pass_fake = 'labels/' + str(idx_passes_fake) + '.png'
#
#         if index%8 == 0:
#             # continue
#             target = 1
#             f1 = fn
#             f2 = fn_pos
#         elif index%8<6:
#             target = 0
#             f1 = fn
#
#             f2 = 'labels/' + str(idx_passes_fakes[index%8 - 1] ) + '.png'
#         else:
#             # continue
#             target = 0
#             f1 = fn
#             f2 = fn_pass
#         sample_train.append([str(target),f1,f2])
# for _ in range(2):
#     for index,v in enumerate(test_data):
#         fn = test_data[index]
#         tag = picname2tag(fn)
#         # label = fn.split('/')[-1].split('_')[0]
#         fn_pos = 'labels/'+str(tag)+'.png'
#         idx_passes = [w for w in range(43) if w != tag]
#         np.random.shuffle(idx_passes)
#         idx_pass = idx_passes[0]
#         fn_pass = 'labels/'+str(idx_pass)+'.png'
#         idx_passes_fakes = [w for w in train_tag_set if w != tag]
#         np.random.shuffle(idx_passes_fakes)
#         idx_passes_fake = idx_passes_fakes[0]
#         fn_pass_fake = 'labels/' + str(idx_passes_fake) + '.png'
#
#         if index%4 == 0:
#             # continue
#             target = 1
#             f1 = fn
#             f2 = fn_pos
#         elif index%4 == 1:
#             target = 0
#             f1 = fn
#             f2 = fn_pass_fake
#         else:
#             # continue
#             target = 0
#             f1 = fn
#             f2 = fn_pass
#         sample_val.append([str(target),f1,f2])
# class traincropset(Dataset):
#     def __init__(self, stage = None,transform = None,loader=default_loader,enhance_for_img2_transforms=enhance_for_img2_transforms):
#         self.transform = transform
#         self.loader = loader
#         if stage=='train':
#             self.file = sample_train
#             # self.labels_file = labels_file
#             # self.target = train_target
#             self.enhance_for_img2_transforms=enhance_for_img2_transforms
#         else:
#             self.file = sample_val
#             # self.labels_file = labels_file
#             # self.target = test_target
#             self.enhance_for_img2_transforms=None
#         self.size = len(self.file)
#     def __getitem__(self, index):## same is 0 else 1
#         fn = self.file[index][1]
#         fn_2 = self.file[index][2]
#         target = int(self.file[index][0])
#
#         img1 = self.loader(fn,is_observ=True)
#         img2 = self.loader(fn_2)
#
#         if self.enhance_for_img2_transforms:
#             img1 = self.enhance_for_img2_transforms(img1)
#             img2 = self.enhance_for_img2_transforms(img2)
#         img1 = self.transform(img1)
#         img2 = self.transform(img2)
#         return img1,img2, target
#     def __len__(self):
#         return len(self.file)