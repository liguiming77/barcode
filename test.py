# -*- coding: utf-8 -*-
from torchvision import transforms
import torch
from fnet import barmodel
from PIL import Image
from tools import default_loader
from glob import glob
input_size = 50
newsize=(50,50)
label_num =44
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


