# coding=utf-8
from barcode.writer import ImageWriter
from barcode.ean import EuropeanArticleNumber13
import barcode
from PIL import Image
import numpy as np
import os

## https://www.cnpython.com/pypi/pybarcode
code_list = barcode.PROVIDED_BARCODES
# print(code_list) SVG
# writer = ImageWriter(options={'format': 'JPEG'})
# writer = ImageWriter(options={'format': 'SVG'})
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
id2chars = {idx:w for idx,w in enumerate(chars)}
char_list = [w for w in chars]
writer = ImageWriter()
code39 = barcode.get_barcode_class('code39')
# code_pic = code39(u'123', writer=writer)
# fullname = code_pic.save('code39_barcode')
def diff_list(ls):
    ls_fore = ls[0:-1]
    ls_fore=np.insert(ls_fore,0,ls[0])
    # ls_fore.insert(ls[0])
    ls = np.asarray(ls).astype(np.int32)

    ls_fore = np.asarray(ls_fore).astype(np.int32)

    hold_place = np.zeros(ls.size)
    diff = ls - ls_fore
    index_no_equal_0 = np.where(diff!=0)
    hold_place[index_no_equal_0] = 1
    # print(hold_place.sum())
    assert hold_place.sum()%2==0
    # print(hold_place.sum()//2)
    # assert 1>2
    return hold_place

## return 每个条纹对应的右边宽度，条纹所在的像素位置
def get_width_b2b(ls):
    idx_equal_1 = np.where(ls==1)
    idx_equal_1 = idx_equal_1[0]
    idx_fore = idx_equal_1[0:-1]
    idx = idx_equal_1[1:]
    idx_fore = np.asarray(idx_fore)
    idx = np.asarray(idx)
    diff = idx - idx_fore
    # print(diff)
    diff = np.insert(diff,-1,10)
    return diff,idx_equal_1



def get_x1_x2(ls,idx): ## exclude 0,-1,-2
    dst_np = diff_list(ls)
    no_slide1 = (idx+1)*10 - 1
    no_slide2 = no_slide1 + 10
    each_wds,point_black = get_width_b2b(dst_np)
    x1 = point_black[no_slide1]+each_wds[no_slide1]//2
    x2 = point_black[no_slide2] + each_wds[no_slide2] // 2
    return x1,x2
    ### 0,1,3

def write_char(ch,outname):
    code_pic = code39(ch, writer=writer)
    fullname = code_pic.save(outname,options={'format':'PNG'})

# write_char('*','1.png') #funame=12345.png
# assert 1>2
## exclude * X*

def split_png(fname,outname,idx = 0,exclude=[0,-1,-2]):
    img = Image.open(fname).convert('L')
    img_np = np.asarray(img)
    h,w = img_np.shape
    h_s = h//2
    w_points = img_np[h_s]
    x1,x2 = get_x1_x2(w_points,idx=0)
    img_np = img_np[:,x1:x2]
    img_pic = Image.fromarray(np.uint8(img_np)) # np 2 PIL
    img_pic.save(outname,quality=95)
    # ls = diff_list(w_points)


#for idx,char in id2chars.items():
    #write_char(char,'labels/'+str(idx))
#    picpath = 'samples/'+str(idx)+'.png'
#    outpath = 'labels/'+str(idx)+'.png'
#    split_png(picpath,outpath)

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

dest_size=(40,280)
pic1 = np.asarray(Image.open('badslice/5.png').convert('L'))
pic1 = np.resize(pic1,(dest_size[1],dest_size[0]))
# print(pic1.shape)
maxsims = 0.
reschars = None
for idx,char in id2chars.items():
    picpath = 'labels/'+str(idx)+'.png'
    pic2 = np.asarray(Image.open(picpath).convert('L'))
    pic2 = np.resize(pic2, (dest_size[1],dest_size[0]))
    dim1,dim2 = pic2.shape
    choice_no = dim1//3
    pic2 = pic2[choice_no]
    #print(pic1.shape)
    #print(pic2.shape)
    # pic2 = np.transpose(pic2)
    # print(pic2)
    # pic2 = np.resize(pic2,dest_size)

    res = np.asarray(get_cos_similar_multi(pic2,pic1)[0])
    sim = np.average(res)
    print(sim)
    if sim>=maxsims:
        dest_char = char
        maxsims = sim
print(dest_char)
    # assert 1>2



# split_png('1.png')
