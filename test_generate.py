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
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%*'
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

def write_char(ch=None):
    code_pic = code39(ch, writer=writer)
    fullname = code_pic.save(ch,options={'format':'PNG'})

write_char('1') #funame=12345.png
## exclude * X*
def split_png(fname,idx = 0,exclude=[0,-1,-2]):
    img = Image.open(fname).convert('L')
    img_np = np.asarray(img)
    h,w = img_np.shape
    h_s = h//2
    w_points = img_np[h_s]
    x1,x2 = get_x1_x2(w_points,idx=0)
    img_np = img_np[:,x1:x2]
    img_pic = Image.fromarray(np.uint8(img_np)) # np 2 PIL
    img_pic.save(os.path.basename(fname)+'_split_pil.png',quality=95)
    # ls = diff_list(w_points)


split_png('1.png')