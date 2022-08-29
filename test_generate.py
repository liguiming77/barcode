from barcode.writer import ImageWriter
from barcode.ean import EuropeanArticleNumber13
import barcode
from PIL import Image
import numpy as np

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
    ls_fore = ls[1:]
    ls_fore=np.insert(ls_fore,0,ls[0])
    # ls_fore.insert(ls[0])
    ls = np.asarray(ls).astype(np.int32)
    ls_fore = np.asarray(ls_fore).astype(np.int32)
    hold_place = np.zeros(ls.size)
    diff = ls - ls_fore
    print(type(diff))
    index_no_equal_0 = np.where(diff!=0)
    hold_place[index_no_equal_0] = 1
    print(hold_place.sum())
    assert hold_place.sum()%2==0
    print(hold_place.sum()//2)

def write_char(ch=None):
    code_pic = code39(ch, writer=writer)
    fullname = code_pic.save(ch,options={'format':'PNG'})

write_char('123456789') #funame=12345.png
def split_png(fname):
    img = Image.open(fname).convert('L')
    img_np = np.asarray(img)
    h,w = img_np.shape
    h_s = h//2
    w_points = img_np[h_s]
    diff_list(w_points)


split_png('123456789.png')