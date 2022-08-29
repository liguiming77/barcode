from barcode.writer import ImageWriter
from barcode.ean import EuropeanArticleNumber13
import barcode

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
def write_char(ch=None):
    code_pic = code39(ch, writer=writer)
    fullname = code_pic.save(ch,'format'='PNG')

write_char('12345')