from barcode.writer import ImageWriter
from barcode.ean import EuropeanArticleNumber13
import barcode

## https://www.cnpython.com/pypi/pybarcode
code_list = barcode.PROVIDED_BARCODES
# print(code_list)
code39 = barcode.get_barcode_class('code39')
code_pic = code39(u'123', writer=ImageWriter())
fullname = code_pic.save('code39_barcode')