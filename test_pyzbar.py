from pyzbar import pyzbar
import matplotlib.pyplot as plt
import cv2
#条形码定位及识别
def decode(image,barcodes):
    #循环监测条形码
    for barcode in barcodes:
        #提取条形码边界框位置
        #画出图中条形码的边界框
        (x,y,w,h)=barcode.rect#获得这个图吗的x,y坐标和宽和高区域
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)#把它框起来用蓝色，线粗5

        #条形码数据为字节对象，所以如果想在输出图像上
        #画出来，就需要先将它装换为字符串
        barcodeData=barcode.data.decode("utf-8")#将barcode的数据识别出来
        barcodeType=barcode.type#类型也直接识别出来了

        #绘制出图像上条形码的数据和条形码的类型
        text="{} ({})".format(barcodeData , barcodeType)
        cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,8,(255,0,0),2)  # cv2.putText(image,text,(x,y-10)
        #像终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode:{}".format(barcodeType,barcodeData))
        plt.figure(figsize=(10,10))
        plt.imshow(image)
    plt.show()

#1,读取条形码图片
image=cv2.imread('9.png')
bacodes=pyzbar.decode(image)#找到图片中的条形码并进行解码
decode(image,bacodes)#识别条形码


# #二维码
# image=cv2.imread('erwei.png')
# bacodes=pyzbar.decode(image)
# decode(image,bacodes)

