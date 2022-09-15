from PIL import Image
a=Image.open('labels/1.png').convert('RGB')
# a.show()

a=a.crop( (0,0,a.width,int((a.height*2)/3) )) #x1,y1,x2,y2
a.show()