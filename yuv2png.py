import os
import cv2
import numpy as np


def YUVvideo2IMGs(path, savepath, height, width):
    """
    Convert the YUV video to RGB images and save the images at the target folder
    Args:
        file: input yuv file name
        savepath: save path of each RGB images
        height: height of images
        width: width of images

    Returns:

    """
    # 该文件夹下有多个yuv视频
    files = os.listdir(path)
    for file in files:
        img_size = int(width * height * 3 / 2)
        filepath = path + file
        print(filepath)
        size = os.path.getsize(filepath)
        frames = int(size / img_size)
        print(size, ", ", frames)
        dirname = savepath + file[:-4]
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filepath, "rb") as f:
            for frame_idx in range(frames):
                yuv = np.zeros(shape=img_size, dtype='uint8', order='C')
                for j in range(img_size):
                    yuv[j] = ord(f.read(1))
                img = yuv.reshape((height * 3 // 2, width)).astype('uint8')
                # cv2.COLOR_YUV2BGR_NV21：根据自己的yuv转变为rgb格式自行选择修改
                bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV21)
                if bgr_img is not None:
                    print(frame_idx)
                    cv2.imwrite(os.path.join(dirname, "%03d.png" % (frame_idx + 1)), bgr_img)


# 读取yuv420p的一帧文件，并转化为png图片
if __name__ == '__main__':
    filepath = '../视频文件路径/'
    savepath = '/生成的png图像路径/'
    height = 2160
    width = 3840
    YUVvideo2IMGs(filepath, savepath, height, width)
