import os
import shutil

import numpy as np
import cv2


def read_images(input_path, img_height=320, img_width=240):
    """
    解析yuv图像
    """
    images_path = os.path.join(input_path, "src")
    output_path = os.path.join(input_path, "dest")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)
    yuv_images = os.listdir(images_path)
    for yuv_img in yuv_images:
        if yuv_img.endswith("yuv"):
            print(yuv_img)
            img_name = yuv_img.split(".")[0]
            yuv_frame = np.fromfile(os.path.join(images_path, yuv_img), dtype=np.uint8)
            # yuv_frame = yuv_frame.reshape(img_height * 3 // 2, img_width)
            yuv_frame = yuv_frame.reshape( img_height,img_width)
            # img_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV21)
            img_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV420p2GRAY) #COLOR_YUV2GRAY_420  COLOR_YUV2GRAY_NV21
            cv2.imwrite(os.path.join(output_path, img_name + ".jpg"), img_bgr)


if __name__ == "__main__":
    data_path = r"yuv"
    read_images(data_path)


