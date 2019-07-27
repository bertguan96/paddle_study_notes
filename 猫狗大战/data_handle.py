#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 数据处理
import shutil  # 复制文件
import os
from PIL import Image
import numpy as np
import image_reader
import imghdr
"""
    0 猫
    1 狗
"""
label = [1,0]
#源地址
dog_origin_path = "E:\\dataSet\\dog_and_cat\\origin\\Dog"
cat_origin_path = "E:\\dataSet\\dog_and_cat\\origin\\Cat"
# 训练地址
dog_train_path = "E:\\dataSet\\dog_and_cat\\train\\Dog"
cat_train_path = "E:\\dataSet\\dog_and_cat\\train\\Cat"

# 删除不是JPEG或者PNG格式的图片
def delete_error_image(father_path):
    print(father_path)
    # 获取父级目录的所有文件以及文件夹
    try:
        image_dirs = os.listdir(father_path)
        for image_dir in image_dirs:
            image_dir = os.path.join(father_path, image_dir)
            # 如果是文件夹就继续获取文件夹中的图片
            if os.path.isdir(image_dir):
                images = os.listdir(image_dir)
                for image in images:
                    image = os.path.join(image_dir, image)
                    try:
                        # 获取图片的类型
                        image_type = imghdr.what(image)
                        # 如果图片格式不是JPEG同时也不是PNG就删除图片
                        if image_type is not 'jpeg' and image_type is not 'png':
                            os.remove(image)
                            print('已删除：%s' % image)
                            continue
                        # 删除灰度图
                        img = np.array(Image.open(image))
                        if len(img.shape) is 2:
                            os.remove(image)
                            print('已删除：%s' % image)
                        # 如果图片大小为0(因为猫狗大战的数据集里面存在大小为0B的图片)
                        if img.size == (0,0):
                            os.remove(image)
                            print('已删除：%s' % image)
                    except:
                        os.remove(image)
                        print('已删除：%s' % image)
    except:
        pass
"""
    加载模块完成，测试数据和训练数据的划分
"""
def __load_data_set():
    srcDog = os.listdir(dog_origin_path)
    # 读取前400张图片复制到训练集中
    fnames = ['{}.jpg'.format(i) for i in range(0,1000)]
    i = 0
    for fname in fnames:
        src = os.path.join(dog_origin_path, srcDog[i])
        dst = os.path.join(dog_train_path, "dog." + fname)
        shutil.copyfile(src, dst)
        i+=1
    srcCat = os.listdir(cat_origin_path)
    j = 0
    for fname in fnames:
        src = os.path.join(cat_origin_path, srcCat[j])
        dst = os.path.join(cat_train_path, "cat." + fname)
        shutil.copyfile(src, dst)
        j+=1
    print('total training cat images:',len(os.listdir(cat_train_path)))
    print('total training dog images:',len(os.listdir(dog_train_path)))

# 预处理图片
def load_image(file):
    img = Image.open(file)  
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    print("the file path is " + str(file) + "，the size is " + str(img.size))
    img = img.convert("RGB")
    img.save(file)


# __load_data_set()

def pre_img():
    srcDog = os.listdir(dog_origin_path)
    for srcName in srcDog:
        load_image(dog_origin_path + "\\" + srcName)


    srcCat = os.listdir(cat_origin_path)
    for srcName in srcCat:
        load_image(cat_origin_path + "\\" + srcName)

if __name__ == "__main__":
    # 删除非法图片
    # delete_error_image("E:\\dataSet\\dog_and_cat\\origin")
    # print("非法图片删除完成！")
    # 预处理图片
    # pre_img()
    # print("数据预处理完成！")
    # 生成数据集
    __load_data_set()
    print("数据集生成成功！")