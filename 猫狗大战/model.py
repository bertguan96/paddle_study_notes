#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 模型文件
import paddle.fluid as fluid
import numpy as np 
import paddle as paddle

# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=2, act='softmax')
    return fc

"""
    定义卷积神经网络
"""
def convolutional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input = input, num_filters=32,filter_size=3, stride=1)
    # 第一个池化层,类型使用默认类型
    pool = fluid.layers.pool2d(input= conv1,pool_size=2)
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool, size=2, act='softmax')
    return fc