#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import paddle.fluid as fluid


"""
    多模型学习，暂时还没完成。。有空再修改
"""
class Model():

    def __init__(self,class_dim_1,data,label):
        # 第一层预测模型
        self.label1_cnn = self.cnn_net(data,label,class_dim_1,256,3)
        # 第二层预测模型
        self.label2_cnn = self.cnn_net(data,label,class_dim_1,128,3)
        # 第三层预测模型
        self.label3_cnn = self.cnn_net(data,label,class_dim_1,128,3)
        # 模型合并
        concat_list = [self.label1_cnn,self.label2_cnn,self.label3_cnn]
        # 模型连接
        self.model = fluid.layers.concat(input=concat_list,axis=1,name="model_concat")

            # full connect layer
        self.fc1 = fluid.layers.fc(input=self.model, size=128, act='tanh')
        param_attrs = fluid.ParamAttr(name="fc_weight",
                            regularizer=fluid.regularizer.L2DecayRegularizer(0.01))

        self.prediction = fluid.layers.fc(input=self.fc1,size=10,act="softmax",param_attr=param_attrs,name="perdiction")

    
    def get_models(self):
        return self.prediction

    def cnn_net(self,data,
                label,
                dict_dim,
                emb_dim,
                win_size,
                hid_dim=128,
                is_infer=False):
        # embedding layer
        emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
        # convolution layer
        conv_3 = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="relu",
            pool_type="max")
       
        

        return conv_3