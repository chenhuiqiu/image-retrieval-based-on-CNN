# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:35:16 2018

@author: 24681
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from data_utils import load_CIFAR10
from tensorflow.contrib.data import Iterator
import glob
import os
from data_utils import load_CIFAR10
from Image import ImageDataGenerator
from tensorflow.contrib.data import Dataset
from Alexnet import AlexNet

classes=['acerola','apple','apricots','avocado','bannaa','balckberry','blueberry',
         'cataloupes','cherry','coconut','fig','grapefruit','grape','guava','kiwifruit'
         ,'lemon','lime','mango','olive','orange','passionfruit','peaches','pear',
         'pineapple','plum','pomegranates','raspberry','strawberry','tomatoes',
         'watermelon']
train_image=[]
train_label=[]
test_image=[]
test_label=[]
  
image=os.walk(r'D:\360download\FIDS30')
classnum=0
for i in image:
    if i[1]==[]:
        
        imagepath=glob.glob('%s\\*.jpg' %(i[0]))
    
        for i in range(len(imagepath[0:-5])):#取后五张作为测试数据，其余训练
            train_image.append(imagepath[i])
            train_label.append(classnum)
        for i in range(5):
            test_image.append(imagepath[i-6])
            test_label.append(classnum)
        classnum=classnum+1
# 调用图片生成器，把训练集图片转换成三维数组
tr_data = ImageDataGenerator(
    images=train_image,
    labels=train_label,
    batch_size=100,
    num_classes=classnum,
    shuffle=False)

# 调用图片生成器，把测试集图片转换成三维数组
test_data = ImageDataGenerator(
    images=test_image,
    labels=test_label,
    batch_size=len(test_image),
    num_classes=classnum,
    shuffle=False)
tr_data=tr_data.data
test_data=test_data.data

#创建迭代器，向网络传输数据
tr_iterator=Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
tr_initalize=tr_iterator.make_initializer(tr_data)#tr_data作为检索数据库
tr_batch=tr_iterator.get_next()

test_iterator=Iterator.from_structure(test_data.output_types,test_data.output_shapes)
test_initalize=test_iterator.make_initializer(test_data)#test_data作为被检索图像
test_batch=test_iterator.get_next()

#创建模型
x=tf.placeholder(tf.float32,[None,227,227,3])
model = AlexNet(x, 1, classnum, skip_layer='')
l = model.fc8#取第七层作为检索特征
fc8=model.fc8
saver = tf.train.Saver()

epoch=int(len(train_image)/100)+1
feature=np.empty(shape=[0,30])
with tf.Session() as sess:
    sess.run(tr_initalize)
    sess.run(test_initalize)
    sess.run(tf.global_variables_initializer())
    test_batch,label=sess.run(test_batch)
    saver.restore(sess, "./tmp/fruitcheckpoints/model_epoch20.ckpt") # 导入训练好的参数

    query_feature=sess.run(l,feed_dict={x:test_batch})
    query_fc8=sess.run(fc8,feed_dict={x:test_batch})
    for i in range(epoch):
        train_batch,label=sess.run(tr_batch)
        nfeature=sess.run(l,feed_dict={x:train_batch})
        feature=np.append(feature,nfeature,axis=0)
    
    prec=0#检索正确的个数
    ap=0
    #计算l2距离
    for i in range(20):#(query_feature.shape[0]):#对每一张要查询图片进行检索
        print(i)
        dists = np.zeros((1, feature.shape[0]))
        for j in range(feature.shape[0]):
            dist=np.sqrt(np.sum(np.square(feature[j,:]-query_feature[i,:])))
            dists[:,j]=dist
        
        #显示要检索的图片
        query_image=test_image[i]
        img_string = tf.read_file(query_image)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        score=np.argmax(query_fc8,1)
        panding=classes[score[i]]
        trueclass=classes[test_label[i]]
        
        plt.figure()
        plt.imshow(img_decoded.eval(session=sess))
        
        plt.title('Class:%s'% (trueclass))
        plt.show()
            
        indexes = np.argsort(dists)
        k=10#检索个数
        train_label=np.array(train_label)
        tclass=train_label[indexes[0,0:k]]
        
        plt.figure()
        p=0
        for i,j in enumerate(tclass):
            
            plt.subplot(1, k, i+1)
            sql_image=train_image[indexes[0,i]]
            sql_string=tf.read_file(sql_image)
            query_decoded=tf.image.decode_png(sql_string, channels=3)
            
            plt.imshow(query_decoded.eval(session=sess))
            cls=classes[j]
            #计算准确率
            if cls==trueclass:
                p=p+1
                ap=ap+p/(i+1)
#            plt.title(cls)
            plt.axis('off')
        plt.show()   
        
        prec=prec+p
    pk=prec/1500
    mAP=ap/1500

a=0
for i in range(len(score)):
    if score[i]==test_label[i]:
        a=a+1
    
acc=a/len(score)


