# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:09:43 2018

@author: 24681
"""
import tensorflow as tf
import glob
import os
from data_utils import load_CIFAR10
from Image import ImageDataGenerator
from tensorflow.contrib.data import Dataset

def resize(x,y):
    one_hot=tf.one_hot(y,10)
    x_resized=tf.image.resize_images(x,[227,227])
    return x_resized,one_hot

def getimage(image,batch_size,trainnum=2000,testnum=500):
    
    train_image=[]
    train_label=[]
    test_image=[]
    test_label=[]
    if image=='FID':   
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
            batch_size=batch_size,
            num_classes=classnum)
        
        # 调用图片生成器，把测试集图片转换成三维数组
        test_data = ImageDataGenerator(
            images=test_image,
            labels=test_label,
            batch_size=batch_size,
            num_classes=classnum,
            shuffle=False)
        tr_data=tr_data.data
        test_data=test_data.data
        return tr_data,test_data,classnum
    if image=='cifar10':
        cifar10_dir = 'cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)#加载cifar数据
        train_image=X_train[list(range(trainnum))]
        train_label=y_train[list(range(trainnum))]
        test_image=X_test[list(range(testnum))]
        test_label=y_test[list(range(testnum))]
        classnum=10
        tr_data = Dataset.from_tensor_slices((train_image,train_label))
        tr_data = tr_data.map(resize)
        tr_data=tr_data.batch(batch_size)
        test_data = Dataset.from_tensor_slices((test_image,test_label))
        test_data = test_data.map(resize)
        test_data=test_data.batch(batch_size)
        return tr_data,test_data,classnum


    

