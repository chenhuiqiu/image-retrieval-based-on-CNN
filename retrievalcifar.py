# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:27:26 2018

@author: 24681
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from data_utils import load_CIFAR10

from tensorflow.python.framework import dtypes
from tensorflow.contrib.data import Dataset
from Alexnet import AlexNet
from tensorflow.contrib.data import Iterator

cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)#加载cifar数据
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


#取出数据库，resize，生成迭代器为后面输入网络准备
num_test = 10000#数据库的大小
batch_num=100
mask = list(range(num_test))
X_test = X_test[mask]#取出500个样本作为数据库
y_test=y_test[mask]
num_query=100#要查询数量的大小

X_query=X_test[-num_query:]
y_query=y_test[-num_query:]

#X_testensor=convert_to_tensor(X_test,dtype=tf.float32)
#y_testensor=convert_to_tensor(y_test,dtype=tf.int32)#转化为tensor#看来dataset的输入并不用为tensor
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

def resize(x):
    x_resized=tf.image.resize_images(x,[227,227])
    return x_resized

#生成数据库生成器
data = Dataset.from_tensor_slices(X_test)
data=data.map(resize)
data=data.batch(batch_num)

iterator = Iterator.from_structure(data.output_types,
                               data.output_shapes)


testing_initalize=iterator.make_initializer(data)
next_batch = iterator.get_next()


#生成查询图片生成器
q_data = Dataset.from_tensor_slices(X_query)
q_data=q_data.map(resize)
q_data=q_data.batch(num_query)

q_iterator = Iterator.from_structure(q_data.output_types,
                               q_data.output_shapes)


query_initalize=q_iterator.make_initializer(q_data)
query_batch = q_iterator.get_next()
# 数据库通过AlexNet
model = AlexNet(x, 1, 10, skip_layer='')
l = model.fc8
fc8=model.fc8
saver = tf.train.Saver()

epoch=int(num_test/batch_num)#因为batch最大为600，所以要分代输入
test_feature=np.empty(shape=[0, 10])

with tf.Session() as sess:
    sess.run(testing_initalize)
    sess.run(query_initalize)    
    sess.run(tf.global_variables_initializer())
    
    query_batch=sess.run(query_batch)
    saver.restore(sess, "./tmp/cifarcheckpoints/cifarmodel_epoch20.ckpt") # 导入训练好的参数
    for i in range(epoch):
        img_batch=sess.run(next_batch)
        feature=sess.run(l, feed_dict={x: img_batch})#数据库的中间层特征
        test_feature=np.append(test_feature,feature,axis=0)
    query_feature=sess.run(l,feed_dict={x:query_batch})#测试图片的中间层特征
    query_fc8=sess.run(fc8,feed_dict={x:query_batch})
    


precnum=0
ap=0    
#计算l2距离
for i in range(20):#对每一张要查询图片进行检索
    dists = np.zeros((1, test_feature.shape[0]))
    for j in range(test_feature.shape[0]):
        dist=np.sqrt(np.sum(np.square(test_feature[j,:]-query_feature[i,:])))
        dists[:,j]=dist
    
    #显示要检索的图片
    score=np.argmax(query_fc8,1)
    panding=classes[score[i]]
    plt.figure()
    plt.imshow(X_query[i].astype('uint8'))
    trueclass=classes[y_query[i]]
    plt.title('Class:%s'% (trueclass))
    plt.show()
    
    indexes = np.argsort(dists)
    k=10#检索个数
    tclass=y_test[indexes[0,0:k]]
    
    p=0
    plt.figure()
    for i,j in enumerate(tclass):
        plt.subplot(1, k, i+1)
        plt.imshow(X_test[indexes[0,i]].astype('uint8'))
        cls=classes[j]
        if cls==trueclass:
            p=p+1
            ap=ap+p/(i+1)
        plt.title(cls)
        plt.axis('off')
    plt.show()
    precnum=precnum+p

pk=precnum/(num_query*k)
mAP=ap/(num_query*k)

a=0
for i in range(len(score)):
    if score[i]==y_query[i]:
        a=a+1
    
acc=a/len(score)