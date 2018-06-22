# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:12:47 2018

@author: 24681
"""

import os
import numpy as np
import tensorflow as tf
from Alexnet import AlexNet
from Image import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.contrib.data import Iterator
from getimage import getimage

learning_rate = 1e-4
num_epochs = 30  # 代的个数
batch_size = 100
dropout_rate = 0.5
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 20

filewriter_path = "./tmp/100bcifatensorboard"  # 存储tensorboard文件
checkpoint_path = "./tmp/100bcifarcheckpoints"  # 训练好的模型和参数存放目录

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

trainnum=10000
testnum=500
(tr_data,test_data,num_classes)=getimage('cifar10',batch_size,trainnum,testnum)



with tf.name_scope('input'):
    # 定义迭代器
    iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)

    training_initalize=iterator.make_initializer(tr_data)
    testing_initalize=iterator.make_initializer(test_data)

    # 定义每次迭代的数据
    next_batch = iterator.get_next()

x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 图片数据通过AlexNet网络处理
model = AlexNet(x, keep_prob, num_classes, train_layers)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# 执行整个网络图
score = model.fc8

with tf.name_scope('loss'):
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                              labels=y))

gradients = tf.gradients(loss, var_list)

gradients = list(zip(gradients, var_list))

with tf.name_scope('optimizer'):
    # 优化器，采用梯度下降算法进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)



# 定义网络精确度
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 把精确度加入到Tensorboard
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

# 定义一代的迭代次数
train_batches_per_epoch = int(np.floor(10000 / batch_size))
test_batches_per_epoch = int(np.floor(500/ batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 把模型图加入Tensorboard
    writer.add_graph(sess.graph)

    # 把训练好的权重加入未训练的网络中
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # 总共训练epochs代
    for epoch in range(num_epochs):
        sess.run(training_initalize)
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))

        #开始训练每一代
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

        # 测试模型精确度
        print("{} Start validation".format(datetime.now()))
        sess.run(testing_initalize)
        test_acc = 0.
        test_count = 0

        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.0})
    
            test_acc += acc
            test_count += 1

        test_acc /= test_count

        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # 把训练好的模型存储起来
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))