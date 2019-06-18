import numpy as np
import tensorflow as tf
from YOLO.yolo import config as cfg
#https://github.com/hizhangp/yolo_tensorflow/
slim = tf.contrib.slim

class YOLONet(object):
    '''
    实现网络结构，最终返回7x7x30的一个tensor
    实现损失函数，由四部分组成
    '''

    def __init__(self,is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        #输出tensor的大小7x7x30
        self.output_size = (self.cell_size * self.cell_size) * \
                           (self.num_class + self.boxes_per_cell * 5)
        #每个grid cell的大小
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        #得到一个7x7x2的一个tensor。
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        #前向传播预测结果
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        #如果是训练，还需要计算loss
        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                #每个grid cell预测两个bounding box,bounding box一共有20（类别数）+5（x,y,w,h,confidence）个数据需要预测
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            #记录到日志，保存训练过程以及参数分布图，在tensorboard中显示
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,images,num_outputs,alpha,keep_prob=0.5,is_training=True,scope='yolo'):
        with tf.variable_scope(scope):
            #为tensorflow里面的layer层提供默认的参数。只有用@slim.add_arg_scope修饰过的方法才能使用arg_scope。
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),name = 'pad_1') #原图448x448，上下左右加3padding变为454x454x3
                net = slim.conv2d(net, 64, 7, 2, padding = 'VALID', scope = 'conv_2')   #输入：454x454x3，输出224x224x64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')   #输入：224x224x64，输出112x112x64(论文中第二个tensor)
                net = slim.conv2d(net, 192, 3, scope='conv_4')  #输入：112x112x64，输出：112x112x192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')   #输入：112x112x192，输出56x56x192(论文中第三个tensor)
                net = slim.conv2d(net, 128, 1, scope='conv_6')  #输入：56x56x192，输出56x56x128
                net = slim.conv2d(net, 256, 3, scope='conv_7')  #输入：56x56x128，输出56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_8')  #输入：56x56x256，输出56x56x256
                net = slim.conv2d(net, 512, 3, scope='conv_9')  #输入：56x56x256，输出56x56x512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')  #输入：56x56x512，输出：28x28x512(论文中第四个tensor)
                net = slim.conv2d(net, 256, 1, scope='conv_11') #输入：28x28x512，输出：28x28x256
                net = slim.conv2d(net, 512, 3, scope='conv_12') #输入：28x28x256，输出：28x28x512
                net = slim.conv2d(net, 256, 1, scope='conv_13') #输入：28x28x512，输出：28x28x256
                net = slim.conv2d(net, 512, 3, scope='conv_14') #输入：28x28x256，输出：28x28x512
                net = slim.conv2d(net, 256, 1, scope='conv_15') #输入：28x28x512，输出：28x28x256
                net = slim.conv2d(net, 512, 3, scope='conv_16') #输入：28x28x256，输出：28x28x512
                net = slim.conv2d(net, 256, 1, scope='conv_17') #输入：28x28x512，输出：28x28x256
                net = slim.conv2d(net, 512, 3, scope='conv_18') #输入：28x28x256，输出：28x28x512
                net = slim.conv2d(net, 512, 1, scope='conv_19') #输入：28x28x512，输出：28x28x512
                net = slim.conv2d(net, 1024, 3, scope='conv_20')    #输入：28x28x512，输出：28x28x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')  #输入：28x28x1024，输出：14x14x1024(论文中第五个tensor)
                net = slim.conv2d(net, 512, 1, scope='conv_22') #输入：14x14x1024，输出：14x14x512
                net = slim.conv2d(net, 1024, 3, scope='conv_23')    #输入：14x14x512，输出：14x14x1024
                net = slim.conv2d(net, 512, 1, scope='conv_24')     #输入：14x14x1024，输出：14x14x512
                net = slim.conv2d(net, 1024, 3, scope='conv_25')    #输入：14x14x512，输出：14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_26')    #输入：14x14x1024，输出：14x14x1024
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),name = 'pad_27')   #输入：14x14x1024，输出：16x16x1024
                net = slim.conv2d(net, 1024, 3, 2, padding = 'VALID', scope = 'conv_28')    #输入：16x16x1024，输出：7x7x1024（论文中第六个tensor）
                net = slim.conv2d(net, 1024, 3, scope='conv_29')    #输入：7x7x1024，输出：7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_30')    #输入：7x7x1024，输出：7x7x1024(论文中第七个tensor)
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')  #交换矩阵维度
                net = slim.flatten(net, scope='flat_32')    #输入：7x7x1024，输出：50176个神经元
                net = slim.fully_connected(net, 512, scope='fc_33') #输入：50176个神经元，输出：512个神经元
                net = slim.fully_connected(net, 4096, scope='fc_34')    #输入：512个神经元，输出：4096个神经元（论文中第八个tensor）
                net = slim.dropout(net, keep_prob = keep_prob, is_training = is_training,scope = 'dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn = None, scope = 'fc_36') #输入：4096个神经元，输出：7x7x30=1470个预测值（论文中第九个tensor）
                return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])  #原shape:batch x980，改变后：batch x7x7x20(每个grid cell预测的类别概率score)
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell]) #原shape:batch x98，改变后：batch x7x7x2(每个grid cell的confidence)
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])  #原shape:batch x392，改变后：batch x7x7x2x4(每个bounding box的x,y,w,h)

            response = tf.reshape(
                #label中第一位存的是这个bounding box是否包含物体的confidence。
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(
                #label中二至五位存的是x,y,w,h。
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(
                #每个grid cell预测两个bounding box，所以这里扩充bounding box。box大小变为batch x7x7x2x4
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            #label中后二十位存的是物体类别的概率
            classes = labels[..., 5:]

            #偏移
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])    #batch x7x7x2
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))    #逆时针旋转90度
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,     #x
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,       #y
                 tf.square(predict_boxes[..., 2]),      #w²
                 tf.square(predict_boxes[..., 3])], axis=-1)       #h²

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)    #计算bounding box与ground truth的IOU

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op