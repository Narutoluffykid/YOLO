import os

#
# path and dataset parameter
#

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']    #物体类别

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448    #输入yolo网络图片大小

CELL_SIZE = 7       #grid cell网格大小，将图片分成7x7小网格

BOXES_PER_CELL = 2  #每一个grid cell预测多少个bounding box

ALPHA = 0.1

DISP_CONSOLE = False

#loss权重系数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001  #基础学习率

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 20         #一次迭代45张图片

MAX_ITER = 1000        #最大迭代次数

SUMMARY_ITER = 10       #每迭代N次输出一次损失信息

SAVE_ITER = 100        #每迭代N次保存一次模型


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5