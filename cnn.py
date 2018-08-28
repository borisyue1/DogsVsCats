import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # for resizing arrays
from random import shuffle
from tqdm import tqdm # for visualizing loops (progress bar)
import PIL # for image processing (resizing, etc.)
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


TRAIN_DIR = "../input/train"
TEST_DIR = "../input/test"
IMG_SIZE = 70
LR = 1e-3
CHANNELS = 3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv')

# assign vector depending on name of image
def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat': return [1,0]
    else: return [0,1]


# resize image
def resize_image(img):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = IMG_SIZE
        n_x_new = int(IMG_SIZE * n_x / n_y + 0.5)
    else:
        n_x_new = IMG_SIZE
        n_y_new = int(IMG_SIZE * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    #create new image of (size, size) with color (0,0,0) - black
    img_pad = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    ulc = ((IMG_SIZE - n_x_new) // 2, (IMG_SIZE - n_y_new) // 2) #borders
    img_pad.paste(img_res, ulc) #insert img_res into img_pad with padding-ulc (if necessary)
    return img_pad


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = Image.open(path)
#         img = cv2.imread(path,cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
#         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
#         img = norm_image(img)
        img = resize_image(img)
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0] # names different in test data
        img = Image.open(path)
#         img = cv2.imread(path,cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
#         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
#         img = norm_image(img)
        img = resize_image(img)
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    # np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, CHANNELS], name='input')
# 5x5 windows for convolution and pooling layers
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)


# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')

def fit_model():
    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,CHANNELS)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,CHANNELS)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    # run_id used for tensorboard
    model.save(MODEL_NAME)


fit_model()

test_data = process_test_data()
# test_data = np.load('test_data.npy')

table_data = {"id": [], "label": []}
for data in tqdm(test_data):
    img_num = data[1]
    img_data = data[0]
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,CHANNELS)
    model_out = model.predict([data])[0]
    table_data["id"].append(img_num)
    table_data["label"].append(model_out[1])

df = pd.DataFrame(table_data["label"], table_data["id"], columns=["label"])
df.to_csv("submission.csv", index_label=["id"])