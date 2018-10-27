import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    weights = np.load("data/vgg16_weights.npz");
    weights0 = np.load("data/VGG_imagenet.npy", encoding='latin1');
    print("打印权重：")
    #keys = sorted(weights.keys())
    for i, k in enumerate(weights.keys()):
        print(i, k, np.shape(weights[k]))