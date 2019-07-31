#-*- coding:utf-8 -*-
import numpy as np
from constant import input_cnt, output_cnt, RND_MEAN, RND_STD, LEARNING_RATE

class Perceptron:
    '''
    loss func = square(y - y`)

    model param derivative  = dL/d(w, b)
        dL/dy * dy/dw = 2(y - y`) * x
            dL/dy = d(square(y - y`))/dy = 2(y - y`) * d(y - y`)/dy
                  = 2(y - y`) * 1
            dy/dw = d(wx + b)/dw = x

        dL/dy * dy/db = 2(y - y`) * 1
            dL/dy = d(square(y - y`))/dy = 2(y - y`) * d(y - y`)/dy
                  = 2(y - y`) * 1
            dy/db = d(wx + b)/db = l

    sgd -> (w,b) - lr/mb_size * sigma dL/d(w, b)
        w - lr/mb_size * sigma 2(y - y`) * x
        b - lr/mb_size * sigma 2(y - y`) * 1
    '''
    def __init__(self):
        self.weight = np.random.normal(RND_MEAN, RND_STD,[input_cnt, output_cnt])
        self.bias = np.zeros([output_cnt])

    def forward_neuralnet(self, x):
        output = np.matmul(x, self.weight) + self.bias
        return output, x

    def backprop_neuralnet(self, G_output, x):

        g_output_w = x.transpose()

        G_w = np.matmul(g_output_w, G_output)
        G_b = np.sum(G_output, axis=0)

        self.weight -= LEARNING_RATE * G_w
        self.bias -= LEARNING_RATE * G_b

    def forward_postproc(self, output, y):
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        return loss, diff

    def backprop_postproc(self, G_loss, diff):

        shape = diff.shape
        # print(shape)
        # print(np.prod(shape))
        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff

        return G_output