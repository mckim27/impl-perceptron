#-*- coding:utf-8 -*-

import csv
import numpy as np
from constant import input_cnt, output_cnt

class DataLoader:

    def __init__(self, path):
        self.data = None
        self.shuffle_map = None
        self.test_begin_idx = None

        with open(path) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)
        #             print(row)

        self.data = np.zeros([len(rows), input_cnt + output_cnt])

        # I, M, F 값은 one hot encoding 으로 표현.
        for n, row in enumerate(rows):
            if row[0] == 'I': self.data[n, 0] = 1
            if row[0] == 'M': self.data[n, 1] = 1
            if row[0] == 'F': self.data[n, 2] = 1
            self.data[n, 3:] = row[1:]


    def print_head1(self):
        print(self.data[0])


    def arrange_data(self, mb_size):
        self.shuffle_map = np.arange(self.data.shape[0])
        np.random.shuffle(self.shuffle_map)
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        self.test_begin_idx = step_count * mb_size
        return step_count


    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
        return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


    def get_train_data(self, mb_size, nth):
        if nth == 0:
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        train_data = self.data[self.shuffle_map[mb_size*nth:mb_size*(nth+1)]]
        return train_data[:, :-output_cnt], train_data[:, -output_cnt:]
