# -*- coding: utf-8 -*-
import logging
import sys
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_std_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    return logging

class log_tool():
    def __init__(self, bucket=None, load_old=True, log_path=''):
        self.step = []
        self.value = []
        self.bucket = bucket
        self.log_path = log_path
        if not (bucket is None or log_path == '') and load_old:
            self.load_log()

    def update(self, s, v):
        self.step.append(s)
        self.value.append(v)

    def sample(self, sample_num = 100):
        step_list = np.argsort(np.array(self.step))
        value_list = np.array(self.value)[step_list]
        if len(self.step) > sample_num:
            self.x = np.array(self.step)[::len(step_list) // sample_num].tolist()
            self.y = value_list[::len(step_list) // sample_num].tolist()
        else:
            self.x = np.array(self.step)[step_list].tolist()
            self.y = value_list.tolist()

    def plot(self, plot_path, sample_num = 100, label = '', x_label = 'iter', y_label = 'loss'):
        self.sample(sample_num=sample_num)
        plt.ioff()
        fig = plt.figure(figsize=(12, 10), dpi=180)
        style = 'r*-'
        plt.plot(self.x, self.y, style, label = f'{label}_last:{self.y[-1]:.2f}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(np.arange(0, max(self.x)+0.1, max(self.x)/10))
        plt.yticks(np.arange(-0.1, max(self.y)+0.1, max(self.y)/10))
        plt.grid()
        plt.legend(loc=4)
        buf = BytesIO()
        plt.savefig(buf)
        plt.clf()
        plt.cla()
        plt.close()
        try:
            self.bucket.put_object(plot_path, buf.getvalue())
        except Exception as e:
            print(e)

    def load_log(self):
        try:
            lines = str(self.bucket.get_object(self.log_path).read(), 'utf-8').split('\n')
            for line in lines:
                if line == '':
                    continue
                s, v = line.split(' ')
                self.step.append(int(s))
                self.value.append(float(v))
        except Exception as e:
            print(e)
            pass

    def save_log(self):
        if not (self.bucket is None or self.log_path == ''):
            fp = BytesIO()
            for idx, s in enumerate(self.step):
                line = f'{s} {self.value[idx]}\n'
                fp.write(line.encode('utf-8'))
            try:
                self.bucket.put_object(self.log_path, fp.getvalue())
            except Exception as e:
                print(e)
