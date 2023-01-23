# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2023年01月12日
"""


'''     early stopping类     '''
class EarlyStopping():
    def __init__(self, patience=40, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None    # 以准确率之间的最小差异为判断
        self.early_stop = False
    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif self.best_acc - val_acc < self.min_delta:
            self.best_acc = val_acc
            self.counter = 0    # reset counter if validation loss improve
        elif self.best_acc - val_acc >= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True










