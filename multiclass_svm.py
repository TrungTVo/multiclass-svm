'''
Trung Vo
CSE 512 - HW2
problem 3.2 - toydata - Multiclass SVM
'''
import numpy as np
from cvxopt import matrix, solvers
import scipy.io as sio
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
from utils import *

data = sio.loadmat('./hw2data/q3_1_data.mat')
x_train = data['trD']
y_train = data['trLb']
x_val = data['valD']
y_val = data['valLb']
'''
x_train: (1984, 362)
y_train: (362, 1)
x_val: (1984, 367)
y_val: (367, 1)
'''
N = np.shape(x_train)[1]
D = np.shape(x_train)[0]
y_train = y_train.T 		# [1, N]
y_val = y_val.T
y_train[y_train == -1] += 1;
y_val[y_val == -1] += 1;


# feed data to train and evaluate
k = 2
W = np.zeros((D, k))
num_epoch = 2000
C = 0.1
W_optimal, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history = batch_gd(num_epoch, W, x_train, y_train, x_val, y_val, C=C)
print ('Training ends...\n')
print ('Best validation accuracy: {:.4f} %'.format(best_val_acc*100))
print ('Sum of squares of W:', np.sum(W_optimal*W_optimal))

# visualize result
epochs = np.arange(num_epoch)
plt.subplot(121)
plt.plot(epochs, train_loss_history, 'b', label='train')
plt.plot(epochs, val_loss_history, 'g', label='val')
plt.title('loss')
plt.subplot(122)
plt.plot(epochs, train_acc_history, 'b', label='train')
plt.plot(epochs, val_acc_history, 'g', label='val')
plt.title('accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Compute confusion matrix
pred_scores = np.dot(W_optimal.T, x_train)
y_pred = np.argmax(pred_scores, axis=0)
cnf_matrix = confusion_matrix(y_train[0], y_pred)
np.set_printoptions(precision=2)

class_names = [1,-1]
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Unnormalized confusion matrix')

plt.show()

