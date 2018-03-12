'''
Trung Vo
CSE 512 - HW2
problem 3.2 - UCF101 data (activity recognition)
'''
import numpy as np
import csv
from cvxopt import matrix, solvers
import scipy.io as sio
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
from utils import multiclass_svm_loss, evaluate, num_corrects, random_mini_batches, minibatch_gd, batch_gd

data = sio.loadmat('./hw2data/q3_2_data.mat')
x_train = data['trD']
y_train = data['trLb']
x_val = data['valD']
y_val = data['valLb']
x_test = data['tstD']

# data dimension
'''
x_train:  (4096, 7930)		[D, N]
y_train:  (7930, 1)
x_val:  (4096, 2120)
y_val:  (2120, 1)
x_test: (4096, 3190)
'''

# normalize
mean = np.array([np.mean(x_train, axis=1)]).T
x_train -= mean
x_val -= mean
x_test -= mean

x_train = np.concatenate((x_train, np.ones((1,np.shape(x_train)[1] ))), axis=0)
x_val = np.concatenate((x_val, np.ones((1, np.shape(x_val)[1]))), axis=0)
x_test = np.concatenate((x_test, np.ones((1, np.shape(x_test)[1]))), axis=0)

N = np.shape(x_train)[1]			# number of training examples
D = np.shape(x_train)[0]			# number of features
y_train = y_train.T 				# [1, N]
y_val = y_val.T
y_train -= 1
y_val -= 1

y_train = y_train.astype(int)
y_val = y_val.astype(int)


# feed data to train and evaluate
k = 10
W = np.zeros((D, k))
num_epoch = 3000
C = 10
W_optimal, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history = minibatch_gd(num_epoch, W, x_train, y_train, x_val, y_val, C)
print ('Training ends...\n')
print ('Best validation accuracy: {:.4f} %'.format(best_val_acc*100))

# predict
pred_scores = np.dot(W_optimal.T, x_test)
y_pred = np.array([np.argmax(pred_scores, axis=0)]) + 1
ids = np.array([np.arange(np.shape(x_test)[1]) + 1])
result = np.concatenate((ids, y_pred), axis=0).T
	
# write to file
submissionFile = open('109845485.csv', 'w')
with submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(result)

paramsFile = open('params_3_2.csv', 'w')
with paramsFile:
	writer = csv.writer(paramsFile)
	writer.writerows(W_optimal)    


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
	
