'''
Trung Vo
CSE 512 - HW2
problem 3.1 - Solving binary classification SVM with QP
'''
import numpy as np
from cvxopt import matrix, solvers
import scipy.io as sio
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

N = np.shape(x_train)[1]      # number of training samples
D = np.shape(x_train)[0]      # number of features
y_train = y_train.T
y_val = y_val.T


# form QP form
V = np.multiply(x_train, y_train)	# [d,n]
H = matrix(V.T.dot(V))				# [d,d] matrix (d is number of features)
f = matrix(-np.ones((N,1)))
C = 10
A = matrix(np.vstack(( -np.eye(N), np.eye(N) )))
b = matrix(np.vstack(( np.zeros((N,1)), C*np.ones((N,1)) )))
Aeq = matrix(y_train.astype(np.double))
beq = matrix(np.zeros((1,1)))
solvers.options['show_progress'] = False
sol = solvers.qp(H, f, A, b, Aeq, beq)
alpha = np.array(sol['x'])			# [N,1]


# compute weights optimal
w = np.dot(V, alpha)		# [D,1]
b = np.mean(y_train - np.dot(w.T, x_train))


# evaluate on validation set
y_pred = np.dot(w.T, x_val) + b
y_pred = 1/(1+np.exp(-y_pred))
y_pred = y_pred[0]
for i in range(len(y_pred)):
	y_pred[i] = 1 if (y_pred[i] > 0.5) else -1;

acc = np.mean(y_pred == y_val)*100
print ('accuracy: {:4f} %'.format(acc))

# loss compute
def compute_loss(W, b, X, y, C):
  N = np.shape(X)[1]
  loss = 0.0
  for i in range(N):
    loss += np.maximum(0, 1 - y[0][i]*(np.dot(W.T, X[:,i])+b) )

  loss *= C
  loss += 0.5*np.sum(W*W)
  return loss


objective_value = compute_loss(w, b, x_train, y_train, C=0.1)
print ('train_loss/objective_value: ', objective_value)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val[0], y_pred)
np.set_printoptions(precision=2)

class_names = [1,-1]
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

