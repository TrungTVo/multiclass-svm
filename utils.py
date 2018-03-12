'''
Trung Vo - CSE 512
Homework 2 - Problem 3.2
Multiclass SVM with gradient descent
'''

import numpy as np
from cvxopt import matrix, solvers
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

# vectorized implementation
def multiclass_svm_loss(W, X, y, C):
	d, k = W.shape		# number of features, classes	
	_, N = X.shape
	loss = 0
	dW = np.zeros_like(W) 		# [d,k]	

	Z = np.dot(W.T, X) 			# [k, N]
	correct_class_score = np.choose(y.ravel(), Z)
	loss_matrix = np.maximum(0, 1-correct_class_score+Z)
	loss_matrix[y.ravel(), np.arange(loss_matrix.shape[1])] = 0

	loss = C*np.sum(loss_matrix, axis=(0,1))	
	loss += (1/(2*N))*np.sum(W*W)	
	F = (loss_matrix > 0).astype(int)
	F[y, np.arange(F.shape[1])] = np.sum(-1*F, axis=0)	
	dW = C*(X.dot(F.T)) + (1/N)*W			
	return loss, dW


# evaluate
def evaluate(x, y, W):
	pred_scores = np.dot(W.T, x)
	y_pred = np.argmax(pred_scores, axis=0)
	acc = np.mean(y_pred == y)
	return acc


# number of corrects
def num_corrects(x,y,W):
	pred_scores = np.dot(W.T, x)
	y_pred = np.argmax(pred_scores, axis=0)
	num_corrects = np.sum(y_pred == y)
	return num_corrects

# get minibatches
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



# run minibatch gradient descent
def minibatch_gd(num_epoch, W, X, y, x_val, y_val, n0=1, n1=100, C=0.1):
	W_optimal = W
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []

	best_val_acc = 0.0
	best_w = np.zeros_like(W_optimal)

	mini_batch_size = 32
	for epoch in range(num_epoch):
		lr = n0/(n1+epoch+1)
		total_train_loss, total_train_num_corrects = 0, 0

		# shuffle minibatches
		batches = random_mini_batches(X, y, mini_batch_size = mini_batch_size, seed = 0)
		num_batches = len(batches)

		for minibatch in batches:
			(batch_x, batch_y) = minibatch
			
			# train
			train_loss, dW = multiclass_svm_loss(W_optimal, batch_x, batch_y, C)
			W_optimal = W_optimal - lr*dW			

			train_num_corrects = num_corrects(batch_x, batch_y, W_optimal) 		# good

			total_train_loss += train_loss 		# good
			total_train_num_corrects += train_num_corrects 		# good


		# avg_loss and accuracy
		avg_train_loss = total_train_loss/num_batches		
		avg_train_acc = total_train_num_corrects/np.shape(X)[1]
		avg_val_loss, _ = multiclass_svm_loss(W_optimal, x_val, y_val, C)
		avg_val_acc = evaluate(x_val, y_val, W_optimal)

		if avg_val_acc > best_val_acc:
			best_val_acc = avg_val_acc
			best_w = W_optimal

		# save history
		train_loss_history.append(avg_train_loss)
		train_acc_history.append(avg_train_acc)
		val_loss_history.append(avg_val_loss)
		val_acc_history.append(avg_val_acc)

		# print result
		if epoch % 100 == 0:
			print ('epoch {}/{}:'.format(epoch, num_epoch-1))
			print ('train_loss: {:.4f}\t train_acc: {:.4f} %'.format(avg_train_loss, avg_train_acc*100))
			print ('val_loss: {:.4f}\t val_acc: {:.4f} %'.format(avg_val_loss, avg_val_acc*100))
			print ('Best val accuracy so far: {:.4f} %'.format(best_val_acc*100))
			print ('-'*50)

	return best_w, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history

# batch gradient descent
def batch_gd(num_epoch, W, X, y, x_val, y_val, C):
	m = np.shape(X)[1]
	W_optimal = W
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []

	best_val_acc = 0.0
	best_w = np.zeros_like(W_optimal)
	for epoch in range(num_epoch):
		lr = 1/(100+epoch+1)
		# shuffle data
		permutation = list(np.random.permutation(m))
		shuffled_X = X[:,permutation]
		shuffled_Y = y[:, permutation]
		
		train_loss, dW = multiclass_svm_loss(W_optimal, shuffled_X, shuffled_Y, C)
		W_optimal -= lr*dW
		val_loss,_ = multiclass_svm_loss(W_optimal, x_val, y_val, C)
		train_acc = evaluate(shuffled_X, shuffled_Y, W_optimal)
		val_acc = evaluate(x_val, y_val, W_optimal)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_w = W_optimal

		train_loss_history.append(train_loss)
		train_acc_history.append(train_acc)
		val_loss_history.append(val_loss)
		val_acc_history.append(val_acc)

		# print result
		if (epoch % 100 == 0) or epoch == 1999:
			print ('epoch {}/{}:'.format(epoch, num_epoch-1))
			print ('train_loss: {:.4f}\t train_acc: {:.4f} %'.format(train_loss, train_acc*100))
			print ('val_loss: {:.4f}\t val_acc: {:.4f} %'.format(val_loss, val_acc*100))
			print ('Best val accuracy so far: {:.4f} %'.format(best_val_acc*100))
			print ('-'*50)

	return best_w, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history


# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label \n')


