% Trung Vo - CSE 512
% Homework 2 - UCF101 data 3.2 multiclass svm
data = load('./q3_2_data.mat');
x_train = getfield(data, 'trD');
y_train = getfield(data, 'trLb');
x_val = getfield(data, 'valD');
y_val = getfield(data, 'valLb');
x_test = getfield(data, 'tstD');

% normalize data
mean = mean(x_train,2);
x_train = x_train - mean;
x_val = x_val - mean;
x_test = x_test - mean;

x_train = [x_train; ones(1, size(x_train,2))];
x_test = [x_test; ones(1, size(x_test,2))];
x_val = [x_val; ones(1, size(x_val,2))];

N = size(x_train, 2);
D = size(x_train, 1);
y_train = y_train';
y_val = y_val';

% feed data to train and evaluate
K = 10;
W = normrnd(0,1,D,K)*0.1;
num_epoch = 3000;
C = 10;
[W_optimal, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history] = utils.batch_gd(num_epoch, W, x_train, y_train, x_val, y_val, C);
fprintf('Training end...\n')
fprintf('Best validation accuracy: %.4f\n', best_val_acc);