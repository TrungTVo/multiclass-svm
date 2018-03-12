% Trung Vo - CSE 512
% Homework 2 - Toydata 3.2 multiclass svm
data = load('./q3_1_data.mat');
x_train = getfield(data, 'trD');
y_train = getfield(data, 'trLb');
x_val = getfield(data, 'valD');
y_val = getfield(data, 'valLb');

N = size(x_train, 2);
D = size(x_train, 1);
y_train = y_train';
y_val = y_val';
y_train(y_train == -1) = y_train(y_train == -1) + 3;
y_val(y_val == -1) = y_val(y_val == -1) + 3;

% feed data to train and evaluate
K = 2;
W = zeros(D,K);
num_epoch = 2000;
C = 0.1;
[W_optimal, best_val_acc, train_loss_history, train_acc_history, val_loss_history, val_acc_history] = utils.batch_gd(num_epoch, W, x_train, y_train, x_val, y_val, C);
fprintf('Training end...\n')
fprintf('Best validation accuracy: %.4f\n', best_val_acc);