% Author: Trung Vo
% For QP - problem 3.1, 3.2 (toydata)

data = load('./q3_1_data.mat');
x_train = getfield(data, 'trD');
y_train = getfield(data, 'trLb');
x_val = getfield(data, 'valD');
y_val = getfield(data, 'valLb');

% prepare dataset
N = size(x_train, 2);
D = size(x_train, 1);
y_train = y_train';
y_val = y_val';

fprintf('With C=0.1:\n');
C = 0.1;
[alpha, w, b] = qp_utils.train_qp(x_train, y_train, C);

% evaluate on validation set
acc = qp_utils.compute_acc(w,b,x_val,y_val);
fprintf('Accuracy: %.4f\n', acc);

% loss compute   
loss = qp_utils.compute_loss(w,b,x_train,y_train,C);
fprintf('Loss: %.4f\n', loss);
fprintf('Number of support vectors: %d\n', sum(alpha > 1e-6));

fprintf('With C=10:\n');
C = 10;
[alpha, w, b] = qp_utils.train_qp(x_train, y_train, C);

% evaluate on validation set
acc = qp_utils.compute_acc(w,b,x_val,y_val);
fprintf('Accuracy: %.4f\n', acc);

% loss compute   
loss = qp_utils.compute_loss(w,b,x_train,y_train,C);
fprintf('Loss: %.4f\n', loss);
fprintf('Number of support vectors: %d\n', sum(alpha > 1e-6));
