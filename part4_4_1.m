% Author: Trung Vo

data = load('./trainval_random.mat');
x_train = double(getfield(data, 'trD'));
y_train = getfield(data, 'trLb');
x_val = double(getfield(data, 'valD'));
y_val = getfield(data, 'valLb');

% prepare dataset
N = size(x_train, 2);
D = size(x_train, 1);
y_train = y_train';
y_val = y_val';

fprintf('With C=10:\n');
C = 10;
[alpha, w, b] = qp_utils.train_qp(x_train, y_train, C);

% evaluate on validation set
acc = qp_utils.compute_acc(w,b,x_val,y_val);
fprintf('Accuracy: %.4f\n', acc);

% loss compute   
loss = qp_utils.compute_loss(w,b,x_train,y_train,C);
fprintf('Loss: %.4f\n', loss);

HW2_Utils.genRsltFile(w, b, 'val', 'part4_4_1_result');
[ap, prec, rec] = HW2_Utils.cmpAP('part4_4_1_result.mat','val');
fprintf('AP score: %.4f\n', ap);