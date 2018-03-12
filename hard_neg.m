% Author: Trung Vo

data = load('./trainval_random.mat');
train_ubAnno = struct2cell(load('./trainAnno.mat'));
x_train = double(getfield(data, 'trD'));
y_train = getfield(data, 'trLb');
x_val = double(getfield(data, 'valD'));
y_val = getfield(data, 'valLb');

% prepare dataset
N = size(x_train, 2);
D = size(x_train, 1);
y_train = y_train';
y_val = y_val';

% separate positive and negative training set
y_pos = y_train(y_train == 1);
y_neg = y_train(y_train == -1);
x_pos = x_train(:,y_train == 1);
x_neg = x_train(:,y_train == -1);


C = 10;
[alpha, w, b] = qp_utils.train_qp(x_train, y_train, C);

loss_hist = [];
% loss compute   
loss = qp_utils.compute_loss(w,b,x_train,y_train,C);
loss_hist = [loss_hist, loss];
fprintf('Loss: %.4f\n', loss);

alpha_neg = alpha(y_train == -1);
temp_neg_train = x_neg;
temp_neg_lb = y_neg;

% hard negative mining
num_epoch = 10;
best_val_ap = 0.0;
best_w = w;
best_b = b;
ap_hist = [];
for epoch = 1:num_epoch
    fprintf('Epoch %i\n',epoch);
    % remove non SVs in negative set
    s_x = [];
    for i=1:size(temp_neg_train,2)
        if alpha_neg(i) > 1e-6
            s_x = [s_x, temp_neg_train(:,i)];
        end
    end
    
    % find hardest negative sample
    random_imgs = randperm(size(train_ubAnno{1,1},2));
    img_list = random_imgs(1:10);
    for i=1:10
        img_file = '';
        img_index = img_list(i);
        if img_index < 10
            img_file = strcat('./trainIms/000', num2str(img_index));
            img_file = strcat(img_file, '.jpg');
        else
            img_file = strcat('./trainIms/00', num2str(img_index));
            img_file = strcat(img_file, '.jpg');
        end
        im = imread(img_file);
        rects = HW2_Utils.detect(im, w, b, false);
        
        rects = rects(1:4,:);
        a = cell2mat(train_ubAnno{1,1}(img_index));
        
        % run through detected regions in each image
        % and check if they overlap much with groundtruth provided in ubAnno file
        indices = [];
        for j=1:size(a, 2)
            overlap = HW2_Utils.rectOverlap(rects, a(:,j));
            if j == 1
                indices = (find(overlap < 0.3))';
            else
                indices = intersect(indices, (find(overlap < 0.3))');
            end
        end
        
        if isempty(indices)
            continue;
        end
        rects = rects(:,indices);
        for k=1:size(rects,2)
            x1_im = max(1, floor(rects(2,k)));
            x2_im = min(size(im,1), floor(rects(4,k)));
            y1_im = max(1, floor(rects(1,k)));
            y2_im = min(size(im,2), floor(rects(3,k)));

            imReg = im(x1_im:x2_im, y1_im:y2_im, :);
            imReg = imresize(imReg, HW2_Utils.normImSz);
            feat = HW2_Utils.cmpFeat(rgb2gray(imReg));  
            s_x = [s_x, feat];
        end
    end
    
    new_train = double([x_pos, s_x]);
    new_lb = [y_pos, -1*ones(1,size(s_x,2))];
    [alpha, w, b] = qp_utils.train_qp(new_train, new_lb, C);
    
    % loss compute   
    loss = qp_utils.compute_loss(w,b,new_train,new_lb,C);
    loss_hist = [loss_hist, loss];
    fprintf('Loss: %.4f\n', loss);
    
    HW2_Utils.genRsltFile(w, b, 'val', 'test_file');
    [ap, prec, rec] = HW2_Utils.cmpAP('test_file.mat','val');
    fprintf('AP score: %.4f\n', ap);
    ap_hist = [ap_hist, ap];
    
    % update best parameters
    if ap > best_val_ap
        best_val_ap = ap;
        best_w = w;
        best_b = b;
    end
    fprintf('best val AP score so far: %.4f\n', best_val_ap);
    
    alpha_neg = alpha(new_lb == -1);
    temp_neg_train = s_x;
    temp_neg_lb = new_lb;
end

fprintf('Best val AP score: %.4f: \n', best_val_ap);
% plot loss
plot(loss_hist, 'LineWidth',2);
title('Loss');
xlabel('epochs');
ylabel('loss');
% plot AP scores
plot(ap_hist, 'LineWidth',2);
title('Val AP scores');
xlabel('epochs');
ylabel('AP score');

% generate test file
HW2_Utils.genRsltFile(best_w, best_b, 'test', '109845485');

