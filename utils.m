% Trung Vo - CSE 512
% Homework 2 - Problem 3.2
% Multiclass SVM with gradient descent

classdef utils
    methods (Static)        
        % compute loss, gradients
        function [loss,dW] = multiclass_svm_loss(W, X, y, C)
            [D,K] = size(W);
            [D,N] = size(X);
            loss = 0.0;
            dW = zeros(size(W),'like',W);
                        
            Z = W'*X;
            correct_class_score = zeros(1,N);
            for i = 1:N
                correct_class_score(1,i) = Z(y(i),i);
            end;
            loss_matrix = max(0, 1-correct_class_score+Z);
            for i=1:N
                loss_matrix(y(i),i) = 0;
            end;
            loss = C*sum(sum(loss_matrix,2),1);             
            loss = loss + (1/(2*N))*sum(sum(W.*W,2),1);            
            
            F = double(loss_matrix > 0);
            for i=1:N
                F(y(i),i) = sum(-1*F(:,i), 1);
            end;            
            dW = C*(X*F') + (1/N)*W;                
        end;
        
        % evaluate model
        function acc = evaluate(x, y, W)
            pred_scores = W'*x;
            [vals, y_pred] = max(pred_scores);
            acc = mean(y_pred == y);
        end;
        
        % number of corrects
        function num_corrects = num_corrects(x,y,W)
            pred_scores = W'*x;
            [vals, y_pred] = max(pred_scores);
            num_corrects = sum(y_pred == y);
        end;                
        
        % batch gradient descent
        function [best_w, best_val_acc, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist] = batch_gd(num_epoch, W, X, y, x_val, y_val, C)             
            W_optimal = zeros(size(W), 'like', W);
            W_optimal = W_optimal + W;
            train_loss_hist = [];
            train_acc_hist = [];
            val_loss_hist = [];
            val_acc_hist = [];
            
            best_val_acc = 0.0;
            best_w = zeros(size(W_optimal), 'like', W_optimal);
            
            % train
            for epoch = 1:num_epoch
                lr = 1/(100+epoch+1);
                [train_loss, dW] = utils.multiclass_svm_loss(W_optimal, X, y, C);
                W_optimal = W_optimal - lr*dW;
                [val_loss, x] = utils.multiclass_svm_loss(W_optimal, x_val, y_val, C);
                train_acc = utils.evaluate(X, y, W_optimal);
                val_acc = utils.evaluate(x_val, y_val, W_optimal);
                
                % update best result
                if val_acc > best_val_acc
                    best_val_acc = val_acc;
                    best_w = W_optimal;
                end;
                
                % save history
                train_loss_hist = [train_loss_hist, train_loss];
                train_acc_hist = [train_acc_hist, train_acc];
                val_loss_hist = [val_loss_hist, val_loss];
                val_acc_hist = [val_acc_hist, val_acc];
                                
                % print result
                if mod(epoch, 100) == 0
                    fprintf('epoch %i/%i:\n', epoch, num_epoch)                    
                    fprintf('train_loss: %.4f\t\t train_acc: %.4f\n', train_loss, train_acc)
                    fprintf('val_loss: %.4f\t\t val_acc: %.4f\n', val_loss, val_acc)
                    fprintf('Best val accuracy so far: %.4f\n', best_val_acc)
                    fprintf('-------------------------------------------------------\n')
                end;
            end;
        end;
    end;
end