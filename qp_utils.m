% Author: Trung Vo

classdef qp_utils
    methods (Static)
        % train quadratic programming
        function [alpha, w, b] = train_qp(x_train,y_train,C)
            N = size(x_train, 2);
            % form QP form
            V = x_train.*y_train;
            H = V'*V;
            f = -1*ones(N,1);            
            A = [-1*eye(N); eye(N)];
            b = [zeros(N,1); C*ones(N,1)];
            Aeq = y_train;
            beq = zeros(1,1);
            alpha = quadprog(H,f,A,b,Aeq,beq);

            % compute weights and biases optimal
            w = V*alpha;
            b = mean(y_train - w'*x_train);
        end
        
        % compute accuracy
        function acc = compute_acc(w,b,x_val,y_val)
            % evaluate on validation set
            y_pred = w'*x_val + b;
            y_pred = 1./(1+exp(-1*y_pred));
            for i = 1:size(y_pred,2)
                if y_pred(i) > 0.5
                    y_pred(i) = 1;
                else
                    y_pred(i) = -1;
                end
            end
            acc = mean(y_pred == y_val);
        end
        
        % loss compute  
        function loss = compute_loss(w,b,x_train,y_train,C)
            N = size(x_train,2);
            loss = 0.0;
            for i = 1:N
                loss = loss + max(0, 1-y_train(i)*(w'*x_train(:,i)+b));
            end
            loss = loss*C;
            loss = loss + 0.5*sum(sum(w.*w,1));
        end
    end
end