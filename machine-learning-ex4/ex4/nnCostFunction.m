function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];

hidden = sigmoid(X*Theta1');
hidden = [ones(m,1) hidden];

hypothesis = sigmoid(hidden*Theta2');

orig_y = zeros(m,num_labels);
for i = 1:size(orig_y,1)
    orig_y(i,y(i))=1;
end
J = (-1/m)*sum((orig_y.*log(hypothesis)) + ((1-orig_y).*log(1-hypothesis)), 'all');
J = J + ((lambda/(2*m))*(sum(Theta1(:, 2:end).^2, 'all') + sum(Theta2(:, 2:end).^2, 'all')));
%{
mat = zeros(m,1);
%mat = ((-transpose(orig_y))*(log(hypothesis)))+((-transpose(1-orig_y))*(log(1-hypothesis)));
for i = 1:m
    mat(i) = ((log(hypothesis(i,:)))*(-transpose(orig_y(i,:))))+((log(1-hypothesis(i,:)))*(-transpose(1-orig_y(i,:))));    
end
J = sum(mat)/m;

sumTh1 = 0;
sumTh2 = 0;
for i = 1:size(Theta1,1)
    for j = 2:size(Theta1,2)
        sumTh1 = sumTh1 + Theta1(i,j)*Theta1(i,j);
    end
end
for i = 1:size(Theta2,1)
    for j = 2:size(Theta2,2)
        sumTh2 = sumTh2 + Theta2(i,j)*Theta2(i,j);
    end
end
J = J + (lambda/(2*m)) * (sumTh1+sumTh2);
%} 
z2 = X*Theta1';
z2 = [ones(m,1) z2];

del3 = hypothesis - orig_y;
del2 = (del3*Theta2).*sigmoidGradient(z2);
del2 = del2(:, 2:end);
Theta1_grad = del2'*X;
Theta2_grad =del3'*hidden;
    
reg_term1 = (lambda/m)*(Theta1);
reg_term1(:,1)=0;
reg_term2 = (lambda/m)*(Theta2);
reg_term2(:,1)=0;
Theta1_grad = Theta1_grad/m + reg_term1;
Theta2_grad = Theta2_grad/m + reg_term2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
