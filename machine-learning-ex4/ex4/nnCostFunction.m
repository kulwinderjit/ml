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
%fprintf("size of nn_params\n");
%size(nn_params)
% Setup some useful variables
m = size(X, 1);
%size X = 5000x400         
% You need to return the following variables correctly 
xx=(Theta1 * [ones(m,1) X]')';
% size xx = 5000x25
%fprintf("size of xx\n");
%size(xx)
mm = size(xx, 1);
xxx=(Theta2 * [ones(mm,1) sigmoid(xx)]')';
xxx=xxx';
% size xxx = 10x5000
%fprintf("size of xxx\n");
%size(xxx)
yy = y == (1:num_labels);
yy=yy';
% size yy = 10x5000
%fprintf("size of yy\n");
%size(yy)
J=sum(sum(-1*(yy).*log(sigmoid(xxx)) - (1-yy) .* log(1 - sigmoid(xxx))))/m + (sum(sum(Theta1(:, 2:end).^2)) + sum(sum((Theta2(:, 2:end).^2))))*lambda/(2*m);

d3 = sigmoid(xxx) - yy;
% size d3 = 10x5000
%fprintf("size of d3\n");
%size(d3)
d2 = (Theta2' * d3) .* sigmoidGradient([ones(m, 1) xx]');
d2 = d2(2:end, :);
% size d2 = 25x5000
%fprintf("size of d2\n");
%size(d2)
Theta1_grad = (d2*[ones(m, 1) X])/m + [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)].*(lambda/m);
%Theta1_grad = zeros(size(Theta1));
Theta2_grad = (d3*[ones(m, 1) sigmoid(xx)])/m + [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)].*(lambda/m);
%Theta2_grad = zeros(size(Theta2));

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%fprintf("size of grad\n");
%size(grad)

end
