function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%sprintf("Size of X")
%size(X)
%sprintf("Size of theta ")
%size(theta)
J = 0;
grad = zeros(size(theta));
x = X;
J = sum((x*theta - y).^2, 1)/(2*m) + sum(theta(2:end,:).^2)*lambda/(2*m);
grad = sum((x*theta - y) .* x, 1)./m + [0; theta(2:end,:)]' .* lambda/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);
%sprintf("Size of grad")
%size(grad)
end
