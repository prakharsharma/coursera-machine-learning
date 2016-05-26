function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

error = (X * theta) - y;
error_sq = error .^ 2;
sum_sq_error = sum(error_sq);
J = (1 / (2 * m)) * sum_sq_error;

%theta(1) = 0;
%reg = sum(theta .^ 2) * (lambda / (2 * m));
%J = J + reg;

%J = sum(error .^ 2) / (2*m);
%reg = (theta(2, :)' * theta(2, :)) * (lambda / (2 * m));
%5J = J + reg;

%grad = (error' * X)' / m;
%grad(2) = grad(2) + (theta(2) * (lambda / m));

grad = grad(:);

end
