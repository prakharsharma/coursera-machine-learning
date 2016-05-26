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

% transform y from m x 1 vector to m x K matrix
yt = zeros(m, num_labels);
for i=1:m,
    label = y(i);
    yt(i, label) = 1;
end;

% forward propagation
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% cost computation

%% unvectorized implementation
%J2 = 0;
%for i=1:m,
%  for k=1:num_labels,
%    J2 = J2 + (yt(i,k) * log(h2(i,k)));
%    J2 = J2 + ((1-yt(i,k)) * log(1 - h2(i,k)));
%  end;
%end;
%J2 = (-1 * J2)/m;

%% partially vectorized implementation

%J1=0;
%for i=1:m,
%  h_theta_i = h2(i,:)';
%  J1 = J1 + (yt(i,:) * log(h_theta_i));
%  J1 = J1 + ((1 - yt(i,:)) * log(1 - h_theta_i));
%end;
%J1 = (-1 * J1)/m;
%J = J1;

%% fully vectorized implementation
unrolled_y = yt(:);
unrolled_h2 = h2(:);
J3 = 0;
J3 = J3 + (unrolled_y' * log(unrolled_h2));
J3 = J3 + ((1 - unrolled_y)' * log(1 - unrolled_h2));
J3 = (-1 * J3)/m;
J = J3;

% fully vectorized regularized cost computation
unrolled_theta = [Theta1(:) ; Theta2(:)];
theta1_first_params = Theta1(:, 1);
theta2_first_params = Theta2(:, 1);
all_squares = unrolled_theta' * unrolled_theta;
theta1_first_params_squares = theta1_first_params' * theta1_first_params;
theta2_first_params_squares = theta2_first_params' * theta2_first_params;
reg = all_squares - theta1_first_params_squares - theta2_first_params_squares;
reg = (lambda * reg)/(2 * m);

J = J + reg;

% -------------------------------------------------------------
% backprop implementation

D1 = zeros(hidden_layer_size, input_layer_size + 1);
D2 = zeros(num_labels, hidden_layer_size + 1);

for i=1:m,

  % step 1: forward prop
  a1 = X(i, :);
  z2 = [1 X(i,:)] * Theta1';
  a2 = sigmoid(z2);
  %size(a2)

  z3 = [1 a2] * Theta2';
  a3 = sigmoid(z3);
  %size(a3)

  % make row vectors into column vectors
  a1 = a1'; z2 = z2'; a2 = a2'; z3 = z3'; a3 = a3';

  % step 2: find error of output layer
  yi = yt(i,:)';
  %logical_out = zeros(size(a3, 1), 1);
  %[max_val, max_idx] = max(a3, [], 1);
  %logical_out(max_idx) = 1;
  %delta3 = logical_out - yi;
  delta3 = a3 - yi;
  %size(delta3)

  % step 3: find error of hidden layer
  delta2 = (Theta2' * delta3) .* ([1; a2] .* (1 - [1; a2]));
  %size(delta2)

  % accumulate D1 and D2
  D2 = D2 + (delta3 * [1; a2]');
  D1 = D1 + (delta2(2:end) * [1; a1]');

end;

D1 = (1/m) * D1;
D2 = (1/m) * D2;

% add regularization to the gradients
D1(:, 2:end) = D1(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
D2(:, 2:end) = D2(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));


% =========================================================================

Theta1_grad = D1;
Theta2_grad = D2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
