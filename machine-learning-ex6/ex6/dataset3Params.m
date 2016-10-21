function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% =========================================================================

min_error = 0;

possibilities = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for C_prime = possibilities
    for sigma_prime = possibilities
        % fprintf('C_prime: %f, sigma_prime: %f\n', C_prime, sigma_prime);
        model = svmTrain(X, y, C_prime, @(x1, x2) gaussianKernel(x1, x2, sigma_prime));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        % fprintf('error: %f\n', error);
        if (C_prime == 0.01 && sigma_prime == 0.01)
            % fprintf('first time, updated C: %f, sigma: %f, min_error: %f\n', C_prime, sigma_prime, error);
            C = C_prime;
            sigma = sigma_prime;
            min_error = error;
        elseif (error <= min_error)
            % fprintf('updated C: %f, sigma: %f, min_error: %f\n', C_prime, sigma_prime, error);
            C = C_prime;
            sigma = sigma_prime;
            min_error = error; 
        end
    end
end

% C = 1; sigma = 0.1;

end
