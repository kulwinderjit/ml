function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
testVal = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
testvalSize = length(testVal);
predictionRes = zeros(testVal .^ 2, 3);
x1=X;
x2=X;
for c = 1 : testvalSize
  for s = 1 : testvalSize
    model = svmTrain(X, y, testVal(1, c), @(x1, x2) gaussianKernel(x1, x2, testVal(1, s)));
    predictions = svmPredict(model, Xval);
    predictionRes = [predictionRes; testVal(1, c), testVal(1, s), mean(double(predictions ~= yval))];
  endfor
endfor
[m im] = min(predictionRes(:, 3));
C = predictionRes(im, 1);
sigma = predictionRes(im , 2);
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
end
