function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
predictions_h =  sigmoid(X*theta);

leftPart_sum = -y' * log(predictions_h);

rightPart_sum = (1 - y') * log(1 - predictions_h);

theta_temp = theta;

theta_temp(1) = 0;

lambaCostPart = (lambda / (2 * m)) * sum(theta_temp .^ 2);

J = (1 / m) * (leftPart_sum - rightPart_sum) + lambaCostPart;
%J = (1 / m) * (leftPart_sum - rightPart_sum); %without Lamba => overfitting
lambdaGradientPart = lambda / m * theta_temp;

grad = ((1/m) * (X' * (predictions_h - y))) + lambdaGradientPart;
%grad = ((1/m) * (X' * (predictions_h - y)));%without Lamba => overfitting




% =============================================================

end
