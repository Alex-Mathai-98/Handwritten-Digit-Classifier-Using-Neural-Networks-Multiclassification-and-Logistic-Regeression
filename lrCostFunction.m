function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


sigmoid_value = zeros(m,1);
sigmoid_value = sigmoid(X*theta);

temp = ones(m,1);
term1 = y.*log(sigmoid_value) + (temp - y).*(log(temp - sigmoid_value));
J = (-1/m)*(sum(term1));

theta_square = theta.^2;
theta_square(1,1) = 0; % VERY IMPORTANT --> THETA(ZERO) MUST BE EXCLUDED
regularization_term = (lambda/(2*m))*sum(theta_square) ;
J = J+regularization_term;
% ============================================================= %
grad = (1/m)*(X'*(sigmoid_value-y)) + (lambda/m).*theta;
grad(1,1) = (1/m)*(X'*(sigmoid_value-y))(1,1) ;
grad = grad(:);

end
