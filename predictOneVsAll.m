function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

for i=1:m,
  class = 1;
  max = (all_theta(1,:)*(X(i,:)')); % Assuming that the point is in first class
  for k=1:num_labels,
     if (all_theta(k,:)*(X(i,:)')) > max,
       max = (all_theta(k,:)*(X(i,:)')); % Reassigning the maximum probability of the point
       class = k; % Hence assigning that coressponding class to that point
       end;
    end;
  p(i,1) = class;
  end;






% =========================================================================


end
