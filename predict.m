function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

for q=1:m, % for each sample
  
    % calculating the second layer
    z2 = Theta1*(X(q,:)');

    % calculating a2
    a2 = sigmoid(z2);

    % adding the positive bias
    no_of_hiddenunits = size(Theta2,2);
    no_of_hiddenunits_exceptbias = size(Theta2,2) - 1;
    a2_final = zeros(no_of_hiddenunits,1);
    a2_final(1,1) = 1;
    for c=1:no_of_hiddenunits_exceptbias,
      a2_final(c+1,1) = a2(c,1);
    end;
    
    % calculating the third layer
    z3 = Theta2*a2_final;  
    a3 = sigmoid(z3);
    
    %calculating the max probability
    [x,ix] = max(a3);
    p(q,1) = ix;
end;



% =========================================================================


end
