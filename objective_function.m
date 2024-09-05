% Objective Function

%   Input: output weights W2, input weights W1, input features X, target
%          values Y, regularitazion parameter L2 lambda_reg, activation
%          function, biases b
%   Output: value of the objective function (MSE)

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function f_val = objective_function(W2, W1, X, Y, lambda_reg, activation_function, b)

    % Sample size
    N = size(X, 1);
    
    % Application of activation function to obtain the hidden layer output matrix
    H = activation_function((X*W1) + b);
    
    % Predictions
    Y_pred = H*W2;
    
    % MSE computation
    mse_val = (1 / N) * norm((Y_pred - Y ),2).^2;
    
    % Adding L2 regularization lambda
    f_val = mse_val + lambda_reg * norm(W2,2).^2;
    
end
