% Gradient Function

%   Input: output weights W2, output of the hidden layer H, target values Y, regularization parameter L2 lambda_reg
%   Output: Gradient value

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function gradient = compute_gradient(W2, H, Y, lambda_reg)
    % Sample size
    N = size(Y, 1);
    
    % Prediction
    Y_pred = H * W2;
    
    % Calculate error
    error =  Y_pred - Y;
    
    % Gradient 
    grad = (2 / N) * (H' * error);
    
    % Adding regularization L2
    gradient = grad + 2 * lambda_reg * W2;
end
