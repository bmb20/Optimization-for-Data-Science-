% Function for the sigmoid activation function

%   Input: x vector to which the activation function is going to be applied
%   Output: y vector after activation function was applied

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function y = activation_function(x)
    y = 1 ./ (1 + exp(-x)); 
end
