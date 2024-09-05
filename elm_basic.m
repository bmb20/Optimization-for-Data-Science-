% ELM Implementation

%   Input: input features X, target values Y, number of hidden neurons,
%          activation function, seed
%   Output: input weights W1, output weights W2, biases b, hidden layer
%           output matrix H

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function [W1, W2, b, H] = elm_basic(X, Y, num_hidden_neurons, activation_function, seed)
    % Setting seed
    rng(seed);
    
    input_size = size(X, 2);

    % Initialization of input weights W1
    minW = -1/sqrt(input_size); 
    maxW = 1/sqrt(input_size);
    W1 = minW + (maxW-minW).*rand(input_size,num_hidden_neurons);
   
    % Initialization of biases
    b = randn(1, num_hidden_neurons); 

    % Calculation of the output of the hidden layer
    H = activation_function((X*W1)+b);
    
    % Calculation of the pseudoinverse of H
    H_pinv = pinv(H);
    
    % Calculation of the output weights W2
    W2 =  H_pinv *Y;

end
