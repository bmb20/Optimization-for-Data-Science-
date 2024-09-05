% Function to calculate normal equation to resolve the system: 
%                   Ly = Y'*T
%                   L'*W2 = y

% Input: matrix factorized by cholesky_factorization function L, hidden layer output matrix H, target values Y
% Output: optimized output weights W2

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function [W2] = normal_eq(L, H, Y)

    YHT = (H'*Y);
    
    % y computation
    y = forward_substitution(L, YHT);
    
    % W2 computation
    W2 = backward_substitution(L', y); 

end