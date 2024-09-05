% Function for the forward substitution

%   Input: L matrix obtained from Cholesky factorization, b represents (H'*Y) 
%          where H' is the transposed of the hidden layer output matrix and
%          Y are the target values
%   Output: y vector after forward substitution was applied

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function y = forward_substitution(L, b)

    % Get the size of the matrix
    n = length(b);
    
    % Initialize the solution vector y
    y = zeros(n, 1);
    
    % Perform forward substitution
    for i = 1:n
        y(i) = (b(i) - L(i,1:i-1) * y(1:i-1)) / L(i,i);
    end
    
end