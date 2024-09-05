% Function for the backward substituition 

%   Input: U transpose of the matrix obtained with Cholesky factorization and y vector obtained from forward substitution
%   Output: x vector after the backward substitution was applied

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function x = backward_substitution(U, y)

    % Get the size of the matrix
    n = length(y);
    
    % Initialize the solution vector x
    x = zeros(n, 1);
    
    % Perform backward substitution
    for i = n:-1:1
        x(i) = (y(i) - U(i,i+1:end) * x(i+1:end)) / U(i,i);
    end
    
end