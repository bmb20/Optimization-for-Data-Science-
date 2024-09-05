% Cholesky Function 

%   Input: matrix Q for factorization
%   Output: factorized matrix L

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function L = cholesky_factorization(Q)

    % Check if Q is symmetric
    if ~issymmetric(Q)
        error('Matrix Q is not symmetric.');
    end
    
    % Check if Q is positive definite
    if any(eig(Q) <= 0)
        error('Matrix Q is not positive definite.');
    end

    % Input verification
    [m, n] = size(Q);
    if m ~= n
        error('La matrice non è quadrata.');
    end
    
    if ~isequal(Q, Q')
        error('La matrice non è Hermitiana.');
    end
    
    % Initialization of L matrix
    L = zeros(size(Q));

    for j = 1:n
        for i = j:n
            if i == j % on the diagonal
                L(i, j) = sqrt(Q(i, j) - sum(L(i, 1:j-1).^2));
            else % off the diagonal
                num = (Q(i, j) - L(i, 1:j-1) * L(j, 1:j-1)');
                L(i, j) = num / L(j, j);
            end
        end
    end

    % TEST CHOLEKSY CON SOLO UN FOR

    % % Initialization of L matrix
    % n = size(Q, 1);
    % L = zeros(n);
    % 
    % for j = 1:n
    %     % Calcolo della diagonale
    %     L(j, j) = sqrt(Q(j, j) - sum(L(j, 1:j-1).^2));
    % 
    %     % Calcolo degli elementi fuori diagonale
    %     if j < n
    %         L(j+1:n, j) = (Q(j+1:n, j) - L(j+1:n, 1:j-1) * L(j, 1:j-1)') / L(j, j);
    %     end
    % end

end

