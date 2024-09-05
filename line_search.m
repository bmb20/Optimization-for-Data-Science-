% Line Search function for BFGS method

%   Input: output weights W2, hidden layer output H, target values Y,
%          direction of descent d, regularization parameter L2
%   Output: optimal step alpha

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

% function alpha_opt = line_search(W2, H, Y, d, lambda_reg)
% 
%     N = size(H,1);
%     Ht_H = H'*H;
% 
%     numerator = (2/N)* ((W2'*Ht_H*d)- (Y'*H*d))+ 2*(lambda_reg*W2'*d);
%     denominator =  ((2/N)* (d'*Ht_H*d)) + 2*lambda_reg*(d'*d);
%     alpha_opt = - numerator/denominator;
% 
% end

function alpha_opt = line_search(W2, H, Y, d, lambda_reg)

    N = size(H,1);
    
    Hd = H*d;
    
    numerator = (2/N)*((W2'*H'*Hd) - (Y'*Hd))+ 2*(lambda_reg*W2'*d);
    denominator =  ((2/N)* (Hd'*Hd)) + 2*lambda_reg*(d'*d);
    alpha_opt = - numerator/denominator;

end

