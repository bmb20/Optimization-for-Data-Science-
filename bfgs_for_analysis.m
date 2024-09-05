%BFGS function for analysis

%   Input:  output weights W2 initialized, hidden layer output H, target values Y, tolerance value, regularization L2
%           parameter, scalar parameter alpha_1, optimal solution
%   Output: output weights W2 optimizated, final gap value, number of
%           iterations

%Authors: Alessandro Mastrorilli, Biancamaria Bombino


function [W2, gap, iter] = bfgs_for_analysis(W2, H, Y, tol, lambda_reg, alpha_1, opt_sol)
 
    n_params = numel(W2);
    
    % Identity matrix with W2 dimensions
    I = eye(n_params); 

    %Inizialization of hessian inverse matrix
    H_approx = alpha_1*I;

    iter = 1;

    grad = compute_gradient(W2, H, Y, lambda_reg);
   
    while  norm(grad) > tol

        % Direction of descent
        d = -H_approx*grad;
  
        % Optimal alpha pitch
        alpha = line_search(W2, H, Y, d, lambda_reg);
        prev_W2 = W2;
        W2 = W2+alpha*d;

        prev_gradient = grad;
        grad = compute_gradient(W2, H, Y, lambda_reg);

        s = W2 - prev_W2;
        y = grad - prev_gradient;

        if norm(s) < tol || norm(grad) <tol
            break;
        end

        rho = 1 / (y'* s);
        scalar_term = 1 + rho * (y' * H_approx * y);

        % Update Hessian
        tmp1 = H_approx * y * s';
        tmp2 = s * s';
        H_approx = H_approx + rho * ((scalar_term * tmp2) - (tmp1 + s*y'*H_approx));

        iter = iter+1;
       
    end

    gap = norm(W2 - opt_sol)/norm(opt_sol);
    disp(gap);    

end
