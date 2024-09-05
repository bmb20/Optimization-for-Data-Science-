% BFGS function

%   Input:  biases b, input weights W1, output weights W2 initialized, output of the hidden layer H, optimal solution, 
%           input features X, activaction function, target values Y, tolerance value, regularization L2
%           parameter, scalar parameter alpha_1
%   Output: output weights W2 optimizated, objective function final value,
%           gradient norms vector, objective function vector
%           values, relative gaps vector, final gap value

%Authors: Alessandro Mastrorilli, Biancamaria Bombino

function [W2, of_bfgs, gradient_norms, obj_func, rel_gaps, gap] = bfgs(b, W1, W2, H, opt_sol, X, activation_function, Y, tol, lambda_reg, alpha_1)
    n_params = numel(W2);
    
    % Identity matrix with W2 dimensions
    I = eye(n_params); 

    %Inizialization of hessian inverse matrix
    H_approx = alpha_1*I;

    % Array to store norms , mse, gaps
    gradient_norms = [];
    obj_func = [];
    rel_gaps = [];

    gap = 0;
    iter = 1;

    grad = compute_gradient(W2, H, Y, lambda_reg);
   
    while norm(grad) > tol

        % Direction of descent
        d = ((-H_approx)*grad);
  
        % Optimal alpha pitch
        alpha = line_search(W2, H, Y, d, lambda_reg);
        prev_W2 = W2;
        W2 = W2+alpha.*d;
        
        % Gap calculation
        gap = norm(W2 - opt_sol)/norm(opt_sol);
        rel_gaps(iter) = gap;
        
        disp("gap of bfgs");
        disp(gap);
        
        % Gradient computation 
        norm_grad = norm(grad);
        gradient_norms(iter) = norm_grad;

        % Objective function calculation 
        of_bfgs = objective_function(W2, W1, X, Y, lambda_reg, activation_function, b);
        disp('objective function bfgs');
        disp(of_bfgs);
        obj_func(iter) = of_bfgs;
        
        prev_gradient = grad;
        grad = compute_gradient(W2, H, Y, lambda_reg);

        s = W2 - prev_W2;
        y = grad - prev_gradient;
        
        rho = 1 / dot(y, s);
        scalar_term = 1 + rho * (y' * H_approx * y);
        tmp1 = H_approx * y * s';
        tmp2 = s * s';
        H_approx = H_approx + rho * ((scalar_term * tmp2) - (tmp1 + s*y'*H_approx));

        iter = iter+1;
       
    end
    

end
