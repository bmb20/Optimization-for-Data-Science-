% Main script
% Authors: Alessandro Mastrorilli, Biancamaria Bombino

% Number of features and samples for dataset generation function
num_features = 1000;
num_samples = 30000;

% Function to generate the dataset
generate_dataset(num_samples,num_features);

% Uploading dataset from the csv
dataset = readtable('synthetic_dataset.csv');

% The last column is the output
X = table2array(dataset(:, 1:end-1));  
Y = table2array(dataset(:, end)); 

% Make sure Y is 2D for consistency
Y = reshape(Y, [], 1);  

input_size = size(X, 2);

activation_function = @activation_function;

% Setting the seed value
seed = 42;

% Best value for the number of hidden neurons
num_hidden_neurons = 10;

% Setting regularitazion parameter L2 value
lambda_reg = 0.01;

% ELM Basic
    disp("elm basic - 10 neuroni - 1000 features - 300000 samples")
    
    [W1, W2, b, H] =  elm_basic(X, Y, num_hidden_neurons, activation_function, seed);
    
    disp("pesi di output W2 elm basic");
    disp(W2);

    % Objective Fuction value 
    of_val = objective_function(W2, W1, X, Y, lambda_reg, activation_function, b);
    disp("objective function value elm basic");
    disp(of_val);

% OPTIMAL SOLUTION FOR RELATIVE GAP
    N = size(X,1);
    I = eye(size(H'*H));
    opt_sol = ((H'*H) + N*lambda_reg*I)\(H'*Y);
    
    disp("opt_sol")
    disp(opt_sol);
    
    % Objective Function value with the optimal solution
    of_optimal = objective_function(opt_sol, W1, X, Y, lambda_reg, activation_function, b);
    disp("objective value function with opt_sol");
    disp(of_optimal);


% BFGS
    tol = 1e-15;
    alpha_1 = 2.5;
    
    % Initialization of W2 
    W2 = initialize_weights(size(H, 2), size(Y, 2), seed);

   [W2_bfgs, of_bfgs, gradient_norms, obj_func, rel_gaps, gap] = bfgs(b, W1, W2, H, opt_sol, X, activation_function, Y, tol, lambda_reg, alpha_1);

% CHOLESKY
    Q = (H'*H) + (N*lambda_reg*I);

    L = cholesky_factorization(Q);
    W2_ch = normal_eq(L, H, Y);
    
    % Objective Function value with the Cholesky solution
    of_ch = objective_function(W2_ch, W1, X, Y, lambda_reg, activation_function, b);
    disp("objective value function with W2 of cholesky");
    disp(of_ch);
    
    % Gap calculation for Cholesky
    gap_tr = norm(W2_ch-opt_sol)/norm(opt_sol);
    disp('gap cholesky');
    disp(gap_tr);

% GRAPH FOR THE TREND OF THE BFGS
    disp("BFGS line plot con calcolo mse, relative gap e norm grad per ogni iterazione")

    % Obtain vectors index
    iterations = 1:numel(rel_gaps);

    % Plot creation
     figure;
     hold on;
     plot(iterations, rel_gaps, '-o', 'DisplayName', 'Relative Gap');
     plot(iterations, gradient_norms, '-s', 'DisplayName', 'Norm Gradient');
     plot(iterations, obj_func, '-d', 'DisplayName', 'Objective Function value');

     hold off;

    % Axes configuration
    % Logarithmic scale on the y-axis
     set(gca, 'YScale', 'log'); 
     xlabel('Numero di Iterazioni');
     ylabel('Valore (scala logaritmica)');
     title('Andamento di Relative Gap, Norm Gradient e Objective Function durante le Iterazioni');
     legend show;
     grid on;

    % Visualization of the graphic
     shg;
     disp("Fine line plot bfgs")

% GRAPH FOR COMPARISON OF OPTIMIZATION METHODS AS DIMENSIONS VARY
    % Range of number of variables to explore
    num_samples_range = [2000, 3000, 4000];

    % Preallocation of matrices for timing results
    times_bfgs = zeros(length(num_samples_range), 1);
    times_cholesky = zeros(length(num_samples_range), 1);

    % Iteration on the different dimensions to test
    for i = 1:length(num_samples_range)

            num_samples = num_samples_range(i);
            num_hidden_neurons =  num_samples_range(i);

            % Dataset generation with the dimension of H to test
            generate_dataset(num_samples,num_features);
            dataset = readtable('synthetic_dataset.csv');

            X = table2array(dataset(:, 1:end-1));
            Y = table2array(dataset(:, end));
            Y = reshape(Y, [], 1);

            % ELM Basic
            [W1, ~, b, H] =  elm_basic(X, Y, num_hidden_neurons, activation_function, seed);
            
            disp('size H');
            disp(size(H));
            
            % OPTIMAL SOLUTION FOR RELATIVE GAP
            N = size(X,1);
            I = eye(size(H'*H));
            opt_sol = ((H'*H) + N*lambda_reg*I)\(H'*Y);

            of_optimal = objective_function(opt_sol, W1, X, Y, lambda_reg, activation_function, b);

            % BFGS
            W2 = initialize_weights(size(H, 2), size(Y, 2), seed);

            tic;
            [W2,  gap, iter] = bfgs_for_analysis(W2, H, Y, tol, lambda_reg, alpha_1, opt_sol);
            disp('gap bfgs');
            disp(gap);
            time_bfgs = toc;

            % Cholesky
            tic;
            Q = (H'*H) + (N*lambda_reg*I);

            L = cholesky_factorization(Q);
            W2_ch = normal_eq(L, H, Y);

            gap_ch = norm(W2_ch-opt_sol)/norm(opt_sol);
            disp('gap cholesky');
            disp(gap_ch);

            time_cholesky = toc;

            % Salvataggio dei risultati
            times_bfgs(i) = time_bfgs;
            times_cholesky(i) = time_cholesky;

    end

    disp("tempo bfgs")
    disp(times_bfgs)
    disp("tempo cholesky")
    disp(times_cholesky)


    % Plot dei risultati
    figure;

    plot(num_samples_range, times_bfgs, '-o', 'DisplayName', 'BFGS');
    hold on;
    plot(num_samples_range, times_cholesky, '-s', 'DisplayName', 'Cholesky');
    hold off;
    xlabel('Numero di Variabili');
    ylabel('Tempo di Esecuzione (s)');
    title('Tempo di Esecuzione di BFGS e Cholesky');
    legend('show');
    grid on;

    % Mostra il grafico
    shg;


% GRAPH ANALYSIS OF THE METHODS VARYING LAMBDA FOR EACH DIMENSION
    % Range of number of variables to explore
    num_samples_range = [2000, 3000, 4000];
    % Range of value of lambda to test
    lambda_reg_range = [1e+2, 1e+0, 1e-2];
    %lambda_reg_range = [1e+2, 1e+0, 1e-2, 1e-4];

    % Iteration over different numbers of variables
    for i = 1:length(num_samples_range)
        num_samples = num_samples_range(i);
        num_hidden_neurons =  num_samples_range(i);
        
        % Generazione del dataset
        generate_dataset(num_samples, num_features);
        dataset = readtable('synthetic_dataset.csv');
        
        X = table2array(dataset(:, 1:end-1));
        Y = table2array(dataset(:, end));
        Y = reshape(Y, [], 1);
        
        % Iteration over different values of lambda_reg
        for j = 1 : length(lambda_reg_range)

            % ELM Basic
                [W1, ~, b, H] =  elm_basic(X, Y, num_hidden_neurons, activation_function, seed);

                disp('size H:');
                disp(size(H));
                disp('value of lambda:');
                disp(lambda_reg_range(j));

                % OPTIMAL SOLUTION FOR RELATIVE GAP
                N = size(X,1);
                I = eye(size(H'*H));
                opt_sol = ((H'*H) + N*lambda_reg_range(j)*I)\(H'*Y);
    
                of_optimal = objective_function(opt_sol, W1, X, Y, lambda_reg_range(j), activation_function, b);
    
                % BFGS
                W2 = initialize_weights(size(H, 2), size(Y, 2), seed);
                
                tic;
                [W2,  gap,iter] = bfgs_for_analysis(W2, H, Y, tol, lambda_reg_range(j), alpha_1, opt_sol);
                disp('gap bfgs');
                disp(gap);
                
                disp('numero iterazioni bfgs')
                disp(iter);
                time_bfgs = toc;

                % Cholesky
                tic;
                Q = (H'*H) + (N*lambda_reg_range(j)*I);
    
                L = cholesky_factorization(Q);
                W2_ch = normal_eq(L, H, Y);
    
                gap_ch = norm(W2_ch-opt_sol)/norm(opt_sol);
                disp('gap cholesky');
                disp(gap_ch);
    
                time_cholesky = toc;
    
                disp("tempo bfgs");
                disp(time_bfgs);
                disp("tempo cholesky");
                disp(time_cholesky);
                
        end

    end


   