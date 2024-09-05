% Function to initialize output weights W2

%   Input: input_size, output_size, seed
%   Output: output weights W2 initializated

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function W2 = initialize_weights(input_size, output_size, seed)
    % Setting seed
    rng(seed);

    % Lower and Upper bound
    l_boud = -1/sqrt(input_size);
    u_boud = 1/sqrt(input_size);
    
    % W2 output weights initialization
    W2 = l_boud+(u_boud-l_boud)*rand(input_size, output_size);
end
