% Function for synthetic dataset generation

    % Output: file csv with synthetic data

% Authors: Alessandro Mastrorilli, Biancamaria Bombino

function generate_dataset(num_samples, num_features)

    % Parameters for synthetic dataset
    num_classes = 2;

    % Set random seed for reproducibility
    rng(42);  

    % Generate synthetic random data
    X = randn(num_samples, num_features);  

    % Standardization (z-score normalization)
    X = zscore(X);

    % Generate random class labels (0 or 1)
    y = randi([0 num_classes-1], num_samples, 1);

    % Create DataFrame 
    df = array2table(X);

    % Conversion in categoric for the target 
    df.target = categorical(y); 

    % Save to CSV
    writetable(df, 'synthetic_dataset.csv');

end


