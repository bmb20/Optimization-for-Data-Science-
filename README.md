# Optimization-for-Data-Science-
Solving an unconstrained optimization problem, based on the Extreme Learning Machine. The objective is to minimize the objective function (MSE) using the Quasi Newton BFGS method and the closed form solution with normal equations and Cholesky factorization.

# EXTREME LEARNING MACHINE with BFGS and Cholesky Factorization

Project for the Optimization course 2023/2024 @ [University of Pisa]

Authors: [Alessandro Mastrorilli] , [Biancamaria Bombino]

## Description

In this project, our objective is to train an ELM neural network, considering our optimization goal and utilizing the following optimization approaches:
**Quasi-Newton Method: BFGS**
**Closed-form Solution with Normal Equations and Cholesky Factorization**
The main difference between the two techniques lies in the method used to optimize the model's weights. The first method employs an iterative approach based on gradient descent, while the second uses a closed-form solution based on normal equations and Cholesky factorization. Both aim to find the optimal model weights that minimize the error between the model's predictions and the desired targets, which is the objective function.

## Code Structure

The project structure is organized as follows:

### Core Scripts

- **main.m**: The main script that orchestrates the execution of the project, including dataset generation, model training, and evaluation.
- **generate_dataset.m**: This script generates a sample dataset that we can use to test our problem.

### ELM Scripts

- **activation_function.m**: Script for the implementation of the sigmoid activation function.
- **elm_basic.m**: Contains the implementation of the basic ELM neural network model.
- **initialize_weights.m**: Script for initializing the output weights W2 of the ELM model.

### Objective Function Script

- **objective_function.m**: Script that defines the objective function, representing the error (MSE) between the predicted outputs of the ELM and the target values (plus the regularization L2 parameter).

### Cholesky Method Scripts

- **normal_eq.m**: Script that implements the normal equations for finding the closed-form solution to optimize the model weights.
- **cholesky_factorization.m**: Implements the Cholesky factorization on the matrix Q and return the matrix L for the normal equations.
- **forward_substitution.m**: Implements forward substitution, used in conjunction with Cholesky factorization, to resolve the normal equations.
- **backward_substitution.m**: Implements backward substitution, used in the context of solving linear system.

### BFGS Method Scripts 

- **bfgs.m**: Implementation of the BFGS optimization algorithm, the Quasi-Newton method used to iteratively optimize the model output weights W2.
- **line_search.m**: Implements a line search method, which is used within the BFGS algorithm to find the optimal step size during the optimization process.
- **compute_gradient.m**: Script to compute the gradient of the objective function, used in gradient-based optimization methods.
- **bfgs_for_analysis.m**: A variant of the BFGS algorithm tailored for performance analysis and comparison with other methods.

## Execution and production of results

To reproduce the results just run the **main.m** code. To view the results in a multi-digit format, first type the **format longg** command.


