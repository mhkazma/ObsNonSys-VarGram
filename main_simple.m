%% Observability for Nonlinear Systems --- Example
%
% This example computes the following.
% 1. Variational observability Gramian (VarGramObs)
% 2. Empirical observability Gramian (EmpGramObs)
% 3. Computational time comparison
%
% System: 9-state nonlinear chemical network
% No external toolboxes required such as Cantera

%by: Mohamad Kazma (mohamad.h.kazma@vanderbilt.edu)
%Date: January 11th, 2026

%% Clearing Commands
close all, clear, clc

fprintf('Example on computing the proposed observability Gramian\n');

%% Create a 9-State Nonlinear System
% Create the nonlinear dynamics model
model = MakeSimpleModel();

% Create variational dynamics model
modelVar = MakeSimpleModelVar();

%% Discretization Constants
sys.step = 1e-3;      % Time step 
sys.nstep = 500;      % Number of steps
disc = MakeDiscretization(sys.step, sys.nstep);

%% Define Sensor Configuration
% Observe 4 out of 9 states (similar to H2O2 example)
observations = [3, 5, 7, 9];  % Observe S3, S5, S7, S9
sensor_ratio = length(observations) / model.N;

fprintf('System Configuration:\n');
fprintf('  States: %d\n', model.N);
fprintf('  Observed states: %d (%.1f%%)\n', length(observations), sensor_ratio*100);
fprintf('  Time horizon: %.3f seconds\n', sys.step * sys.nstep);
fprintf('  Time steps: %d\n\n', sys.nstep);

%% Initial Conditions
% True initial state (all species start with positive concentrations)
x0_actual = [2.0, 1.5, 1.2, 1.0, 0.8, 0.9, 1.1, 0.7, 0.6]';

% Perturbation for variational dynamics
delta_x0 = 0.01 * ones(model.N, 1);

% Perturbed initial guess for estimation
pert_x0 = 0.3;  % perturbation magnitude
rng(42);  % Fixed seed for reproducibility
X0initial = x0_actual + pert_x0 * randn(model.N, 1);
X0initial = max(X0initial, 0.01); 

fprintf('Initial Conditions:\n');
fprintf('  True initial state norm: %.4f\n', norm(x0_actual));
fprintf('  Perturbed initial state norm: %.4f\n', norm(X0initial));
fprintf('  Perturbation magnitude: %.4f\n\n', pert_x0);

%% Setup System Parameters
% Non-participating species (always observed)
sys.np = false(model.N, 1);  % All false (no always-observed states)
sys.npc = sum(sys.np);
sys.numberInitial = model.N;  % N trajectories, one per basis direction

%% Simulate Nonlinear System
fprintf('Simulating nonlinear system\n');
tic;
[xsim, zeta1, zeta2] = TimeIntegrateIRK(model, disc, x0_actual);
t_nominal = toc;
fprintf('Simulation time: %.4f seconds\n', t_nominal);

% Check for numerical issues
if any(isnan(xsim(:))) || any(isinf(xsim(:)))
    error('Numerical instability');
end

%% Simulate Variational System with MATLAB ODE solver
fprintf('Simulating variational system\n');
tic;

% We need N perturbation trajectories (one for each basis direction)
% to properly reconstruct the full state transition matrix
xsimVar = cell(1, model.N);
deltax = cell(1, model.N);

for idx = 1:model.N
    % Create perturbation along basis vector e_idx
    delta_x0_basis = zeros(model.N, 1);
    delta_x0_basis(idx) = 0.01;  % Small perturbation magnitude
    
    x0_basis = [x0_actual; delta_x0_basis];
    
    [xsimVar{1,idx}, deltax{1,idx}] = ...
        AVarModelTimeIntegrateMatlab45(modelVar, disc, x0_basis);
end
t_var_sim_matlab = toc;
fprintf(' Variational simulation time: %.4f seconds\n', t_var_sim_matlab);

%% VarGram Observability Computation
fprintf('Computing Observability Gramians\n');

% Setup observation matrix (observe all non-participating states)
C_random = ones(model.N - sys.npc, 1);

fprintf('VarGramObs:\n');
tic;
Wo_vargram = VarGramObs(C_random, xsim, model, disc, sys);
t_vargram = toc;
fprintf('  Computation time: %.4f seconds\n', t_vargram);
fprintf('  Condition number: %.2e\n', cond(Wo_vargram));
fprintf('  Log-determinant: %.6e\n\n', log(det(Wo_vargram)));

%% Empirical Observability Gramian
fprintf('EmpGramObs\n');
fprintf(' (Perturbs each state independently)\n');
sys.Eps = 0.001;  % Epsilon for finite difference
tic;
[Wo_Emp] = AEmpObsGram(C_random, X0initial, model, disc, sys);
t_emp = toc;
fprintf('  Computation time: %.4f seconds\n', t_emp);
fprintf('  Condition number: %.2e\n', cond(Wo_Emp));
fprintf('  Log-determinant: %.6e\n\n', log(det(Wo_Emp)));

%% Results Summary
fprintf('Results Summary\n');

% Log-Determinants (measure of observability)
logdet_vargram = log(det(Wo_vargram));
logdet_empgram = log(det(Wo_Emp));

fprintf('Observability Measures (Log-Determinants):\n');
fprintf('  VarGram: %.6e\n', logdet_vargram);
fprintf('  EmpGram: %.6e\n', logdet_empgram);
fprintf('  Difference: %.6e\n\n', abs(logdet_vargram - logdet_empgram));

% Trace (alternative observability measure)
trace_vargram = trace(Wo_vargram);
trace_empgram = trace(Wo_Emp);

fprintf('Observability Measures (Traces):\n');
fprintf('  VarGram: %.6e\n', trace_vargram);
fprintf('  EmpGram: %.6e\n', trace_empgram);
fprintf('  Ratio (VarGram/EmpGram): %.4f\n\n', trace_vargram/trace_empgram);

% Computation times
fprintf('Computation Times:\n');
fprintf('  VarGram: %.4f seconds\n', t_vargram);
fprintf('  EmpGram: %.4f seconds\n\n', t_emp);

%% Save Results
fprintf('Saving Results\n');

filename = "Simple_VarGram_ObsMat.mat";
save(filename, 'Wo_Emp', 'Wo_vargram', 't_vargram', 't_emp', ...
     'model', 'xsim', 'x0_actual', 'X0initial', '-mat');

fprintf('Results saved to: %s\n\n', filename);

%% Visualization
fprintf('Creating visualizations\n');

% Plot 1: System trajectories
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
time_vector = 0:disc.timestep:(size(xsim,2)-1)*disc.timestep;
plot(time_vector, xsim', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('State Values');
title('System Trajectories');
legend(model.species, 'Location', 'best', 'FontSize', 8);
grid on;

% Plot 2: Gramian eigenvalues
subplot(1, 3, 2);
eig_var = sort(eig(Wo_vargram), 'descend');
eig_emp = sort(eig(Wo_Emp), 'descend');
semilogy(1:model.N, eig_var, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(1:model.N, eig_emp, 's-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Eigenvalue Index');
ylabel('Eigenvalue Magnitude');
title('Gramian Eigenvalues');
legend('VarGramObs', 'EmpGram', 'Location', 'best');
grid on;

% Plot 3: Computation time comparison
subplot(1, 3, 3);
methods = {'VarGramObs', 'EmpGram'};
times = [t_vargram, t_emp];
bar(times);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('Computation Time (s)');
title('Computational Efficiency');
grid on;
sgtitle('Nonlinear 9-State System: Observability Analysis');

%% Function to compute VarGramObs
function Wo = VarGramObs(C_mat, xsim, model, disc, sys)
% VarGramObs computes observability Gramian along variational system trajectory
%
% Compute Phi_0^k recursively using:
% Phi_0^k = (I + dF/dx_k) * Phi_0^{k-1}

model_N = model.N;
nSteps = size(xsim, 2) - 1;

% Create observability matrix C
if isempty(sys.np) || sum(sys.np) == 0
    xx2 = C_mat;
else
    xx2 = zeros(model_N, 1);
    xx2(~sys.np) = C_mat;
    xx2(sys.np) = 1;
end
loc = find(xx2);
C = sparse(1:length(loc), loc, ones(length(loc), 1), length(loc), model_N);

n_y = size(C, 1);
Psi = zeros(nSteps * n_y, model_N);

% Initialize state transition matrix
Phi_k = eye(model_N);  % Phi_0^0 = I

for k = 1:nSteps
    % Get state at time k
    x_k = xsim(:, k);
    
    % Compute Jacobian at current state
    J_k = model.jrhs(x_k);
    
    % Update state transition matrix: Phi_0^k = (I + dt*J_k) * Phi_0^{k-1}
    % For IRK method: Phi_{k-1}^k = I + dF_tilde/dx_k
    Phi_k = (eye(model_N) + disc.timestep * J_k) * Phi_k;
    
    % Compute Psi_0^k = C * Phi_0^k
    Psi_k = C * Phi_k;
    
    % Fill observability matrix
    row_start = (k-1) * n_y + 1;
    row_end = k * n_y;
    Psi(row_start:row_end, :) = Psi_k;
end

% Compute Gramian
Wo = Psi' * Psi;

end
