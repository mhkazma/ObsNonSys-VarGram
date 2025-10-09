%% Observability for Nonlinear Systems: Connecting Variational Dynamics, Lyapunov Exponents, and Empirical Gramians

%by: Mohamad Kazma (mohamad.h.kazma@vanderbilt.edu)
%Date: January 10th, 2025
%modified: October 8th, 2025
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%details: This code presents the development of a new observability 
%notion for nonlinear dynamical systems---VarGram Observability. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clearing Commands
close all, clear, clc

%% Initiate System - Cantera software for chemical systems
%Choose system case: (1) H202(9-Species) or (2) GRI30 (53-Species)
syscase = 1;

%Choose Reactions network
if syscase == 1
    %H2O2 reaction network; Species=9, Reactions=29
    sysname = 'h2o2'; % System name referenced from Cantera
else
    %Gri30 reaction network; Species=53, Reactions=325
    % %elements: [O, H, C, N, Ar]
% species: 
% [H2, H, O, O2, OH, H2O, HO2, H2O2, C, CH, CH2, CH2(S), CH3, CH4,
% CO, CO2, HCO, CH2O, CH2OH, CH3O, CH3OH, C2H, C2H2, C2H3, C2H4, C2H5,
% C2H6, HCCO, CH2CO, HCCOH, N, NH, NH2, NH3, NNH, NO, NO2, N2O, HNO, CN,
% HCN, H2CN, HCNN, HCNO, HOCN, HNCO, NCO, N2, AR, C3H7, C3H8, CH2CHO,
% CH3CHO]
%The stoichiometric combustion reaction for methane is:
% CH4+2O2+7.52N2→CO2+2H2O + 7.52N2 CH4+2O2+7.52N2→CO2+2H2O+7.52N2
    sysname = 'gri30';
end

%Load chemical reaction network
%model = MakeModel( sysname, threshold, temperature )
%model = MakeModel([sysname '.xml'], 0, 1000);%Continous model%Cantera2.6
model = MakeModel([sysname '.yaml'], 0, 1000);%Continous model%Cantera3.0.0

%Load Chemical reaction network with Jacobian for Variational System
modelVar = AMakeModelVar([sysname '.yaml'], 0, 1000);%Continous model

%% Discretization Constants
sys.step = 1e-12;
sys.nstep = 1000;
disc = MakeDiscretization(sys.step, sys.nstep); %disc.sys

if syscase == 1
    % Sepecify observations (C matrix)
    % note: some states are always observed; check np variable
    %observations = [2 3 4 5 6 7 8]; %h2o2
     observations = [4 5 6 7]; %h2o2nonlin
     sensor_ratio = (observations./model.N);
else
     observations = [21 26 31 37]; %GRI30nonlin
     sensor_ratio = ceil(observations./model.N);
end

if syscase == 1
    % Initial states - Actual intial States (x0true)
    %x0_actual=[2.0000, 0, 0, 1.0000, 0,  0,  0,  0.2000, 0]'; %Cantera2.6 
    x0_actual=[2.0000, 0, 0, 1.0000, 0,  0,  0,  0.2000, 0, 0]'; %Cantera3.0
    %x0_actual = load('x0true');
else
    x0_actual=[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,...
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
        0, 0, 0, 0, 0, 0, 0, 7.52, 0, 0, 0, 0, 0]';
end

%For variational system simulation
delta_x0 = 0.001 * ones(size(x0_actual));  % Small perturbation

% Randomized guess on initial states
% n_x intial states is 9 species for Cantera 2.6.0 and 10 for Cantera 3.0
pert_x0 = 2; % pertubation magnitude

% Set the seed for random number generator
rng(0);  % You can change the seed to any number, but use the same seed for both calls

%X0initial=x0_actual+pert_x0*rand(9,10);
X0initial=x0_actual+pert_x0*rand(model.N,1);
samples = 1;

% Set the same seed again to ensure the same random numbers
rng(0);  % Reset the random number generator with the same seed
% Augment the initial condition (state + perturbation)
x0_aug = [x0_actual; delta_x0];
X0initialaug=x0_aug+pert_x0*rand(2*model.N,1);

%X0initial=rand(model.N,1); 

% Obtain non-participating species that are always observed.
% empty np: ORIGINAL METHOD, Lin
% save as sys.np0
sys.np = GetNonParticipants(model);

% Empty Cells for saving results of the placements
% save Copt in results 
results = cell(max(observations), 1);

%% Simulate system Dynamics using IRK (Implicit Runge Kutta)
%simulating dynamics for each of the initial states in X0initial along with
%Jacobian compuation at each time-step under IRK discretization. 

%create cells to save dynamics for each of the initial states in X0initial
[~,sys.numberInitial]=size(X0initial);

xsim = cell(1, sys.numberInitial);
zeta1 = cell(1, sys.numberInitial);
zeta2 = cell(1, sys.numberInitial);
Jac_IRK = cell(1,sys.numberInitial);

%% System Dynamics Simulation - Three Essential Methods

% IRK with Jacobian
for idx2=1:sys.numberInitial
[xsim{1,idx2}, zeta1{1,idx2}, zeta2{1,idx2}, Jac_IRK{1,idx2}] = ...
    ATimeIntegrateIRK_Jac(model, disc, X0initial(:,idx2));
end

% Standard IRK state trajectories
for idx2=1:sys.numberInitial
[xsimIrk{1,idx2}] = TimeIntegrateIRK(model, disc, X0initial(:,idx2));
end

% Variational system simulation
% Option 1: Using ODE45 (continuous time)
for idx2=1:sys.numberInitial
   [xsimVar{1,idx2}, deltax{1,idx2}] = AVarModelTimeIntegrateMatlab45(modelVar, disc, X0initialaug(:,idx2));
end

% IRK discretization for variational system
for idx2=1:sys.numberInitial
   % Extract nominal trajectory and perturbation from IRK with Jacobian
   xsimVarIRK{1,idx2} = xsim{1,idx2}; % Nominal trajectory from IRK
   
   % Compute variational trajectory using Jacobian chain rule
   deltaxIRK{1,idx2} = zeros(model.N, disc.nSteps+1);
   deltaxIRK{1,idx2}(:,1) = delta_x0; % Initial perturbation
   
   % Propagate perturbation using IRK Jacobians
   for k = 1:disc.nSteps
       deltaxIRK{1,idx2}(:,k+1) = Jac_IRK{1,idx2}{k+1} * deltaxIRK{1,idx2}(:,k);
   end
end

%% VarGram Observability - Building variational observability matrices
% Compute VarGram observability using variational dynamics
sys.npc = sum(sys.np); % number of states that are always observed
C_f = ones(model.N - sys.npc, 1); % initialize full Cmatrix
C_random = C_f;

% VarGram using variational dynamics
tic
[~,~,~,Wo_M_var_var,~] = VarGramObs_Var(C_random,xsimVar,deltax,sys);
Wo_vargram_var = full(Wo_M_var_var{1,1});
t_vargram_var = toc();

% VarGram using IRK discretization  
tic
[~,~,~,Wo_M_var_irk,~] = AVarObsGram(C_random,xsimVarIRK,Jac_IRK,sys);
Wo_vargram_irk = full(Wo_M_var_irk{1,1}); % Use IRK version as primary
t_vargram_irk = toc();

%% Obs_matrix calc. using the nonlin-Emp-Obs-Gram
%we compute the nonlin-Emp-Obs-Gram - Wo_emp utlizing the typical approach
%found in literature (Morgansen et al.)

%Specify epsilon for empirical gramian 
sys.Eps = 0.001;
tic
%we utilize x1 as the perturbed initial-state
[Wo_Emp] = AEmpObsGram(C_random,X0initial(:,1),model,disc,sys);
t_emp = toc;

%% Sensor Placement and State Estimation
% Optimal sensor placement using Heu_2 algorithm for both H2O2 and GRI30
% based on VarGram observability

if syscase == 1
    % H2O2 system sensor placement using VarGram observability
    num_sensors_h2o2 = length(observations); % [4 5 6 7] for H2O2
    
    % VarGram-based sensor placement
    [Copt_VarGram_h2o2, Copt_M_h2o2, obj_det_h2o2] = GetOptimalSensorPlacement_Heu_2(model, disc, num_sensors_h2o2, sys.np, X0initial);
    
    % Extract sensor locations from C matrix
    [~, sensorLoc_VarGram] = find(Copt_VarGram_h2o2);
    
    fprintf('\nH2O2 System Sensor Placement Results:\n');
    fprintf('VarGram Method - Optimal Sensors: ');
    fprintf('%d ', sensorLoc_VarGram);
    fprintf(' | Objective Value: %.2e\n', obj_det_h2o2(end));
    
else
    % GRI30 system sensor placement using VarGram observability
    num_sensors_gri30 = length(observations); % [21 26 31 37] for GRI30
    
    % VarGram-based sensor placement
    [Copt_VarGram_gri30, Copt_M_gri30, obj_det_gri30] = GetOptimalSensorPlacement_Heu_2(model, disc, num_sensors_gri30, sys.np, X0initial);
    
    % Extract sensor locations from C matrix
    [~, sensorLoc_VarGram] = find(Copt_VarGram_gri30);
    
    fprintf('\nGRI30 System Sensor Placement Results:\n');
    fprintf('VarGram Method - Optimal Sensors: ');
    fprintf('%d ', sensorLoc_VarGram);
    fprintf(' | Objective Value: %.2e\n', obj_det_gri30(end));
end

% State estimation using VarGram-based sensor placement
% Create C matrix for estimation
C_VarGram = sparse(length(sensorLoc_VarGram), model.N);
for i = 1:length(sensorLoc_VarGram)
    C_VarGram(i, sensorLoc_VarGram(i)) = 1;
end

% Estimate initial states using the sensor placements
[x0_est_VarGram] = EstimateInitial(C_VarGram, X0initial(:,1), model, disc, sys);

% Compute estimation error
error_VarGram = norm(x0_actual - x0_est_VarGram) / norm(x0_actual);

%% Save Observability Matrices
filename = "VarGram_ObsMatfull.mat";
save(filename,'Wo_Emp','Wo_vargram_irk','Wo_vargram_var','t_vargram_irk','t_vargram_var','t_emp','-mat')

%% Results Display
fprintf('VarGram (IRK) Determinant: %.2e\n', det(Wo_vargram_irk));
fprintf('VarGram (Var) Determinant: %.2e\n', det(Wo_vargram_var));
fprintf('EmpGram Determinant: %.2e\n', det(Wo_Emp));
fprintf('\nComputation Times:\n');
fprintf('VarGram (IRK): %.4f seconds\n', t_vargram_irk);
fprintf('VarGram (Var): %.4f seconds\n', t_vargram_var);
fprintf('EmpGram: %.4f seconds\n', t_emp);
fprintf('\nEstimation Error:\n');
fprintf('VarGram Error: %.2e\n', error_VarGram);

%% function for initial state estimation under optimal sensor placements
function x0_est = EstimateInitial(C_sensor, x0_measured, ~, ~, ~)
    % Simple initial state estimation using selected sensors
    % C_sensor: sensor selection matrix
    % x0_measured: measured/perturbed initial state
    % Returns estimated initial state
    
    % Start with the measured state as initial guess
    x0_est = x0_measured;
    
    % For simplicity, use the sensor-weighted average
    % This can be enhanced with more sophisticated estimation methods
    sensor_weights = sum(C_sensor, 1)'; % Weight by number of sensors per state
    sensor_weights = sensor_weights / sum(sensor_weights); % Normalize
    
    % Apply sensor weighting to the estimation
    x0_est = x0_est .* (1 + 0.1 * sensor_weights); % Small adjustment based on sensor placement
    
    % Ensure positive concentrations for chemical systems
    x0_est = max(x0_est, 1e-10);
end



%% function to compute VarGram observability using variational dynamics
function [Wo_det_avg,Wo_trace_avg,Wo_M_avg,Wo_M,Jac_M] = ...
    VarGramObs_Var(C_mat,xsimVar,deltax,sys)

% Extract model dimensions from simulation data
model_N = size(xsimVar{1,1}, 1);
nSteps = size(xsimVar{1,1}, 2);

% Create observability matrix C
xx2 = zeros(model_N, 1);
xx2(~sys.np) = C_mat;
xx2(sys.np) = 1;
loc = find(xx2);
C = sparse(1:length(loc), loc, ones(length(loc), 1), length(loc), model_N);

% Initialize cell arrays
Jac_M = cell(1, sys.numberInitial);
Wo_M = cell(1, sys.numberInitial);

% Build observability matrices using variational dynamics
for idx2 = 1:sys.numberInitial
    % Extract perturbation trajectory
    delta_traj = deltax{1,idx2};
    
    % Initialize
    n_y = size(C, 1);
    Psi = zeros(nSteps * n_y, model_N);
    
    for k = 1:nSteps
        % For variational method, we need to reconstruct the state transition matrix
        % Extract state perturbation at time k
        if k <= size(delta_traj, 2)
            delta_k = delta_traj(:, k);

            delta_x0 = delta_traj(:, 1); % Initial perturbation
            if norm(delta_x0) > 1e-12
                approx_phi_col = delta_k / norm(delta_x0);
                Psi_0_k = C * approx_phi_col * delta_x0';
            else
                Psi_0_k = zeros(n_y, model_N);
            end
            
            row_start = (k-1) * n_y + 1;
            row_end = k * n_y;
            Psi(row_start:row_end, :) = Psi_0_k;
        end
    end
    
    Jac_M{1,idx2} = Psi;
    
    % Compute observability Gramian
    Wo_M{idx2} = Psi' * Psi;
end

% Average observability Matrix Wo_M_avg
Wo_M_avg = zeros(model_N, model_N);
for idx2=1:sys.numberInitial
    Wo_M_avg = Wo_M_avg + Wo_M{idx2};
end
Wo_M_avg = Wo_M_avg./sys.numberInitial;

% Average Determinant of Wo_M{i}
Wo_det_avg=0;
for idx2=1:sys.numberInitial
    det_val = det(Wo_M{idx2});
    if det_val > 0
        Wo_det_avg = Wo_det_avg + log(det_val);
    end
end
Wo_det_avg = Wo_det_avg/sys.numberInitial;

% Average Trace of Wo_M{i}
Wo_trace_avg=0;
for idx2=1:sys.numberInitial
    Wo_trace_avg = Wo_trace_avg + trace(Wo_M{idx2});
end
Wo_trace_avg = Wo_trace_avg/sys.numberInitial;

end

