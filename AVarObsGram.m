function [Wo_det_avg,Wo_trace_avg,Wo_M_avg,Wo_M,Jac_M] = ...
    AVarObsGram(C_mat,xsimVar,Jac_IRK_input,sys)
%AVarObsGram Computes VarGram observability using IRK discretization

% Extract model dimensions from simulation data
model_N = size(xsimVar{1,1}, 1);
nSteps = size(xsimVar{1,1}, 2) - 1;

% Create observation matrix C (∂h(x_k)/∂x_k for linear output)
xx2 = zeros(model_N, 1);
xx2(~sys.np) = C_mat;
xx2(sys.np) = 1;
loc = find(xx2);
C = sparse(1:length(loc), loc, ones(length(loc), 1), length(loc), model_N);

% Initialize cell arrays
Jac_M = cell(1, sys.numberInitial);
Wo_M = cell(1, sys.numberInitial);

% Build observability matrix
for idx2 = 1:sys.numberInitial
    
    % Initialize the observability matrix 
    n_y = size(C, 1); % number of outputs
    Psi = zeros(nSteps * n_y, model_N);
    
    % For each time step k = 0, 1, ..., N-1
    for k = 1:nSteps
        
        % Compute state transition matrix Φ_0^k
        if k == 1
            Phi_0_k = Jac_IRK_input{idx2}{k+1};
        else
            Phi_0_k = Jac_IRK_input{idx2}{2};
            for i = 2:k
                if i+1 <= length(Jac_IRK_input{idx2})
                    Phi_0_k = Jac_IRK_input{idx2}{i+1} * Phi_0_k;
                end
            end
        end
        
        % Compute variational measurement mapping
        Psi_0_k = C * Phi_0_k;
        
        % Fill the k-th block of the observability matrix Ψ
        row_start = (k-1) * n_y + 1;
        row_end = k * n_y;
        Psi(row_start:row_end, :) = Psi_0_k;
    end
    
    % Store the observability matrix
    Jac_M{1,idx2} = Psi;
    
    % Compute VarGram:
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