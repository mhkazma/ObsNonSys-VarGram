function [W0_Emp] = AEmpObsGram(C_mat,x, model, disc,sys)    
% the objective function used in optimization 
%[W0_Emp] = AEmpObsGram(C_mat,x, model, disc,sys)  
    
    %Define Eps for building NonLin-Emp-Gram
    Eps = sys.Eps;
    
    %N = model.N;
    %timestep = disc.timestep;

    %dqDdzeta1 = cell(disc.nSteps, 1);
    %dqDdzeta2 = cell(disc.nSteps, 1);
    %dx_kDdxkm1 = cell(disc.nSteps, 1);

    %I21 = [speye(N, N); speye(N, N)];
    %I22 = speye(2*N,2*N);
    %I11 = speye(N, N);

    % Create X0 perturbed by eps under the action of vector ei
    Identity = eye(model.N);
    x_eps_p = zeros(model.N,model.N);
    x_eps_m = zeros(model.N,model.N);
    
    for idx2=1:model.N
        e = Identity(:,idx2);
        x_eps_p(:,idx2) = x + Eps*e;
        x_eps_m(:,idx2) = x - Eps*e;
    end 

    xsim_p = cell(1, model.N);
    zeta1_p = cell(1, model.N);
    zeta2_p = cell(1, model.N);

    xsim_m = cell(1, model.N);
    zeta1_m = cell(1, model.N);
    zeta2_m = cell(1, model.N);

    % simulate the system from current perturbed initial guess
    for idx2=1:model.N
        [xsim_p{1,idx2}, zeta1_p{1,idx2}, zeta2_p{1,idx2}] = ...
            TimeIntegrateIRK(model, disc, x_eps_p(:,idx2));
        [xsim_m{1,idx2}, zeta1_m{1,idx2}, zeta2_m{1,idx2}] = ...
            TimeIntegrateIRK(model, disc, x_eps_m(:,idx2)); % should be x
    end

    % Build C_mat with sensor placement
    sensed_nodes = zeros(model.N, 1);
    sensed_nodes(~sys.np) = C_mat;
    sensed_nodes(sys.np) = 1;
    sensed_loc = find(sensed_nodes);

    %Building C_mat
    C_sp = sparse(1:length(sensed_loc), sensed_loc, ...
        ones(length(sensed_loc), 1), length(sensed_loc), model.N);

    %Build Emperical NonLinObs Gramian
    W0_Emp  = zeros(model.N,model.N);
    for idx2=1:disc.nSteps-1
        deltaY = zeros(model.N,model.N);
        for idx1=1:model.N
            deltaY(:,idx1) =xsim_p{idx1}(:,idx2) - xsim_m{idx1}(:,idx2);
        end
        W0_Emp = W0_Emp + (deltaY'*(C_sp'*C_sp)*deltaY);
    end
    W0_Emp = (1/(4*sqrt(Eps)))*W0_Emp;
 end