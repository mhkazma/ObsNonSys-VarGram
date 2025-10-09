function [Y1] = EstimationY1IRK(x, model, disc)    
% the objective function used in optimization    
    N = model.N;
    timestep = disc.timestep;
    

    dqDdzeta1 = cell(disc.nSteps, 1);
    dqDdzeta2 = cell(disc.nSteps, 1);
    dx_kDdxkm1 = cell(disc.nSteps, 1);

    I21 = [speye(N, N); speye(N, N)];
    I22 = speye(2*N,2*N);
    I11 = speye(N, N);

    % simulate the system from current initial guess
    [~, zeta1, zeta2] = TimeIntegrateIRK(model, disc, x);
    
    % form the partial derivatives 
    for j = 1 : disc.nSteps - 1
        
        % this is dq/dzeta1 substituted by a real value
        dqDdzeta1{j} = model.jflux(zeta1(:,j));
        dqDdzeta2{j} = model.jflux(zeta2(:,j));

        xx11 = model.Nu * dqDdzeta1{j};
        xx22 = model.Nu * dqDdzeta2{j};

        A11 =[(1/4) * timestep * xx11, -(1/4) * timestep * xx22; 
             (3/12) * timestep * xx11, (5/12) * timestep * xx22];

        stmp = (I22-A11)\I21;

        dx_kDdxkm1{j} = I11 + ...
            (1/4) * timestep * model.Nu * (dqDdzeta1{j} * stmp(1:N,:) + ...
            3 * dqDdzeta2{j} * stmp(N+1 : 2*N,:));

    end

     % form the Jacobian matrix
     tempX = I11;
     J = cell(disc.nSteps-1, 1);
     for j = 1 : disc.nSteps-1
         J{j} = -tempX;
         tempX = dx_kDdxkm1{j} * tempX;
     end

     Y1 = cell2mat(J);     
 end