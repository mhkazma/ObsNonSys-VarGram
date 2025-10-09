function [Y1] = EstimationY1VarIRK(x, model, disc, Jac_IRK_input)    
% VarIRK version - uses precomputed IRK Jacobians from ATimeIntegrateIRK_Jac
% Same structure as EstimationY1IRK but leverages variational IRK approach
    N = model.N;
    
    % Use precomputed IRK Jacobians instead of recomputing
    % Jac_IRK_input contains the dx_kDdxkm1 matrices from ATimeIntegrateIRK_Jac
    
    % Form the Jacobian matrix using the same structure
    tempX = speye(N, N); % I11 identity matrix
    J = cell(disc.nSteps-1, 1);
    
    for j = 1 : disc.nSteps-1
        J{j} = -tempX;
        
        % Use precomputed IRK Jacobian instead of manual computation
        if j+1 <= length(Jac_IRK_input)
            tempX = Jac_IRK_input{j+1} * tempX;
        end
    end

    Y1 = cell2mat(J);     
end