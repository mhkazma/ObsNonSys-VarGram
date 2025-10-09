function [Copt, Copt_M, obj_det] = GetOptimalSensorPlacement_Heu_2( model, disc, nObservations, np, X0initial )

[~,numberInitial]=size(X0initial);

% Compute IRK Jacobians
fprintf('Computing IRK Jacobians for VarIRK sensor placement...\n');
Jac_IRK_all = cell(1, numberInitial);

for idx2=1:numberInitial
    [~, ~, ~, Jac_IRK_all{idx2}] = ATimeIntegrateIRK_Jac(model, disc, X0initial(:,idx2));
end

for idx2=1:numberInitial
    Y1{1,idx2} = EstimationY1VarIRK(X0initial(:,idx2), model, disc, Jac_IRK_all{idx2});
end

%% solve 
npc = sum(np);
obj_det = zeros(1,nObservations-npc);
xx = zeros(model.N - npc, 1);
g_last = 0;

for i = 1 : nObservations-npc
    %choose the ith observed species
    
    allowed = find(xx == 0);
    len = size(allowed, 1);
    
    maxGain = -inf;
    maxJ = -1;
    g_best = 0;
    
    for j = 1 : len
        xx( allowed(j)) = 1;
        
        value = fitnessfcn(xx);
        gain = value - g_last;
        
        if (gain > maxGain)
            %edited by mkazma
            %maxGain = value;
            maxGain = gain;
            
            maxJ = j;
            g_best = value;
        end
        
        xx (allowed(j)) = 0;
    end
    
    % make the minimal selected
    xx(allowed(maxJ)) = 1;
    g_last = g_best;
    obj_det(1,i) = maxGain;
    
    fprintf('  %e\n', g_last);
    
end

% formulate optimal C matrix
xx1 = zeros(model.N, 1);
xx1(~np) = xx;
xx1(np) = 1;
pos = find(xx1);
Copt = sparse(1:length(pos), pos, ones(length(pos), 1), length(pos), model.N);
Copt_M = xx1;


function fun = fitnessfcn(x)
    
    %x is sparse column vector, ones where we have a chosen one
    xx2 = zeros(model.N, 1);
    xx2(~np) = x;
    xx2(np) = 1;
    loc = find(xx2);
    C = sparse(1:length(loc), loc, ones(length(loc), 1), length(loc), model.N);
    [s1,~]=size(Y1{:,1});
    
    for idx2=1:numberInitial
        M{1,idx2}=kron(speye(s1/model.N,s1/model.N),C)*Y1{1,idx2};
    end
    
    [sizeM1,sizeM2]=size(M{1,1});
    fun=0;
    for idx2=1:numberInitial
        fun=fun+log(det(M{1,idx2}' * M{1,idx2}));
    end
    
    fun=fun/numberInitial;
    
    %fun=condest(M'*M);
    %fun=trace(M'*M);
end

end