function [xsim, zeta1, zeta2, Jacobian_IRK] = ATimeIntegrateIRK_Jac(model,disc,x0)
%ATimeIntegrateIRK_Jac Summary of this function goes here
%Detailed explanation goes here:
%This fxn simulates the nonlinear system under IRK discretization.
%Output includes the simulated states (xsim) and the Jacobian of the 
%discrete IRK system denoted as Jacobian_IRK = delx_{j}\delx_{j-i}

%% number of species and check for consistent initial conditions
N = length(x0);
if (model.N ~= N)
    error('inconsitent number of species in model vs. initial condition');
end

% cache locals
timestep = disc.timestep;
nSteps = disc.nSteps;
rhs = model.rhs;
jrhs = model.jrhs;


% options for fsolve
options = optimset(...
    'MaxFunEvals', 50000, ...
    'MaxIter', 50000, ...
    'TolFun', 1e-16, ...
    'TolX', 1e-16, ...
    'LargeScale', 'on', ...
    'Jacobian', 'on', ...
    'Display','off');

% alloc results; initiate cells
xsim = zeros(N, nSteps+1);
zeta1 = zeros(N, nSteps);
zeta2 = zeros(N, nSteps);
Jacobian_IRK = cell(1, nSteps);


xsim(:,1) = x0;
% solve for initial Jacobian at x = x0
Jacobian_IRK{1} = Jac_nonlinear44([x0;x0]);

% simulate system and obtain Jac
for i = 1 : nSteps

    xx0 = [xsim(:,i); xsim(:,i)];
    xsimTemp = fsolve(@nonlinear4, xx0, options);

    zeta1(:,i) = xsimTemp(1 : N);
    zeta2(:,i) = xsimTemp(N+1 : 2*N);

    xsim(:, i+1) = xsim(:,i) + 0.25 * timestep * (model.rhs(xsimTemp(1:N)) + ...
        3 * model.rhs(xsimTemp(N+1 : 2*N)));

    zeta1(:,i+1) = xsim(1 : N);
    zeta2(:,i+1) = xsim(N+1 : 2*N);

    %calculating the jacobain
    %Jacobian_IRK{i+1} = Jac_nonlinear4(xsimTemp);
    Jacobian_IRK{i+1} = Jac_nonlinear4(zeta1(:,i+1),zeta2(:,i+1));
end


%% Local Functions
    function [F, J] = nonlinear4(xx)
    %function to simulate system under IRK 
        % evaluates the RHS at x
        ss11 = rhs(xx(1:N));
        ss22 = rhs(xx(N+1 : 2*N));

        % evaluates the Jacobian of the RHS at x
        ss33 = jrhs(xx(1:N));
        ss44 = jrhs(xx(N+1 : 2*N));

        %compile
        F = xx-[xx0(1:N)+(1/4)*timestep*(ss11-ss22);
            xx0(1:N)+(1/12)*timestep*(3*ss11+5*ss22)];

        J = speye(2*N,2*N) - [
            (1/4) * timestep * ss33, (-1/4) * timestep * ss44;
            (1/12)* timestep * 3 * ss33, (1/12) * timestep * 5 * ss44];
    end


    function [J] = Jac_nonlinear44(xx)
        %evaluates the Jacobian of the RHS at x = x0
        ss33 = jrhs(xx(1:N));
        ss44 = jrhs(xx(N+1 : 2*N));

        J = speye(N,N) - (1/4) * timestep * ss33 + (+3/4) * timestep * ss44;
        J =sparse(J);
    end

    function [J] = Jac_nonlinear4(zeta1,zeta2)
        %evaluates the Jacobian of the RHS at
        I21 = [speye(model.N, model.N); speye(model.N, model.N)];
        I22 = speye(2*model.N,2*model.N);
        I11 = speye(model.N, model.N);

        dqDdzeta1 = model.jflux(zeta1);
        dqDdzeta2 = model.jflux(zeta2);

        xx11 = model.Nu * dqDdzeta1;
        xx22 = model.Nu * dqDdzeta2;

        A11 =[(1/4) * timestep * xx11, -(1/4) * timestep * xx22;
            (3/12) * timestep * xx11, (5/12) * timestep * xx22];

        stmp = (I22-A11)\I21;

        dx_kDdxkm1 = I11 + ...
            (1/4) * timestep * model.Nu * (dqDdzeta1 * stmp(1:N,:) + ...
            3 * dqDdzeta2 * stmp(N+1 : 2*N,:));

        J = dx_kDdxkm1;
        J =sparse(J);
    end

end