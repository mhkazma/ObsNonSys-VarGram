function [xsim, deltaxsim] = AVarModelTimeIntegrateMatlab45(modelVar, disc, x0)
%AVarModelTimeIntegrateMatlab45 runs time integration using builtin ode45 matlab func
% LinearizedSystem computes the linearized system at the operating point x0_actual
% Input:
%   model - the symbolic model created from MakeModel
%   x0_actual - the operating point around which we are linearizing
% Output:
%   A_lin - linearized system matrix A
%   B_lin - linearized input matrix B (if applicable, otherwise can be zeros)

% number of species
N = length(x0);
if (2*modelVar.N ~= N) 
    error('inconsitent number of species in model vs. initial condition');
end

% Simulate the augmented system using ode45
[~, x_aug] = ode45(@(t, x_aug) AVarDyn(t, x_aug, modelVar), disc.timespan, x0);
%[~, y] = ode45(@(t,x) linear_system(t,x,A_lin,x0_actual), disc.timespan, x0);

xsim = x_aug(:, 1:(modelVar.N))';
deltaxsim = x_aug(:, (modelVar.N)+1:end)';

%xsim = y.';

end