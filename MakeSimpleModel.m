function model = MakeSimpleModel()
%MakeSimpleModel Creates a nonlinear 9-state dynamical system

% Number of species/states
N = 9;

% Reaction rate constants
k = [1.0, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];

% Create symbolic variables for the states
S = sym('x', [N, 1]);

% Define the nonlinear ODE system: ẋ = f(x)
SS = sym(zeros(N, 1));

% S1: decays and feeds S2
SS(1) = -k(1)*S(1) + 0.1*S(9);  % small feedback from S9

% S2: receives from S1, decays and feeds S3
SS(2) = k(1)*S(1) - k(2)*S(2);

% S3: receives from S2, decays and feeds S4, with nonlinear term
SS(3) = k(2)*S(2) - k(3)*S(3) - 0.05*S(3)*S(6);

% S4: receives from S3, decays and feeds S5
SS(4) = k(3)*S(3) - k(4)*S(4);

% S5: receives from S4, decays and feeds S6, with saturation
SS(5) = k(4)*S(4) - k(5)*S(5)/(1 + 0.1*S(5));

% S6: receives from S5, decays and feeds S7
SS(6) = k(5)*S(5)/(1 + 0.1*S(5)) - k(6)*S(6);

% S7: receives from S6, decays and feeds S8 and S9
SS(7) = k(6)*S(6) - k(7)*S(7);

% S8: receives from S7, decays
SS(8) = k(7)*S(7) - k(8)*S(8);

% S9: receives from S7, decays with feedback to S1
SS(9) = 0.5*k(7)*S(7) - k(9)*S(9);

% Create MATLAB function handles for numerical integration
model.rhs = matlabFunction(SS, 'vars', {S});

% Compute Jacobian of the RHS
J_symbolic = jacobian(SS, S);
model.jrhs = matlabFunction(J_symbolic, 'vars', {S});

% Save model properties
model.N = N;
model.SS = SS;
model.k = k;
model.J_symbolic = J_symbolic;

% Species names for reference
model.species = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'};

fprintf('Number of states: %d\n', N);

% Check stability at operating point
x_check = ones(N, 1);  % Check around all ones
J_check = model.jrhs(x_check);
eig_check = eig(J_check);
fprintf('Stability check at x=[1,...,1]:\n');
fprintf('  Max real part of eigenvalues: %.4f', max(real(eig_check)));
if max(real(eig_check)) < 0
    fprintf(' (STABLE)\n');
else
    fprintf(' (UNSTABLE)\n');
end

end
