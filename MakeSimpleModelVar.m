function modelVar = MakeSimpleModelVar()
%MakeSimpleModelVar Creates variational dynamics model for nonlinear 9-state system
% for computing the variational observability Gramian

% Get the base model
model = MakeSimpleModel();

N = model.N;

% Create symbolic variables for state and perturbation
x = sym('x', [N, 1]);
delta_x = sym('delta_x', [N, 1]);

% Augmented state vector [x; δx]
x_aug = [x; delta_x];

% Original dynamics: ẋ = f(x)
f_x = model.SS;

% Jacobian: J(x) = ∂f/∂x
J_x = model.J_symbolic;

% Variational dynamics: δẋ = J(x)·δx
f_delta = J_x * delta_x;

% Combined augmented dynamics: [ẋ; δẋ] = [f(x); J(x)·δx]
f_aug = [f_x; f_delta];

% Create function handle for augmented system
modelVar.rhs_aug = matlabFunction(f_aug, 'vars', {x_aug});

% Create Jacobian of augmented system for implicit integration
J_aug_symbolic = jacobian(f_aug, x_aug);
modelVar.jrhs_aug = matlabFunction(J_aug_symbolic, 'vars', {x_aug});

% Save model properties
modelVar.N = N;
modelVar.rhs = model.rhs;
modelVar.jrhs = model.jrhs;
modelVar.jacobian = model.jrhs;  % Alias for AVarDyn compatibility
modelVar.f_aug = f_aug;
modelVar.J_aug_symbolic = J_aug_symbolic;
modelVar.species = model.species;

end
