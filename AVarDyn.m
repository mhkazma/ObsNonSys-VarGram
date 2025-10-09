function dxdt_aug = AVarDyn(t, x_aug, modelVar)
    % Separate the original state and the perturbation (variation)
    N = length(x_aug) / 2;
    x = x_aug(1:N);       % Original state
    delta_x = x_aug(N+1:end);  % Perturbation

    % Nonlinear dynamics: dx/dt = f(x)
    dxdt = modelVar.rhs(x);

    % Variational dynamics: d(delta_x)/dt = A(x) * delta_x
    A_x = modelVar.jacobian(x);  % Jacobian of the system at x
    ddelta_xdt = A_x * delta_x;

    % Combine both dynamics into an augmented state vector
    dxdt_aug = [dxdt; ddelta_xdt];
end