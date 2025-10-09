function [xsim, zeta1, zeta2] = TimeIntegrateIRK( model, disc, x0 )
% TimeIntegrateIRK runs time integration using implicit runge-kutta method
% Uses the given model and discretization setup (see MakeDiscretization)

% number of species
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
  
% alloc results
xsim = zeros(N, nSteps+1);
zeta1 = zeros(N, nSteps);
zeta2 = zeros(N, nSteps);
 
% timesteps
xsim(:,1) = x0;
for i = 1 : nSteps
          
   xx0 = [xsim(:,i); xsim(:,i)];
   xsimTemp = fsolve(@nonlinear4, xx0, options);
   
   zeta1(:,i) = xsimTemp(1 : N); 
   zeta2(:,i) = xsimTemp(N+1 : 2*N);
   
   xsim(:, i+1) = xsim(:,i) + 0.25 * timestep * (model.rhs(xsimTemp(1:N)) + ...
    3 * model.rhs(xsimTemp(N+1 : 2*N)));
    
end


function [F, J] = nonlinear4(xx)
  
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


end