function model = AMakeModelVar(sysname, threshold, temperature)
%MakeModel Creates a symbolic model of the given chemical reaction network
%Returns the rhs (xdot=f(x)), the jacobian of the flux vector,
%and the jacobian of the rhs. Any reaction with rate constant lower than 
%threshold will be ignored.

if nargin < 2
    threshold = 0;
end

if nargin < 3
    temperature = 2500;
end


% Load chemical reaction network (using cantera)
g = Solution(sysname);  %'h2o2.xml' 'GRI30.cti'
set(g, 'T', temperature, 'P', oneatm);

% forward and backward reaction rates
kf = fwdRateConstants(g);
kb = revRateConstants(g);

% filter those with rates less than threshold
filter_fwd = kf < threshold;
filter_bck = kb < threshold;
kf(filter_fwd) = 0;
kb(filter_bck) = 0;
model.filter_fwd = filter_fwd;
model.filter_bck = filter_bck;

% stoichiometric coefficients
Nu_r = stoich_r(g);    % reactant stoichiometric coefficient matrix 
Nu_p = stoich_p(g);    % product stoichiometric coefficient matrix

Nu = Nu_p - Nu_r; % net stoichiometric matrix (NOTE: species appearing on both sides with same stoichiometry will cancel out)

N = size(Nu, 1);    % number of species
R = size(Nu, 2);    % number of reactions

fprintf('Species: %d\n', N);
fprintf('Reactions: %d\n', R);

% start building the symbolic model
S = sym('x', [N, 1]);
q = sym(zeros(length(kf),1)); %preallocate

% calculate fluxes of each reaction
for i = 1 : R
    q_fwd = prod(S .^ Nu_r(:,i)); %ith reaction, each column is a reaction, rows: coefs for species participating
    q_bck = prod(S .^ Nu_p(:,i));        
    q(i) = kf(i) * q_fwd - kb(i) * q_bck; % sym exp: flux of reaction i    
end

SS = Nu * q;

% make rhs function for \dot{x}=f(x)
%model.rhs = matlabFunction(SS, 'file', 'rhs.txt', 'vars', {S});
model.rhs = matlabFunction(SS, 'vars', {S});

% jacobian of the flux vector
dqDdx = jacobian(q, S);

% formulate it as a function
%model.jflux = matlabFunction(dqDdx, 'file', ['jflux' name], 'vars', {S});
model.jflux = matlabFunction(dqDdx, 'vars', {S});

% jacobian of rhs
%model.jrhs = matlabFunction(K * dqDdx, 'file', ['jrhs' name], 'vars', {S});
model.jrhs = matlabFunction(Nu * dqDdx, 'vars', {S});

% Compute Jacobian of the system
model.jacobian = matlabFunction(jacobian(SS, S), 'vars', {S});

% save
model.N = N;
%model.g = g;
model.Nu_p = Nu_p;
model.Nu_r = Nu_r;
model.Nu = Nu;
model.kf = kf;
model.kb = kb;
model.SS = SS;

model.species = speciesName(g, 1:N);

end