function np = GetNonParticipants( model )
%Gets the non-participating species that should be always observed in C

if ~isfield(model, 'Nu_p') || ~isfield(model, 'Nu_r')
    np = sparse(false(model.N, 1));
    return;
end

K = model.Nu_p - model.Nu_r;

%species that don't participate in any reaction are always observed
np = all(K==0, 2); %all-zero row in net stoichiometric matrix


end