function disc = MakeDiscretization( timestep, nSteps )
%MakeDiscretization Creates options for time discretization methods

disc.timestep = timestep;
disc.nSteps = nSteps;
disc.timespan = 0 : timestep : nSteps * timestep;


end