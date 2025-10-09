# Observability for Nonlinear Systems: Connecting Variational Dynamics, Lyapunov Exponents, and Empirical Gramians

This code studying variational observability Gramian while analyzing combustion reaction networks observability. The code accompanies the research paper mentioned below.

## Requirements

- MATLAB (tested with recent versions)
- Cantera 2.6.0 and 3.0.0 (for chemical reaction network modeling) 
   (Please note that Cantera's Matlab toolbox is not included in Cantera 3.1.0. A new Matlab toolbox is being developed. The code herein requires the legacy Matlab Cantera toolbox to run.)
- Symbolic Math Toolbox
- Optimization Toolbox

## Files Description

### Main Files
- `main.m` - Main script that demonstrates all methods (VarGram, EmpGram, sensor placement)

### Chemical System Models
- `MakeModel.m` - Creates symbolic chemical reaction network model
- `AMakeModelVar.m` - Extended model with state-variational mappings for variational analysis
- `h2o2.yaml` - H2O2 reaction mechanism (9 or 10 species)
- `gri30.yaml` - GRI30 reaction mechanism (53 species)

### Time Integration
- `TimeIntegrateIRK.m` - Implicit Runge-Kutta integration
- `ATimeIntegrateIRK_Jac.m` - IRK integration with Jacobian computation
- `AVarModelTimeIntegrateMatlab45.m` - Variational system integration
- `AVarDyn.m` - Variational dynamics function
- `MakeDiscretization.m` - Time discretization setup

### Observability Analysis
- `AVarObsGram.m` - Variationa observability Gramian
- `AEmpObsGram.m` - Empirical observability Gramian

### Sensor Placement
- `GetOptimalSensorPlacement_Heu_2.m` - Greedy sensor placement algorithm
- `GetNonParticipants.m` - Identify non-participating species

## Usage

1. Run the main script:
   ```matlab
   main.m
   ```

2. The script will:
   - Load H2O2 (9 species) and GRI30 (53 species) chemical systems
   - Compute variational Gramian observability matrices
   - Compute empirical Gramian observability matrices
   - Perform optimal sensor placement using greedy algorithm

## Chemical Systems

### H2O2 System (9 species)
- Small hydrogen-oxygen reaction network
- Used for detailed analysis and illustration

### GRI30 System (53 species)
- Large methane combustion mechanism
- Used for scalability demonstration

### VarGram (Variational Gramian)

### EmpGram (Empirical Gramian)

### Sensor Placement
Greedy algorithm:
- Maximizes logdeterminant of observability Gramian
- Considers non-participating species constraints

## Output

The code generates:
- Observability Gramian matrices
- Optimal sensor placement configurations

## Citations and toolboxes
1. Cantera toolbox: David G. Goodwin, Harry K. Moffat, Ingmar Schoegl, Raymond L. Speth, and Bryan W. Weber. Cantera: An object-oriented software toolkit for chemical kinetics, thermodynamics, and transport processes. https://www.cantera.org, 2024. Version 3.0.0. doi:10.5281/zenodo.14455267
2. For Combustion Reaction MakeModel.m code: A. Haber, F. Molnar, and A. E. Motter, "State Observation and Sensor Selection for Nonlinear Networks," IEEE Transactions on Control of Network Systems, vol. 5, no. 2, pp. 694-708, 2018.

## Citing this code
If you use this code, please cite the accompanying paper:

" M. H. Kazma and A. F. Taha, "Observability for Nonlinear Systems: Connecting Variational Dynamics, Lyapunov Exponents, and Empirical Gramians."

## Contact

For questions or issues, please contact the authors.

