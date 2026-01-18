"""
Load and view the Simple_VarGram_ObsMat.mat results
"""

import scipy.io
import numpy as np

# Load the .mat file
mat_file = 'Simple_VarGram_ObsMat.mat'
print(f"Loading {mat_file}...")

try:
    data = scipy.io.loadmat(mat_file)
    
    # Extract the main variables
    Wo_vargram = data['Wo_vargram']  # VarGramObs
    Wo_Emp = data['Wo_Emp']          # EmprGramObs
    t_vargram = data['t_vargram'][0, 0]
    t_emp = data['t_emp'][0, 0]
    xsim = data['xsim']
    x0_actual = data['x0_actual']
    X0initial = data['X0initial']
    
    print(f"\nSuccessfully loaded data from {mat_file}")
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Gramian properties
    print(f"\nVarGramObs (Jacobian Method):")
    print(f"  Shape: {Wo_vargram.shape}")
    print(f"  Condition number: {np.linalg.cond(Wo_vargram):.2e}")
    print(f"  Log-determinant: {np.log(np.linalg.det(Wo_vargram)):.6e}")
    print(f"  Trace: {np.trace(Wo_vargram):.6e}")
    print(f"  Computation time: {t_vargram:.4f} seconds")
    
    print(f"\nEmpGram (Empirical Gramian):")
    print(f"  Shape: {Wo_Emp.shape}")
    print(f"  Condition number: {np.linalg.cond(Wo_Emp):.2e}")
    print(f"  Log-determinant: {np.log(np.linalg.det(Wo_Emp)):.6e}")
    print(f"  Trace: {np.trace(Wo_Emp):.6e}")
    print(f"  Computation time: {t_emp:.4f} seconds")
    
    # Speedup
    speedup = t_emp / t_vargram
    print(f"\nComputational Speedup: {speedup:.2f}x")
    
    # Comparison
    rel_diff = np.linalg.norm(Wo_vargram - Wo_Emp, 'fro') / np.linalg.norm(Wo_vargram, 'fro')
    print(f"\nComparison:")
    print(f"  Relative difference (Frobenius): {rel_diff*100:.4f}%")
    
    # Eigenvalues
    eig_vargram = np.sort(np.linalg.eigvals(Wo_vargram))[::-1]
    eig_emp = np.sort(np.linalg.eigvals(Wo_Emp))[::-1]
    
    print(f"\nVarGramObs eigenvalues (sorted):")
    for i, ev in enumerate(eig_vargram):
        print(f"  λ_{i+1}: {ev:.6e}")
    
    print(f"\nEmpGram eigenvalues (sorted):")
    for i, ev in enumerate(eig_emp):
        print(f"  λ_{i+1}: {ev:.6e}")
    
    # System trajectory info
    print(f"\nSystem Trajectory:")
    print(f"  Shape: {xsim.shape} (states x time steps)")
    print(f"  Initial state (actual): {x0_actual.flatten()}")
    print(f"  Initial state (perturbed): {X0initial.flatten()}")

    
except FileNotFoundError:
    print(f"Error: {mat_file} not found.")
    print("Please run main_simple.m first to generate the results file.")
except Exception as e:
    print(f"Error loading file: {e}")
