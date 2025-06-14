import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
import itertools

def load_and_prepare_data(file1_path, file2_path, file3_path):
    """
    Load the three CSV files and prepare interpolation functions
    """
    # Load data
    data1 = pd.read_csv(file1_path, header=None, names=['x', 'y'])
    data2 = pd.read_csv(file2_path, header=None, names=['x', 'y'])
    data3 = pd.read_csv(file3_path, header=None, names=['x', 'y'])
    
    # Sort by x values to ensure proper interpolation
    data1 = data1.sort_values('x')
    data2 = data2.sort_values('x')
    data3 = data3.sort_values('x')
    
    # Create interpolation functions
    f1_interp = interp1d(data1['x'], data1['y'], kind='cubic', fill_value='extrapolate')
    f2_interp = interp1d(data2['x'], data2['y'], kind='cubic', fill_value='extrapolate')
    f3_interp = interp1d(data3['x'], data3['y'], kind='cubic', fill_value='extrapolate')
    
    # Get x-axis bounds
    x_min = max(data1['x'].min(), data2['x'].min(), data3['x'].min())
    x_max = min(data1['x'].max(), data2['x'].max(), data3['x'].max())
    
    return f1_interp, f2_interp, f3_interp, x_min, x_max

def calculate_c3(c1, c2):
    """
    Calculate c3 using the given formula
    c3 = (c1 - c2)/(125+10)*60 + c1 - 125*(c1 - c2)/(125+10)
    """
    return (c1 - c2)/135*60 + c1 - 125*(c1 - c2)/135

def get_derivative(func, x, dx=1e-6):
    """
    Calculate numerical derivative of function at point x
    """
    return derivative(func, x, dx=dx)

def check_conditions(c1, c2, k, f1, f2, f3, targets, error_tolerance=0.05, debug=False):
    """
    Check if the conditions are met within error tolerance
    
    Parameters:
    - c1, c2: x-axis values
    - k: scaling factor
    - f1, f2, f3: interpolation functions
    - targets: [d_1, d_2, d_3, d_4]
    - error_tolerance: allowed relative error (default 5%)
    - debug: if True, print detailed calculation steps
    
    Returns:
    - bool: True if all conditions are met
    - dict: actual values and errors
    
    Conditions:
    1) f1(c2) - f1(c1) >= d_1  (inequality constraint)
    2) df1(c3) = d_2           (equality within tolerance)
    3) df2(c1) = d_3           (equality within tolerance)
    4) df3(c2) = d_4           (equality within tolerance)
    """
    d_1, d_2, d_3, d_4 = targets
    
    if debug:
        print(f"    DEBUG: Checking c1={c1:.3f}, c2={c2:.3f}, k={k:.2f}")
        print(f"    DEBUG: Targets = {targets}")
        print(f"    DEBUG: Condition 1: f1(c2)-f1(c1) >= {d_1}")
        print(f"    DEBUG: Conditions 2-4: equalities within {error_tolerance:.1%} tolerance")
    
    # Calculate c3
    c3 = calculate_c3(c1, c2)
    if debug:
        print(f"    DEBUG: Calculated c3 = {c3:.3f}")
    
    # Create scaled functions
    f1_scaled = lambda x: k * f1(x)
    f2_scaled = lambda x: k * f2(x)
    f3_scaled = lambda x: k * f3(x)
    
    # Calculate actual values with error handling
    try:
        actual_1 = f1_scaled(c2) - f1_scaled(c1)  # f1(c2) - f1(c1) >= d_1
        if debug:
            print(f"    DEBUG: f1({c2:.3f}) = {f1_scaled(c2):.3f}, f1({c1:.3f}) = {f1_scaled(c1):.3f}")
            print(f"    DEBUG: actual_1 = {actual_1:.3f}, must be >= {d_1:.3f}")
            print(f"    DEBUG: Condition 1 satisfied: {actual_1 >= d_1}")
    except Exception as e:
        if debug:
            print(f"    DEBUG: Error calculating actual_1: {e}")
        raise
    
    try:
        actual_2 = get_derivative(f1_scaled, c3)   # df1(c3) = d_2
        if debug:
            print(f"    DEBUG: df1({c3:.3f}) = {actual_2:.3f}, target_2 = {d_2:.3f}")
    except Exception as e:
        if debug:
            print(f"    DEBUG: Error calculating actual_2: {e}")
        raise
    
    try:
        actual_3 = get_derivative(f2_scaled, c1)   # df2(c1) = d_3
        if debug:
            print(f"    DEBUG: df2({c1:.3f}) = {actual_3:.3f}, target_3 = {d_3:.3f}")
    except Exception as e:
        if debug:
            print(f"    DEBUG: Error calculating actual_3: {e}")
        raise
    
    try:
        actual_4 = get_derivative(f3_scaled, c2)   # df3(c2) = d_4
        if debug:
            print(f"    DEBUG: df3({c2:.3f}) = {actual_4:.3f}, target_4 = {d_4:.3f}")
    except Exception as e:
        if debug:
            print(f"    DEBUG: Error calculating actual_4: {e}")
        raise
    
    # Check condition 1: inequality constraint f1(c2) - f1(c1) >= d_1
    condition_1_met = actual_1 >= d_1
    
    # For condition 1, calculate how much it exceeds the minimum (for reporting)
    excess_1 = actual_1 - d_1 if condition_1_met else d_1 - actual_1
    
    # Calculate relative errors for conditions 2-4 (equality constraints)
    error_2 = abs(actual_2 - d_2) / abs(d_2) if d_2 != 0 else abs(actual_2)
    error_3 = abs(actual_3 - d_3) / abs(d_3) if d_3 != 0 else abs(actual_3)
    error_4 = abs(actual_4 - d_4) / abs(d_4) if d_4 != 0 else abs(actual_4)
    
    # Check if equality conditions are met
    condition_2_met = error_2 <= error_tolerance
    condition_3_met = error_3 <= error_tolerance
    condition_4_met = error_4 <= error_tolerance
    
    if debug:
        print(f"    DEBUG: Condition 1 (>=): {condition_1_met} (excess: {excess_1:.3f})")
        print(f"    DEBUG: Condition 2 (=): {condition_2_met} (error: {error_2:.1%})")
        print(f"    DEBUG: Condition 3 (=): {condition_3_met} (error: {error_3:.1%})")
        print(f"    DEBUG: Condition 4 (=): {condition_4_met} (error: {error_4:.1%})")
    
    # All conditions must be met
    all_met = condition_1_met and condition_2_met and condition_3_met and condition_4_met
    
    if debug:
        print(f"    DEBUG: All conditions met: {all_met}")
    
    results = {
        'c3': c3,
        'actual_values': [actual_1, actual_2, actual_3, actual_4],
        'target_values': targets,
        'condition_1_met': condition_1_met,
        'excess_1': excess_1,
        'errors': [excess_1 if not condition_1_met else 0, error_2, error_3, error_4],  # Modified for inequality
        'equality_errors': [error_2, error_3, error_4],  # Just the equality errors
        'all_conditions_met': all_met
    }
    
    return all_met, results

def optimize_parameters(f1, f2, f3, x_min, x_max, targets, error_tolerance=0.05, 
                       c_step=0.1, k_step=0.1, verbose=True):
    """
    Find optimal c1, c2, and k values that satisfy the conditions
    
    Parameters:
    - f1, f2, f3: interpolation functions
    - x_min, x_max: bounds for c1 and c2
    - targets: [d_1, d_2, d_3, d_4]
    - error_tolerance: allowed relative error
    - c_step: step size for c1 and c2 search
    - k_step: step size for k search
    - verbose: if True, print detailed progress
    
    Returns:
    - list of solutions
    """
    solutions = []
    
    # Generate search ranges
    c_values = np.arange(x_min, x_max + c_step, c_step)
    k_values = np.arange(0.5, 3.0 + k_step, k_step)
    
    print("=" * 60)  # This creates a line of 60 equal signs for visual separation
    print("OPTIMIZATION SETUP")
    print("=" * 60)
    print(f"X-axis bounds: [{x_min:.2f}, {x_max:.2f}]")
    print(f"C-step size: {c_step}")
    print(f"K-step size: {k_step}")
    print(f"C-values to test: {len(c_values)} ({c_values[0]:.2f} to {c_values[-1]:.2f})")
    print(f"K-values to test: {len(k_values)} ({k_values[0]:.1f} to {k_values[-1]:.1f})")
    print(f"Error tolerance: {error_tolerance:.1%}")
    print(f"Target values: {targets}")
    
    # Calculate total valid combinations
    valid_combinations = 0
    for c1 in c_values:
        for c2 in c_values:
            if c2 > c1 + 0.5:
                valid_combinations += 1
    
    total_combinations = valid_combinations * len(k_values)
    print(f"Valid (c1,c2) pairs: {valid_combinations}")
    print(f"Total combinations to check: {total_combinations}")
    
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION")
    print("=" * 60)
    
    combinations_checked = 0
    numerical_errors = 0
    constraint_violations = 0
    best_near_misses = []  # Track close solutions even if they don't meet all criteria
    
    for k_idx, k in enumerate(k_values):
        print(f"\n--- Testing k = {k:.1f} ({k_idx+1}/{len(k_values)}) ---")
        k_solutions = 0
        k_combinations = 0
        
        for c1_idx, c1 in enumerate(c_values):
            if verbose and c1_idx % max(1, len(c_values)//10) == 0:
                print(f"  c1 progress: {c1_idx+1}/{len(c_values)} (c1={c1:.2f})")
            
            for c2 in c_values:
                # Check constraint: c2 > c1 + 0.5
                if c2 <= c1 + 0.5:
                    constraint_violations += 1
                    continue
                
                combinations_checked += 1
                k_combinations += 1
                
                try:
                    conditions_met, results = check_conditions(c1, c2, k, f1, f2, f3, 
                                                             targets, error_tolerance)
                    
                    # Print detailed info for promising combinations
                    max_equality_error = max(results['equality_errors'])
                    condition_1_ok = results['condition_1_met']
                    
                    # Show info if condition 1 is met and equality errors are reasonable
                    if condition_1_ok and max_equality_error <= error_tolerance * 2:
                        print(f"    Testing: c1={c1:.2f}, c2={c2:.2f}, c3={results['c3']:.2f}")
                        print(f"      Condition 1 (>=): {'✓' if condition_1_ok else '✗'} "
                              f"(actual={results['actual_values'][0]:.3f} >= {d_1:.3f}, "
                              f"excess={results['excess_1']:.3f})")
                        print(f"      Equality errors: {[f'{e:.1%}' for e in results['equality_errors']]}")
                        print(f"      Max equality error: {max_equality_error:.1%} "
                              f"{'✓ SOLUTION!' if conditions_met else '(close)'}")
                    
                    if conditions_met:
                        solution = {
                            'c1': c1,
                            'c2': c2,
                            'c3': results['c3'],
                            'k': k,
                            'actual_values': results['actual_values'],
                            'condition_1_met': results['condition_1_met'],
                            'excess_1': results['excess_1'],
                            'equality_errors': results['equality_errors'],
                            'max_equality_error': max_equality_error
                        }
                        solutions.append(solution)
                        k_solutions += 1
                        
                        print(f"    *** SOLUTION FOUND #{len(solutions)} ***")
                        print(f"        c1={c1:.3f}, c2={c2:.3f}, c3={results['c3']:.3f}, k={k:.1f}")
                        print(f"        Condition 1: f1(c2)-f1(c1)={results['actual_values'][0]:.3f} >= {d_1:.3f} ✓")
                        print(f"        Max equality error: {max_equality_error:.2%}")
                        
                    elif condition_1_ok and max_equality_error <= error_tolerance * 1.5:  # Track very close misses
                        best_near_misses.append({
                            'c1': c1, 'c2': c2, 'c3': results['c3'], 'k': k,
                            'max_equality_error': max_equality_error, 
                            'condition_1_met': condition_1_ok,
                            'excess_1': results['excess_1']
                        })
                
                except Exception as e:
                    numerical_errors += 1
                    if verbose and numerical_errors <= 5:  # Show first few errors
                        print(f"    Numerical error at c1={c1:.2f}, c2={c2:.2f}, k={k:.1f}: {str(e)[:50]}...")
                
                # Frequent progress updates
                if combinations_checked % 1000 == 0:
                    progress = combinations_checked / total_combinations * 100
                    print(f"  Progress: {combinations_checked}/{total_combinations} ({progress:.1f}%) - Solutions: {len(solutions)}")
        
        print(f"  k={k:.1f} summary: {k_solutions} solutions from {k_combinations} valid combinations")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total combinations checked: {combinations_checked}")
    print(f"Constraint violations: {constraint_violations}")
    print(f"Numerical errors encountered: {numerical_errors}")
    print(f"Solutions found: {len(solutions)}")
    
    # Show near misses if no exact solutions found
    if len(solutions) == 0 and best_near_misses:
        print(f"\nNo exact solutions found, but {len(best_near_misses)} near misses detected.")
        print("Best near misses (condition 1 satisfied, equality errors within 1.5x tolerance):")
        best_near_misses.sort(key=lambda x: x['max_equality_error'])
        for i, miss in enumerate(best_near_misses[:3]):  # Show top 3
            print(f"  {i+1}. c1={miss['c1']:.2f}, c2={miss['c2']:.2f}, k={miss['k']:.1f}")
            print(f"      Condition 1: ✓ (excess: {miss['excess_1']:.3f})")
            print(f"      Max equality error: {miss['max_equality_error']:.1%}")
    
    return solutions

def print_solutions(solutions, targets):
    """
    Print solutions in a readable format
    """
    if not solutions:
        print("No solutions found within the specified error tolerance.")
        return
    
    print(f"\nFound {len(solutions)} solution(s):")
    print("="*80)
    
    target_names = ['f1(c2)-f1(c1) >= d_1', 'df1(c3) = d_2', 'df2(c1) = d_3', 'df3(c2) = d_4']
    d_1, d_2, d_3, d_4 = targets
    
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i+1}:")
        print(f"  c1 = {sol['c1']:.3f}")
        print(f"  c2 = {sol['c2']:.3f}")
        print(f"  c3 = {sol['c3']:.3f}")
        print(f"  k  = {sol['k']:.1f}")
        print(f"  Max equality error: {sol['max_equality_error']:.1%}")
        
        print(f"\n  Condition check:")
        
        # Condition 1 (inequality)
        actual_1 = sol['actual_values'][0]
        print(f"    1. f1(c2) - f1(c1) >= {d_1:.3f}: actual={actual_1:.3f} "
              f"{'✓' if sol['condition_1_met'] else '✗'} (excess: {sol['excess_1']:.3f})")
        
        # Conditions 2-4 (equalities)
        equality_targets = [d_2, d_3, d_4]
        equality_names = ['df1(c3)', 'df2(c1)', 'df3(c2)']
        
        for j, (name, target_val, actual_val, error) in enumerate(
            zip(equality_names, equality_targets, sol['actual_values'][1:], sol['equality_errors'])):
            print(f"    {j+2}. {name} = {target_val:.3f}: actual={actual_val:.3f}, error={error:.1%}")
            
        print(f"  Overall: {'✓ All conditions satisfied' if sol['condition_1_met'] else '✗ Failed'}")

# Example usage and testing function
def main():
    """
    Main function to demonstrate usage
    Replace with your actual file paths and target values
    """
    
    print("OPTIMIZATION SOLVER - ENHANCED VERSION")
    print("=" * 50)  # This creates a line of 50 equal signs
    print("Template ready for your data!")
    print("=" * 50)
    
    print("\nWhat you need to do:")
    print("1. Update file paths to your CSV files")
    print("2. Set your target values [d_1, d_2, d_3, d_4]")
    print("3. Adjust error tolerance and search parameters")
    print("4. Run the optimization")
    
    print("\nExample usage:")
    print("```python")
    print("# Load your data")
    print("f1, f2, f3, x_min, x_max = load_and_prepare_data('file1.csv', 'file2.csv', 'file3.csv')")
    print("")
    print("# Set your target values")
    print("targets = [1.5, -0.3, 0.8, -1.2]  # [d_1, d_2, d_3, d_4]")
    print("# Note: d_1 is minimum value for f1(c2)-f1(c1), others are exact targets")
    print("")
    print("# Run optimization with verbose output")
    print("solutions = optimize_parameters(f1, f2, f3, x_min, x_max, targets,")
    print("                              error_tolerance=0.05,  # 5% tolerance for equalities")
    print("                              c_step=0.1,           # c1,c2 step size")
    print("                              k_step=0.1,           # k step size")
    print("                              verbose=True)         # detailed output")
    print("")
    print("# Print results")
    print("print_solutions(solutions, targets)")
    print("```")
    
    print("\nCondition Details:")
    print("1. f1(c2) - f1(c1) >= d_1  (inequality - must exceed minimum)")
    print("2. df1(c3) = d_2           (equality within tolerance)")
    print("3. df2(c1) = d_3           (equality within tolerance)")
    print("4. df3(c2) = d_4           (equality within tolerance)")
    
    print("\nReal-time output features:")
    print("- Setup summary before starting")
    print("- Progress updates every 1000 combinations")
    print("- Detailed info for promising combinations")
    print("- Immediate notification when solutions are found")
    print("- Summary of errors and near-misses")
    print("- Debug mode available for troubleshooting")
    
    print("\nTo enable debug mode for specific combinations:")
    print("Add debug=True to check_conditions() call")
    
    print("\nTo use this code:")
    print("1. Install: pip install numpy pandas scipy")
    print("2. Prepare CSVs with format: x,y (no headers)")
    print("3. Set targets and run!")

if __name__ == "__main__":
    main()