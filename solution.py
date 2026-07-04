import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

df = pd.read_csv('xy_data.csv')
x_target = df['x'].values
y_target = df['y'].values
target_points = np.column_stack((x_target, y_target))

def calculate_geometric_l1_loss(params):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    # Reduced to 300 points for much faster calculation without losing accuracy
    t_dense = np.linspace(6, 60, 300)

    x_pred = t_dense * np.cos(theta) - np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense) * np.sin(theta) + X
    y_pred = 42 + t_dense * np.sin(theta) + np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense) * np.cos(theta)
    
    pred_points = np.column_stack((x_pred, y_pred))
    
    distances = cdist(target_points, pred_points, metric='cityblock')
    l1_distance = np.mean(np.min(distances, axis=1))
    
    return l1_distance

bounds = [
    (0, 50),
    (-0.05, 0.05),
    (0, 100)
]

print("Starting optimization using Differential Evolution...")
result = differential_evolution(
    calculate_geometric_l1_loss,
    bounds,
    strategy='best2bin',
    popsize=15, # Reduced from 50 to 15 for faster iterations
    mutation=(0.5, 1.5),
    tol=1e-6
)

optimal_theta, optimal_M, optimal_X = result.x
minimized_loss = result.fun

print(f"Minimum Geometric L1 Distance Achieved: {minimized_loss:.6f}")
print(f"Estimated Theta : {optimal_theta:.4f}")
print(f"Estimated M     : {optimal_M:.6f}")
print(f"Estimated X     : {optimal_X:.4f}")

t_final = np.linspace(6, 60, len(x_target))
optimal_theta_rad = np.deg2rad(optimal_theta)

x_pred_final = t_final * np.cos(optimal_theta_rad) - np.exp(optimal_M * np.abs(t_final)) * np.sin(0.3 * t_final) * np.sin(optimal_theta_rad) + optimal_X
y_pred_final = 42 + t_final * np.sin(optimal_theta_rad) + np.exp(optimal_M * np.abs(t_final)) * np.sin(0.3 * t_final) * np.cos(optimal_theta_rad)

submission_string = (
    r'"\left(t*\cos({:.4f})-e^{{{:.4f}\left|t\right|}}\cdot\sin(0.3t)\sin({:.4f})'
    r'+{:.4f},42+t*\sin({:.4f})+e^{{{:.4f}\left|t\right|}}\cdot\sin(0.3t)\cos({:.4f})\right)"'
).format(
    optimal_theta_rad, optimal_M, optimal_theta_rad, 
    optimal_X, 
    optimal_theta_rad, optimal_M, optimal_theta_rad
)

print("\nFormatted string for your GitHub repo README.md:")
print(submission_string)
print("="*50)

plt.figure(figsize=(10, 6))
plt.plot(x_target, y_target, 'b.', label='Actual Data', alpha=0.6)
plt.plot(x_pred_final, y_pred_final, 'r-', label=f'Predicted (Loss: {minimized_loss:.4f})', linewidth=2)
plt.title('Parametric Curve Optimization (Geometric Match)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()