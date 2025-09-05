import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors


def plot_transition_landscapes():
    params = {
        'b': 0.2, 'd': 0.2, 'Ec': 450, 'm': 437.3,
        'w': 1.0, 'w1': 0.42, 'mu_meta': 0.0215,
        'd0': 0,
        'd1': 0,
        'N': 600,
        'E': 60000
    }

    # Build grid (avoid zero)
    n_vals = np.linspace(1, params['N'], 100, dtype=float)
    e_vals = np.linspace(1, params['E'], 100, dtype=float)
    n_grid, e_grid = np.meshgrid(n_vals, e_vals)

    # Define functions
    def f_func(n, e):
        return (params["b"] - params["d"] * params["E"] / params["Ec"]) * n / (e**(1/3)) \
               + params["m"] * n / params["N"]

    def h_func(n, e):
        return (params["w"] - params["d"] * params["E"] / params["Ec"]) * n * (e**(2/3)) \
               + params["m"] * n / params["N"]

    # Evaluate on grid
    f_vals = f_func(n_grid, e_grid)
    h_vals = h_func(n_grid, e_grid)

    def safe_clip(Z):
        Z = np.nan_to_num(Z, nan=0.0, posinf=np.nanmax(Z[np.isfinite(Z)]), neginf=np.nanmin(Z[np.isfinite(Z)]))
        return np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))

    f_vals = safe_clip(f_vals)
    h_vals = safe_clip(h_vals)

    norm_f = colors.Normalize(vmin=f_vals.min(), vmax=f_vals.max())
    norm_h = colors.Normalize(vmin=h_vals.min(), vmax=h_vals.max())

    # Plot surfaces + contour projections
    fig = plt.figure(figsize=(16, 6))
    for i, (Z, label, norm) in enumerate(zip([f_vals, h_vals], ["f", "h"], [norm_f, norm_h])):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        surf = ax.plot_surface(
            n_grid, e_grid, Z,
            facecolors=cm.viridis(norm(Z)),
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=False, alpha=0.9
        )
        # Add contour projection on xy-plane
        ax.contour(n_grid, e_grid, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.7)

        ax.set_xlabel("n")
        ax.set_ylabel("ε")
        ax.set_zlabel(label)
        ax.set_title(f"Landscape of {label}(n, ε)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_transition_landscapes()
