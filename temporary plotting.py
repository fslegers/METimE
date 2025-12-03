import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

plt.rcParams.update({
    "font.size": 16,       # base font size
    "axes.titlesize": 24,  # title size
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ------------------------------
# Data Generation
# ------------------------------
np.random.seed(42)  # for reproducibility
n_samples = 30
X = np.linspace(-3, 3, n_samples)
y_true = np.sin(X)                      # underlying function
y = y_true + np.random.normal(scale=0.4, size=n_samples)  # noisy observations
X = X[:, np.newaxis]

# ------------------------------
# Plot setup
# ------------------------------
plt.figure(figsize=(18, 5))

# 1️⃣ Linear Regression
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

plt.subplot(1, 3, 1)
plt.scatter(X, y, color="black", s=50, label="Data")
plt.plot(X, y_pred_linear, color="red", linewidth=4, label="Linear fit")
#plt.plot(X, y_true, color="blue", linestyle="--", linewidth=2, label="True function")
plt.title("Linear Regression")
#plt.legend()
#plt.grid(alpha=0.3)

# 2️⃣ Polynomial Regression (Overfitting)
degree_high = 20
model_poly_high = make_pipeline(PolynomialFeatures(degree_high), LinearRegression())
model_poly_high.fit(X, y)
y_pred_poly_high = model_poly_high.predict(X)

plt.subplot(1, 3, 2)
plt.scatter(X, y, color="black", s=50, label="Data")
plt.plot(X, y_pred_poly_high, color="red", linewidth=4,
         label=f"Degree {degree_high} fit")
#plt.plot(X, y_true, color="blue", linestyle="--", linewidth=2, label="True function")
plt.title("Polynomial Regression")
#plt.legend()
#plt.grid(alpha=0.3)

# 3️⃣ Sparse Polynomial Regression (Lasso)
alpha = 0.005
model_lasso = make_pipeline(PolynomialFeatures(degree_high),
                            Lasso(alpha=alpha, max_iter=10000))
model_lasso.fit(X, y)
y_pred_lasso = model_lasso.predict(X)

plt.subplot(1, 3, 3)
plt.scatter(X, y, color="black", s=50, label="Data")
plt.plot(X, y_pred_lasso, color="red", linewidth=4, label="Lasso fit")
#plt.plot(X, y_true, color="blue", linestyle="--", linewidth=2, label="True function")
plt.title("Sparse Polynomial Regression")
#plt.legend()
#plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig('regression plot.png', dpi=300, transparent=True)
plt.show()