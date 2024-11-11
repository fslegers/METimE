import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad


def dn_dt(n, e, state_variables):
    _, N, _ = state_variables
    # return (1.48 * n -0.0002 * e -2.04e-05*n*e -33.6 * n/N) # LinearRegression
    # return 1.225 * n - 0.00017 * e -1.58e-05*n*e -2.28 * n/N # RidgeRegression
    return 1.223 * n - 1.266 * e - 0.122 * n * e - 2.27 * n / N  # RidgeRegression with e -> e/7692


def R_exponent(n, e, state_variables, lambdas):
    _, N, _ = state_variables
    l1, l2, l3 = lambdas
    return -l1 * n - l2 * n * e - l3 * dn_dt(n, e, state_variables)


def partition_function(state_variables, lambdas):
    S, N, E = state_variables
    return sum(quad(lambda e: np.exp(R_exponent(n, e, state_variables, lambdas)), 0, E)[0] for n in range(1, N))


def entropy(lambdas, state_variables):
    _, N, E = state_variables
    I = sum(quad(lambda e: np.exp(R_exponent(n, e, state_variables, lambdas)) * R_exponent(n, e, state_variables, lambdas),0, E)[0] for n in range(1, N+1))
    return I


def constraint1(lambdas, state_variables):
    S, N, E = state_variables
    lhs = N / S
    rhs = sum(n * quad(lambda e: np.exp(R_exponent(n, e, state_variables, lambdas)), 0, E)[0] for n in range(1, N+1))
    return lhs - rhs


def constraint2(lambdas, state_variables):
    S, N, E = state_variables
    lhs = E / S
    rhs = sum(n * quad(lambda e: e * np.exp(R_exponent(n, e, state_variables, lambdas)), 0, E)[0] for n in range(1, N+1))
    return lhs - rhs


def constraint3(lambdas, state_variables, derivatives):
    S, N, E = state_variables
    _, dN_dt, _ = derivatives
    lhs = dN_dt
    rhs = S * sum(quad(lambda e: dn_dt(n, e, state_variables) * np.exp(R_exponent(n, e, state_variables, lambdas)), 0, E)[0] for n in range(1, N+1))
    return lhs - rhs


# Toy example:
# state_variables = [24,114,7692]
# derivatives = [0, -0.6, -12583]
# initial_lambdas = [0.075059926, 0.003167063, 0]

state_variables = [24,114,1]
derivatives = [0, -0.6, 0]
initial_lambdas = [0.075059926, 0.003167063, 0]


constraints = [
    {'type': 'eq', 'fun': constraint1, 'args': (state_variables,)},
    {'type': 'eq', 'fun': constraint2, 'args': (state_variables,)},
    {'type': 'eq', 'fun': constraint3, 'args': (state_variables, derivatives)}]


boundaries = ((0, None), (0, None), (0, None))


# Run the optimization
result = minimize(entropy, initial_lambdas, args=(state_variables,), constraints=constraints, bounds=boundaries)


# Results
optimized_lambdas = result.x

print("Optimized Lagrange multipliers:", optimized_lambdas)
print("Maximum value of I:", result.fun)


# Turn results into SAD
S, N, E = state_variables
dynaSAD = []
dynaSAD_cum = []

total=0
for n in range(1, N + 1):
    p_n = quad(lambda e: np.exp(R_exponent(n, e, state_variables, optimized_lambdas)), 0, E)[0]
    Z = partition_function(state_variables, optimized_lambdas)
    p_n = p_n/Z
    dynaSAD.append(p_n)

    total += p_n
    dynaSAD_cum.append(total)


plt.plot(range(1, N + 1), dynaSAD)
plt.title("SAD")
plt.xlim((0.9, 4))
plt.show()


plt.plot(range(1, N + 1), dynaSAD_cum)
plt.title("Cumulative SAD")
plt.xlim((0.9, 4))
plt.show()



