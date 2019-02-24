from optim import qnm

# Function to optimize
def f(x, y): return (x+1)**2 + y**2 / 2

# Unconstrained form of the constrains g(x, y) <= 0
g = [
    lambda x, y: x - 3,
    lambda x, y: -y,
    lambda x, y: y - x**3 / 8
]

# Definition of the hyperparameters
h = 10**-8
epsilon = 10**-6
eta = 1.2

# Define a seed point
x_opt = (0, 0)

# Define the penalty function
def p(x, y):
    out = map((lambda g_i: max(0, g_i(x, y))**2), g)
    return sum(out)

# Penalty method main loop
c = 1
while True:
    x_old = x_opt
    x_opt = qnm((lambda x, y: f(x, y) + c * p(x, y)), x_old, h)
    c *= eta
    if abs(f(*x_old) - f(*x_opt)) < epsilon: break

# Print the results
print("\n- Solution Found:\n\tx_opt = {0} \n\tf(x_opt) = {1}\n".format(x_opt, f(*x_opt)))
