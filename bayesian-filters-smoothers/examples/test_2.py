import casadi
import matplotlib.pyplot as plt

opti = casadi.Opti()

x = opti.variable()
y = opti.variable()

opti.minimize((1-x)**2+(y-x**2)**2)

opti.solver('ipopt')
sol = opti.solve()

# Plotting Figures
plt.figure()

# Plotting True States of Nonlinear System
plt.subplot(111)
plt.plot(sol.value(x), sol.value(y), marker='o', linewidth=3)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$y$', fontsize=12)
plt.title('CasADi Optimization with IPOPT Solver', fontsize=14)
#plt.legend(loc='upper right')
plt.grid(True)
plt.show()
