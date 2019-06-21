import numpy as np
import matplotlib.pyplot as plt


def generate_noisy_data(polynomial, lim, std, num=50):
    # x = (np.random.rand(num) - 0.5) * lim * 2
    x = np.linspace(0, lim, num)
    n = np.random.randn(num) * std
    return x, polynomial(x) + n


lim = 4
roots = [0, 1, 2.5, 4]
p = np.poly1d(roots, r=True)
x = np.linspace(0, lim, 100)
deg = [len(roots) + 1, 30]

noisy_data = generate_noisy_data(p, lim=lim, std=1.0, num=5)

fig, ax = plt.subplots()
# lab = ' '.join([f'{c}x^{len(p.coef) - i - 1}' for i, c in enumerate(p.coef)])
ax.plot(x, p(x), label=str(p))
ax.scatter(*noisy_data)
for d, s in zip(deg, ['--', '-.']):
    fit = np.polyfit(*noisy_data, deg=d)
    pfit = np.poly1d(fit)
    ax.plot(x, pfit(x), s, label=str(d) + ' Parameters')
ax.legend()

fig.set_size_inches(w=4.5, h=2.5)
fig.tight_layout()
fig.savefig('polynomial.png', bbox_inches='tight')
plt.show()
