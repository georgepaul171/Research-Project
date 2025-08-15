import numpy as np
import matplotlib.pyplot as plt

# Create constraint regions
theta = np.linspace(0, 2 * np.pi, 100)
# L2: circle
l2_x = np.cos(theta)
l2_y = np.sin(theta)
# L1: diamond
l1_x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))
l1_y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))

fig, ax = plt.subplots(figsize=(6, 6))

# L2 constraint (circle)
ax.plot(l2_x, l2_y, 'b-', label='L2 constraint (circle)')
# L1 constraint (diamond)
ax.plot(l1_x, l1_y, 'r--', label='L1 constraint (diamond)')

# Loss contours for illustration
x = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 0.7) ** 2 + (Y + 0.5) ** 2
ax.contour(X, Y, Z, levels=6, colors='gray', alpha=0.5)

# Solution points
ax.plot([0.7], [-0.5], 'ko', label='Unregularized solution')
ax.plot([1, 0], [0, 1], 'go', label='L1 solution (sparse)')
ax.plot([0.5], [-0.2], 'mo', label='L2 solution (shrinkage)')

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('Coefficient 1')
ax.set_ylabel('Coefficient 2')
ax.set_title('L1 vs L2 Regularization')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('l1_vs_l2_regularization.png', dpi=300)
plt.show() 