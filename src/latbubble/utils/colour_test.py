import matplotlib.pyplot as plt

# Quick code to test different colour schemes for the lattice simulation


final_red = (1.0, 0.3, 0.3)
mid_red   = (0.6, 0.3, 0.3)

final_blue = (0.3, 0.3, 1.0)
mid_blue   = (0.3, 0.3, 0.6)

colors  = [mid_red, final_red, mid_blue, final_blue]
labels  = ["mid_red", "final_red", "mid_blue", "final_blue"]

fig, ax = plt.subplots(figsize=(6, 2))

for i, (col, lab) in enumerate(zip(colors, labels)):
    ax.scatter(i, 0, color=col, s=2000, edgecolor='k', label=lab)

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 1)

ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.tight_layout()
plt.show()