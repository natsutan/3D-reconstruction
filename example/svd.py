import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

def show(name, value):
    """計算結果を表示する。"""
    print(name)
    print(value)
    print()


A = np.array([
    [-4.15, -1.70],
    [-1.70, -1.60],
    [1.80, 1.40],
    [4.10, 1.80],
])


# 1. A^TAを求める
B = A.T @ A
show("B = A^T A", B)

# 2. 固有方程式 det(A^T A - λI)を解いて固有値λを求める。
eigenvalues, eigenvectors = np.linalg.eigh(B)
lambda1 = eigenvalues[1]
lambda2 = eigenvalues[0]

show("lambda1 =", lambda1)
show("lambda2 =", lambda2)



# 3．固有値を使ってVを求める。
v1 = eigenvectors[:, 1]  # 大きい固有値に対応
v2 = eigenvectors[:, 0]  # 小さい固有値に対応

sigma1 = np.sqrt(lambda1)
sigma2 = np.sqrt(lambda2)

V = np.column_stack((v1, v2))
V_transpose = V.T
show("V =", V)

# 4.Uを求める。u = A v / sigma。
# sigma2が0でないかのチェックは省略
Av1 = A @ v1
u1 = Av1 / sigma1
Av2 = A @ v2
u2 = Av2 / sigma2

show("A v1", Av1)
show("A v2", Av2)


U = np.column_stack((u1, u2))
singular_values = np.array([sigma1, sigma2])
Sigma = np.diag(singular_values)
show("U", U)
show("Sigma", Sigma)
show("V^T", V_transpose)


# 可視化

fig, ax = plt.subplots(figsize=(8, 7))

# Aの1行を1点として描く。
x = A[:, 0]
y = A[:, 1]
ax.scatter(x, y, color="#c58b20", s=85, label="Rows of A", zorder=4)

# 各点に行番号を付ける。
for i, point in enumerate(A):
    number = i + 1
    label = rf"$\mathbf{{a}}_{number}$"
    ax.annotate(label, xy=point, xytext=(8, 9), textcoords="offset points", fontsize=13)

# v1に沿う、原点を通る直線を描く。
maximum_coordinate = np.max(np.abs(A))
extent = maximum_coordinate + 1.0
line_parameters = np.linspace(-2.0 * extent, 2.0 * extent, 100)
line_points = np.outer(line_parameters, v1)
ax.plot(line_points[:, 0], line_points[:, 1], "--", color="#2467ad", alpha=0.6,
        label="Line along v1", zorder=2)

# ベクトルは入力された長さのまま描く。単位ベクトルなら矢印の長さも1。
ax.annotate("", xy=v1, xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#2467ad", "lw": 3,
                        "mutation_scale": 20}, zorder=5)
ax.annotate("", xy=v2, xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#c9443d", "lw": 3,
                        "mutation_scale": 20}, zorder=5)

ax.annotate(r"$\mathbf{v}_1$", xy=v1, xytext=(10, -15),
            textcoords="offset points", color="#2467ad", fontsize=16)
ax.annotate(r"$\mathbf{v}_2=\mathbf{f}$", xy=v2, xytext=(12, 12),
            textcoords="offset points", color="#c9443d", fontsize=16)

# 目盛りと縦横の縮尺を揃える。
ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$a_x$", fontsize=13)
ax.set_ylabel(r"$a_y$", fontsize=13)
ax.set_title("Rows of A and the right singular vectors")
ax.grid(alpha=0.2)
ax.legend(loc="lower right")
fig.tight_layout()

# プログラムと同じディレクトリに画像を保存する。
script_path = Path(__file__)
output_path = script_path.with_name("svd_vectors.png")
fig.savefig(output_path, dpi=180)
print(f"保存しました: {output_path}")

plt.show()


