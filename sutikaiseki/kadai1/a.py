import numpy as np

# 乱数シードの固定（再現性確保）
np.random.seed(0)

# 真の点を生成
N = 100
theta_i = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
a_true = 300
b_true = 200
x_true = a_true * np.cos(theta_i)
y_true = b_true * np.sin(theta_i)

# 真の楕円パラメータ（6次元: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0）
true_params = np.array([
    1 / a_true**2,  # A
    0,              # B
    1 / b_true**2,  # C
    0,              # D
    0,              # E
   -1               # F
])

# 最小二乗法による6次元楕円フィッティング
def fit_ellipse_ls(x, y):
    D = np.vstack([
        x**2,
        x * y,
        y**2,
        x,
        y,
        np.ones_like(x)
    ]).T  # shape (N, 6)

    # SVDで解く: 最小特異値に対応する右特異ベクトル
    _, _, Vt = np.linalg.svd(D)
    u = Vt[-1, :]
    return u / np.linalg.norm(u)

def normalize_by_F(params):
    if np.abs(params[-1]) < 1e-10:
        return params  # Fがゼロに近い場合はスキップ
    return params / params[-1]

def rms_error(params, true_params):
    params = normalize_by_F(params)
    true_params = normalize_by_F(true_params)
    return np.sqrt(np.mean((params - true_params) ** 2))
# 実験パラメータ
sigma = 2.0
num_trials = 1000
errors = []

# 試行ループ
for _ in range(num_trials):
    x_noisy = x_true + np.random.normal(0, sigma, size=x_true.shape)
    y_noisy = y_true + np.random.normal(0, sigma, size=y_true.shape)
    est_params = fit_ellipse_ls(x_noisy, y_noisy)
    err = rms_error(est_params, true_params)
    print("est_params:", est_params)
    errors.append(err)

# 結果出力
mean_rms_error = np.mean(errors)
print(f"σ={sigma}, RMS誤差平均={mean_rms_error:.6e}")
