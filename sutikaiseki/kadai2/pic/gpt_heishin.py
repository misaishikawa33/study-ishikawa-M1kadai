import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# --- 1. 画像読み込み ---
img1_path = "/home/misa/kadai/M1_kadai/sutikaiseki/kadai2/pic/kuma_in.png"
img2_path = "/home/misa/kadai/M1_kadai/sutikaiseki/kadai2/pic/transformed_image.jpg"
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("画像が読み込めません。パスを確認してください。")

# --- 2. 特徴点検出（SIFT） ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- 3. 特徴点マッチング ---
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Loweの比率テスト
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) < 3:
    raise ValueError("マッチング数が不足しています")

# 対応点抽出
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])  # 元画像
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])  # 変換後画像

# --- 4. ガウス・ニュートン法で s, θ, tx, ty を推定 ---
# モデル：相似変換 (s, θ, tx, ty)
def residual(params, x, y):
    s, theta, tx, ty = params
    R = s * np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    x_trans = (R @ x.T).T + np.array([tx, ty])  # 平行移動を加える
    return (x_trans - y).ravel()

# 初期値
x0 = np.array([1.0, 0.0, 0.0, 0.0])  # s=1, θ=0, tx=0, ty=0

# 残差履歴を保存するコールバック関数
loss_history = []
def callback(xk, *args, **kwargs):
    loss = np.linalg.norm(residual(xk, pts1, pts2))
    loss_history.append(loss)

# 最適化
res = least_squares(
    residual,
    x0,
    args=(pts1, pts2),
    verbose=1,
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
    callback=callback
)

# 結果表示
s_est, theta_est, tx_est, ty_est = res.x
print(f"\n--- 推定結果 ---")
print(f"スケール s: {s_est:.4f}")
print(f"回転角 θ（ラジアン）: {theta_est:.4f}")
print(f"回転角 θ（度）: {np.degrees(theta_est):.2f}°")
print(f"平行移動 t_x: {tx_est:.2f}, t_y: {ty_est:.2f}")

# --- 5. 残差の収束グラフ ---
plt.figure(figsize=(6, 4))
plt.plot(loss_history, marker='o')
plt.title("ガウス・ニュートン法の収束（残差ノルム）")
plt.xlabel("イテレーション")
plt.ylabel("残差ノルム ||r||")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. 対応点の描画（オプション） ---
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("Feature Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
