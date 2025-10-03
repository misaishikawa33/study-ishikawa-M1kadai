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

# --- 3. 対応点マッチング ---
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) < 3:
    raise ValueError("マッチング数が不足しています")

# 対応点抽出
pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# --- 4. ガウス・ニュートン法で s, theta 推定 ---

# モデル：回転＋スケーリング
def residual(params, x, y):
    s, theta = params
    R = s * np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    x_trans = (R @ x.T).T  # N×2
    return (x_trans - y).ravel()  # 平坦化して残差を返す

# 初期値（スケール=1, 回転=0）
x0 = np.array([1.0, 0.0])

# 最適化
res = least_squares(residual, x0, args=(pts1, pts2))

# 結果表示
s_est, theta_est = res.x
print(f"推定されたスケール s: {s_est:.4f}")
print(f"推定された回転角 θ（ラジアン）: {theta_est:.4f}")
print(f"推定された回転角 θ（度）: {np.degrees(theta_est):.2f}°")

# --- 5. 対応点表示（確認用） ---
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
