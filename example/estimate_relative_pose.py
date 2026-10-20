"""対応点CSVから基礎行列F、相対回転R、並進方向tを求める。

Python 3.13 / Windows 11・Ubuntu。依存: NumPy
準備: python -m pip install numpy
実行: python estimate_relative_pose.py
CSV2個をinputフォルダーに置く。
正規化ON: python estimate_relative_pose.py --normalize
正規化OFF（既定）: python estimate_relative_pose.py --no-normalize
別の場所のCSVも指定可能:
python estimate_relative_pose.py path/to/view_01_points_ideal.csv path/to/view_02_points_ideal.csv

--normalizeでCOLMAP方式の正規化8点法を使用する。
画像ごとに重心を原点へ移し、重心からのRMS距離をsqrt(2)にする。
正規化座標でランク2のFを推定し、最後に元のピクセル座標へ戻す。
SVDにはnumpy.linalg.svdを使用する。
座標の規約: x' = R @ x + t（画像1のカメラ座標→画像2のカメラ座標）。
カメラ間の距離は復元できないため、並進tの長さを1とする。
"""

import argparse
import csv
from pathlib import Path

import numpy as np


# 画像を生成したプログラムで指定した内部パラメータ。CSVからは推定しない。
FOCAL_LENGTH_PX = 450.0
PRINCIPAL_POINT_X_PX = 320.0
PRINCIPAL_POINT_Y_PX = 340.0
K = np.array([
    [FOCAL_LENGTH_PX, 0.0, PRINCIPAL_POINT_X_PX],
    [0.0, FOCAL_LENGTH_PX, PRINCIPAL_POINT_Y_PX],
    [0.0, 0.0, 1.0],
])


def read_points(path):
    """id,x,yのCSVを読み、番号をキーとする辞書にする。丸めない。"""
    points = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"id", "x", "y"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: CSVにはid,x,yの列が必要です。")
        for row in reader:
            point_id = int(row["id"])
            if point_id in points:
                raise ValueError(f"{path}: id={point_id}が重複しています。")
            xy = np.array([float(row["x"]), float(row["y"])])
            if not np.all(np.isfinite(xy)):
                raise ValueError(f"{path}: id={point_id}の座標が有限値ではありません。")
            points[point_id] = xy
    return points


def match_points(path1, path2):
    """同じid同士を対応付け、番号順に並べる。"""
    points1, points2 = read_points(path1), read_points(path2)
    if points1.keys() != points2.keys():
        raise ValueError("2つのCSVのidが一致しません。")
    ids = sorted(points1)
    if len(ids) < 8:
        raise ValueError("少なくとも8組の対応点が必要です。")
    return ids, np.array([points1[i] for i in ids]), np.array([points2[i] for i in ids])


def center_and_normalize_points(points):
    """COLMAP方式: 重心0、重心からの距離の二乗平均平方根をsqrt(2)にする。

    points: (N, 2)のピクセル座標。
    戻り値: 正規化座標、同次座標の変換行列T（x_normalized = T @ x）。
    Kの逆行列によるカメラ座標への変換とは別の前処理。
    """
    points = np.asarray(points, dtype=float)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    # 距離の算術平均ではなく、RMS = sqrt(mean(||point - centroid||^2))。
    rms_distance = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if not np.isfinite(rms_distance) or rms_distance <= np.finfo(float).eps:
        raise ValueError("特徴点の広がりが小さすぎるため、座標を正規化できません。")
    scale = np.sqrt(2.0) / rms_distance
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    normalized_points = centered * scale
    return normalized_points, T


def estimate_fundamental(points1, points2, normalize=False):
    """8点法でFを推定する。normalize=TrueならCOLMAP方式で座標を正規化。

    戻り値A,s_A,f,F0,s_Fは、ONなら正規化座標に対する計算途中の値。
    最後の戻り値Fは、ON/OFFのどちらでも元のピクセル座標に対する行列。
    """
    if normalize:
        work_points1, T1 = center_and_normalize_points(points1)
        work_points2, T2 = center_and_normalize_points(points2)
    else:
        work_points1, work_points2 = points1, points2
    x, y = work_points1.T
    xp, yp = work_points2.T
    A = np.column_stack([
        xp * x, xp * y, xp,
        yp * x, yp * y, yp,
        x, y, np.ones_like(x),
    ])
    # A = U_A @ diag(s_A) @ V_A^T。戻り値Vt_AはV_Aの転置。
    # full_matrices=Trueなら、対応点がちょうど8組でもVt_Aは9×9になる。
    _, s_A, Vt_A = np.linalg.svd(A, full_matrices=True)
    tolerance = max(A.shape) * np.finfo(float).eps * s_A[0]
    if np.count_nonzero(s_A > tolerance) < 8:
        raise ValueError("対応点の配置からFを一意に推定できません（Aのランクが8未満）。")
    f = Vt_A[-1]  # V_Aの最後の列、すなわち最小特異値に対応する方向
    F0 = f.reshape(3, 3)

    # F0の最小特異値を0にして、ランク2の基礎行列Fを作る。
    U_F, s_F, Vt_F = np.linalg.svd(F0)
    s_rank2 = s_F.copy()
    s_rank2[-1] = 0.0
    F = U_F @ np.diag(s_rank2) @ Vt_F
    if normalize:
        # 正規化座標でランク2にした後、元のピクセル座標へ戻す。
        # x2_n.T @ F_n @ x1_n = x2.T @ (T2.T @ F_n @ T1) @ x1。
        F = T2.T @ F @ T1
    return A, s_A, f, F0, s_F, F


def essential_and_pose_candidates(F, K1, K2):
    """E0=K2^T F K1をエッセンシャル行列に修正し、R,tの4候補を得る。"""
    E0 = K2.T @ F @ K1
    U_E, s_E, Vt_E = np.linalg.svd(E0)

    # エッセンシャルの特異値は(s,s,0)。先頭2個を平均してそろえる。
    # 最後の特異値を0にするので、最後の特異ベクトルの符号変更はEに影響しない。
    # U_EとVt_Eの行列式を+1にそろえ、候補Rが正しい回転行列になるようにする。
    if np.linalg.det(U_E) < 0:
        U_E[:, -1] *= -1
    if np.linalg.det(Vt_E) < 0:
        Vt_E[-1, :] *= -1
    s = (s_E[0] + s_E[1]) / 2.0
    E = U_E @ np.diag([s, s, 0.0]) @ Vt_E

    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U_E @ W @ Vt_E
    R2 = U_E @ W.T @ Vt_E
    t = U_E[:, 2]  # 単位ベクトル。実際の移動距離は分からない。
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    return E0, s_E, E, candidates


def triangulate(points1, points2, P1, P2):
    """2画像から線形三角測量し、画像1のカメラ座標で3次元点を返す。"""
    points3d = []
    for (x, y), (xp, yp) in zip(points1, points2):
        B = np.array([
            x * P1[2] - P1[0],
            y * P1[2] - P1[1],
            xp * P2[2] - P2[0],
            yp * P2[2] - P2[1],
        ])
        _, _, Vt_B = np.linalg.svd(B)
        X_h = Vt_B[-1]
        if abs(X_h[3]) < 1e-12:
            points3d.append(np.full(3, np.nan))
        else:
            points3d.append(X_h[:3] / X_h[3])
    return np.array(points3d)


def choose_pose(points1, points2, K1, K2, candidates):
    """両カメラの前方に復元される点が最も多い候補を選ぶ。"""
    P1 = K1 @ np.column_stack([np.eye(3), np.zeros(3)])
    counts, reconstructions = [], []
    for R, t in candidates:
        P2 = K2 @ np.column_stack([R, t])
        X = triangulate(points1, points2, P1, P2)
        X2 = (R @ X.T).T + t
        in_front = np.all(np.isfinite(X), axis=1) & (X[:, 2] > 0) & (X2[:, 2] > 0)
        counts.append(int(np.count_nonzero(in_front)))
        reconstructions.append(X)
    best = int(np.argmax(counts))
    if counts[best] == 0 or counts.count(counts[best]) > 1:
        raise ValueError(f"カメラの前後判定で候補を決められません: {counts}")
    R, t = candidates[best]
    return R, t, reconstructions[best], counts


def rotation_angle_deg(R):
    """回転行列を軸角表現したときの回転角（0〜180度）。"""
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def reprojection_rmse(points1, points2, X, R, t, K1, K2):
    """1観測点あたりの2次元ユークリッド距離に基づく再投影RMSE [px]。"""
    X2 = (R @ X.T).T + t
    q1, q2 = (K1 @ X.T).T, (K2 @ X2.T).T
    uv1, uv2 = q1[:, :2] / q1[:, 2, None], q2[:, :2] / q2[:, 2, None]
    squared = np.concatenate([
        np.sum((uv1 - points1) ** 2, axis=1),
        np.sum((uv2 - points2) ** 2, axis=1),
    ])
    return float(np.sqrt(np.mean(squared)))


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv1", nargs="?", type=Path, default=folder / "input" / "view_01_points.csv")
    parser.add_argument("csv2", nargs="?", type=Path, default=folder / "input" / "view_02_points.csv")
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=False,
        help="COLMAP方式のRMS座標正規化を使う。--no-normalizeでOFF（既定: OFF）。",
    )
    args = parser.parse_args()

    ids, points1, points2 = match_points(args.csv1, args.csv2)
    A, s_A, f, F0, s_F, F = estimate_fundamental(points1, points2, normalize=args.normalize)
    K1, K2 = K.copy(), K.copy()
    E0, s_E, E, candidates = essential_and_pose_candidates(F, K1, K2)
    R, t, X, counts = choose_pose(points1, points2, K1, K2, candidates)
    angle = rotation_angle_deg(R)
    C2 = -R.T @ t  # 画像1のカメラ座標から見た、画像2のカメラ中心。
    rmse = reprojection_rmse(points1, points2, X, R, t, K1, K2)

    np.set_printoptions(precision=10, suppress=False, linewidth=120)
    print(f"対応点: {len(ids)}組 / id = {ids}")
    if args.normalize:
        print("正規化: ON（COLMAP方式: 重心0、RMS距離sqrt(2)）")
        print("A, f, F0は正規化座標での値、Fは元のピクセル座標に戻した値です。")
    else:
        print("正規化: OFF（従来の8点法）")
    print(f"Aの形状: {A.shape}")
    for label, value in [
        ("K", K), ("Aの特異値", s_A), ("f（9要素、長さ1）", f),
        ("F0（fを3×3に並べた行列）", F0), ("F0の特異値", s_F),
        ("F（ランク2に修正後）", F), ("E0 = K2.T @ F @ K1", E0),
        ("E0の特異値", s_E), ("E（特異値を(s,s,0)に修正後）", E),
        ("R（カメラ1の座標→カメラ2の座標）", R),
        ("t（長さ1、移動距離は不明）", t),
        ("C2 = -R.T @ t（カメラ1の座標でのカメラ2中心）", C2),
    ]:
        print(f"\n{label}:\n{value}")
    print(f"\n4候補それぞれの、両カメラの前方にある点数: {counts}")
    print(f"回転角: {angle:.10f} deg")
    print(f"再投影RMSE: {rmse:.6e} px")
    # 30度は答え合わせにだけ使用し、推定処理には使わない。
    print(f"30 degとの差: {abs(angle - 30.0):.6e} deg")
    print("並進と3次元座標はカメラ間距離を1とした尺度です。")


if __name__ == "__main__":
    main()
