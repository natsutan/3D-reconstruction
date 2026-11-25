"""RANSACの出力から、2枚のカメラ姿勢と三次元点を復元する。

配置: COLMAP/sfm1.py
実行: python COLMAP/sfm1.py
可視化: python COLMAP/sfm1.py --visualize
画像ペアの指定例:
    python COLMAP/sfm1.py --pair frame_000001.png frame_000006.png --visualize

入力（このファイルのあるCOLMAPディレクトリを基準とする）:
    camera.json
    output/matching_first/keypoints.csv
    output/ransac/inlier_matches.csv
    output/ransac/fundamental_matrices.csv
出力:
    output/sfm/reconstruction.npz: 次の画像追加で再利用する復元結果
    output/sfm/points3D.csv: 採用した三次元点と元の特徴点番号
    output/sfm/cameras.json: 内部パラメータ、姿勢、座標系、統計
    output/sfm/initialization_candidates.csv: 初期ペアの評価結果
    work/sfm/reconstruction.png: --visualizeを指定した場合の点群・カメラ配置

依存: NumPy（可視化時のみMatplotlibも必要）。元画像の読み込みは不要。
処理: F → E → 姿勢の4候補 → 三角測量 → 前方判定・誤差・視点差の確認。
1枚目のカメラ座標系を世界座標系とし、カメラ間距離を1に固定する。
世界→カメラの変換は X_camera = R @ X_world + t。
カメラ中心は C = -R.T @ t。座標軸は右向きX、下向きY、前向きZ。
この段階では3枚目の追加、バンドル調整、COLMAP形式の保存は行わない。
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CAMERA_PATH = BASE_DIR / "camera.json"
KEYPOINTS_PATH = BASE_DIR / "output" / "matching_first" / "keypoints.csv"
MATCHES_PATH = BASE_DIR / "output" / "ransac" / "inlier_matches.csv"
MATRICES_PATH = BASE_DIR / "output" / "ransac" / "fundamental_matrices.csv"
OUTPUT_DIR = BASE_DIR / "output" / "sfm"
VISUALIZATION_DIR = BASE_DIR / "work" / "sfm"

# 初期化を確認するための簡略化した設定。COLMAPの既定値とは異なる。
MIN_INITIAL_POINTS = 8
MAX_REPROJECTION_ERROR_PX = 2.0  # 左右それぞれの再投影誤差の上限
MIN_TRIANGULATION_ANGLE_DEG = 1.0  # 視線がほぼ平行な不安定な点を除く
DEPTH_EPSILON = 1e-9


@dataclass
class ImagePair:
    image1: str
    image2: str
    frame_gap: int
    fundamental: np.ndarray
    keypoint_indices: np.ndarray
    points1: np.ndarray
    points2: np.ndarray


@dataclass
class Reconstruction:
    pair: ImagePair
    K: np.ndarray
    essential: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    points3D: np.ndarray
    positive_depth: np.ndarray
    errors: np.ndarray
    angles: np.ndarray
    valid: np.ndarray


def read_csv(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    return rows


def load_camera():
    with CAMERA_PATH.open("r", encoding="utf-8-sig") as stream:
        camera = json.load(stream)
    params = np.asarray(camera["params"], dtype=np.float64)
    if camera["model"] == "SIMPLE_PINHOLE" and params.shape == (3,):
        f, cx, cy = params
        fx, fy = f, f
    elif camera["model"] == "PINHOLE" and params.shape == (4,):
        fx, fy, cx, cy = params
    else:
        raise ValueError("camera.jsonはSIMPLE_PINHOLEまたはPINHOLEで指定してください。")
    if not np.all(np.isfinite(params)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError("内部パラメータが不正です。")
    if camera["width"] <= 0 or camera["height"] <= 0:
        raise ValueError("画像サイズは正の値で指定してください。")

    # camera.jsonはCOLMAP規約（左上ピクセル中心が0.5, 0.5）。
    # SIFTのCSVはOpenCV規約（同じ中心が0, 0）なので、主点を0.5ずらす。
    K = np.array([[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]])
    return camera, K


def load_inputs(pair_names=None):
    """入力層: 内部パラメータ、F、インライアの座標・番号を読み込む。"""
    camera, K = load_camera()
    keypoints = {}
    for row in read_csv(KEYPOINTS_PATH):
        key = (row["image_name"], int(row["keypoint_index"]))
        if key in keypoints:
            raise ValueError(f"特徴点番号が重複しています: {key}")
        point = np.array([float(row["x"]), float(row["y"])])
        if not np.all(np.isfinite(point)):
            raise ValueError(f"特徴点座標が不正です: {key}")
        if not (0 <= point[0] < camera["width"] and 0 <= point[1] < camera["height"]):
            raise ValueError("特徴点がcamera.jsonの画像サイズから外れています。")
        keypoints[key] = point

    matches = {}
    for row in read_csv(MATCHES_PATH):
        pair = (row["image1"], row["image2"])
        if pair not in matches:
            matches[pair] = []
        matches[pair].append((int(row["keypoint_index1"]), int(row["keypoint_index2"])))

    pairs = []
    for row in read_csv(MATRICES_PATH):
        pair = (row["image1"], row["image2"])
        if pair_names is not None and pair != tuple(pair_names):
            continue
        if row["status"] != "ok":
            continue
        indices = matches.get(pair, [])
        if len(indices) < MIN_INITIAL_POINTS:
            continue
        values = []
        for r in range(1, 4):
            for c in range(1, 4):
                values.append(float(row[f"F{r}{c}"]))
        fundamental = np.array(values).reshape(3, 3)
        if not np.all(np.isfinite(fundamental)) or np.linalg.norm(fundamental) == 0.0:
            raise ValueError(f"基礎行列が不正です: {pair}")
        points1 = []
        points2 = []
        for index1, index2 in indices:
            if (pair[0], index1) not in keypoints or (pair[1], index2) not in keypoints:
                raise ValueError(f"対応する特徴点が見つかりません: {pair}")
            points1.append(keypoints[(pair[0], index1)])
            points2.append(keypoints[(pair[1], index2)])
        pairs.append(ImagePair(
            pair[0], pair[1], int(row["frame_gap"]), fundamental,
            np.asarray(indices, dtype=np.int64),
            np.asarray(points1), np.asarray(points2),
        ))
    if not pairs:
        raise ValueError("初期化候補がありません。RANSACの出力と--pairの画像名を確認してください。")
    return camera, K, pairs


def normalized_image_points(points, K):
    """ピクセル座標をKの逆行列で正規化画像座標へ変換する。"""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = np.linalg.solve(K, homogeneous.T).T
    return normalized[:, :2] / normalized[:, 2:3]


def essential_and_pose_candidates(fundamental, K):
    """E=K.T@F@Kを本質行列の制約へ補正し、(R, ±t)の4候補を返す。"""
    raw_essential = K.T @ fundamental @ K
    u, singular_values, vt = np.linalg.svd(raw_essential)
    if singular_values[1] <= 1e-12 * singular_values[0]:
        raise ValueError("本質行列が退化しています。")
    # UとV.Tの向きを調整し、以下のRが正しい回転（det(R)=+1）になるようにする。
    if np.linalg.det(u) < 0.0:
        u[:, -1] *= -1.0
    if np.linalg.det(vt) < 0.0:
        vt[-1, :] *= -1.0
    scale = (singular_values[0] + singular_values[1]) / 2.0
    essential = u @ np.diag([scale, scale, 0.0]) @ vt
    essential /= np.linalg.norm(essential)
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation1 = u @ w @ vt
    rotation2 = u @ w.T @ vt
    translation = u[:, 2]
    # 並進の長さは画像だけでは決まらないので、カメラ間距離を1にする。
    translation /= np.linalg.norm(translation)
    candidates = []
    for rotation in (rotation1, rotation2):
        for sign in (1.0, -1.0):
            candidates.append((rotation, sign * translation))
    return essential, candidates


def triangulate_points(points1, points2, rotation, translation):
    """DLTで各対応から三次元点を求める。ここではKを掛けない射影行列を使う。"""
    projection1 = np.column_stack((np.eye(3), np.zeros(3)))
    projection2 = np.column_stack((rotation, translation))
    points3D = np.full((len(points1), 3), np.nan)
    for index, (point1, point2) in enumerate(zip(points1, points2)):
        design = np.array([
            point1[0] * projection1[2] - projection1[0],
            point1[1] * projection1[2] - projection1[1],
            point2[0] * projection2[2] - projection2[0],
            point2[1] * projection2[2] - projection2[1],
        ])
        _, _, vt = np.linalg.svd(design)
        point = vt[-1]
        if abs(point[3]) > 1e-12:
            points3D[index] = point[:3] / point[3]
    return points3D


def project_points(points3D, K, rotation, translation):
    """世界座標の三次元点をピクセル座標へ再投影する。"""
    camera_points = points3D @ rotation.T + translation
    homogeneous = camera_points @ K.T
    projected = np.full((len(points3D), 2), np.nan)
    valid = np.all(np.isfinite(homogeneous), axis=1) & (np.abs(homogeneous[:, 2]) > DEPTH_EPSILON)
    projected[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return projected


def triangulation_angles(points3D, camera_center2):
    """2本の視線のなす角を求める。0度・180度付近はともに不安定とする。"""
    ray1 = points3D
    ray2 = points3D - camera_center2
    denominator = np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
    angles = np.zeros(len(points3D))
    valid = np.all(np.isfinite(points3D), axis=1) & (denominator > DEPTH_EPSILON)
    cosine = np.sum(ray1[valid] * ray2[valid], axis=1) / denominator[valid]
    angles[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.minimum(angles, 180.0 - angles)


def reconstruct_pair(pair, K):
    """推定層: 姿勢の4候補を前方判定で選び、安定した三次元点を残す。"""
    points1 = normalized_image_points(pair.points1, K)
    points2 = normalized_image_points(pair.points2, K)
    essential, candidates = essential_and_pose_candidates(pair.fundamental, K)
    best = None
    best_count = -1
    for rotation, translation in candidates:
        points3D = triangulate_points(points1, points2, rotation, translation)
        camera2_points = points3D @ rotation.T + translation
        positive = (
            np.all(np.isfinite(points3D), axis=1)
            & (points3D[:, 2] > DEPTH_EPSILON)
            & (camera2_points[:, 2] > DEPTH_EPSILON)
        )
        count = int(np.count_nonzero(positive))
        if count > best_count:
            best_count = count
            best = (rotation, translation, points3D, positive)
    rotation, translation, points3D, positive = best
    projected1 = project_points(points3D, K, np.eye(3), np.zeros(3))
    projected2 = project_points(points3D, K, rotation, translation)
    errors = np.column_stack((
        np.linalg.norm(projected1 - pair.points1, axis=1),
        np.linalg.norm(projected2 - pair.points2, axis=1),
    ))
    angles = triangulation_angles(points3D, -rotation.T @ translation)
    valid = (
        positive & np.all(np.isfinite(errors), axis=1)
        & np.all(errors <= MAX_REPROJECTION_ERROR_PX, axis=1)
        & (angles >= MIN_TRIANGULATION_ANGLE_DEG)
    )
    # 3枚目を追加するとき、各特徴点は一つの三次元点だけに対応する必要がある。
    # 同じ特徴点を含む対応が複数あれば、再投影誤差の小さい対応を採用する。
    used1 = set()
    used2 = set()
    for index in np.argsort(np.max(errors, axis=1), kind="stable"):
        if not valid[index]:
            continue
        index1, index2 = pair.keypoint_indices[index]
        if index1 in used1 or index2 in used2:
            valid[index] = False
        else:
            used1.add(index1)
            used2.add(index2)
    return Reconstruction(pair, K, essential, rotation, translation,
                          points3D, positive, errors, angles, valid)


def reconstruct_two_views(pairs, K):
    """候補ごとに評価し、採用できる三次元点が最多のペアを初期ペアとする。"""
    selected = None
    best_score = (-1, -1.0)
    summaries = []
    for pair in pairs:
        summary = {"image1": pair.image1, "image2": pair.image2,
                   "num_inliers": len(pair.points1), "num_in_front": 0,
                   "num_points3D": 0, "median_angle_deg": "",
                   "median_reprojection_error_px": "", "status": "failed"}
        try:
            result = reconstruct_pair(pair, K)
        except (ValueError, np.linalg.LinAlgError):
            summaries.append(summary)
            continue
        count = int(np.count_nonzero(result.valid))
        summary["num_in_front"] = int(np.count_nonzero(result.positive_depth))
        summary["num_points3D"] = count
        if count:
            summary["median_angle_deg"] = float(np.median(result.angles[result.valid]))
            summary["median_reprojection_error_px"] = float(np.median(np.max(result.errors[result.valid], axis=1)))
        if count >= MIN_INITIAL_POINTS:
            summary["status"] = "ok"
            score = (count, summary["median_angle_deg"])
            if score > best_score:
                selected = result
                best_score = score
        else:
            summary["status"] = "not_enough_stable_points"
        summaries.append(summary)
    if selected is None:
        raise ValueError("初期化できる画像ペアがありません。視点差やcamera.jsonの値を確認してください。")
    return selected, summaries


def write_results(camera, result, summaries):
    """出力層: 姿勢・点群・対応を保存する。三次元復元はここでは行わない。"""
    pair = result.pair
    valid = result.valid
    rotations = np.stack((np.eye(3), result.rotation))
    translations = np.stack((np.zeros(3), result.translation))
    centers = np.stack((np.zeros(3), -result.rotation.T @ result.translation))
    point_ids = np.arange(np.count_nonzero(valid), dtype=np.int64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_DIR) as temporary:
        staging = Path(temporary)
        # オブジェクト配列を含めず、次の段階でallow_pickle=Falseのまま読み込める。
        np.savez_compressed(
            staging / "reconstruction.npz", schema_version=np.array(1),
            image_names=np.array([pair.image1, pair.image2]), K=result.K,
            image_size=np.array([camera["width"], camera["height"]]),
            rotations=rotations, translations=translations, camera_centers=centers,
            point_ids=point_ids, points3D=result.points3D[valid],
            keypoint_indices=pair.keypoint_indices[valid],
            points2D=np.stack((pair.points1[valid], pair.points2[valid]), axis=1),
            reprojection_errors=result.errors[valid], triangulation_angles=result.angles[valid],
            F=pair.fundamental, E=result.essential, coordinate_convention=np.array("opencv"),
        )
        with (staging / "points3D.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "x", "y", "z", "image1", "keypoint_index1",
                             "image2", "keypoint_index2", "error1_px", "error2_px", "triangulation_angle_deg"])
            for point_id, index in enumerate(np.flatnonzero(valid)):
                x, y, z = result.points3D[index]
                index1, index2 = pair.keypoint_indices[index]
                writer.writerow([point_id, x, y, z, pair.image1, index1, pair.image2, index2,
                                 result.errors[index, 0], result.errors[index, 1], result.angles[index]])
        with (staging / "initialization_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        metadata = {
            "image_names": [pair.image1, pair.image2], "input_camera": camera,
            "K": result.K.tolist(), "coordinate_convention": "opencv",
            "axes": "x right, y down, z forward; first camera is the world frame",
            "pose_convention": "X_camera = R @ X_world + t; camera_center = -R.T @ t",
            "scale": "arbitrary; camera baseline = 1",
            "rotations": rotations.tolist(), "translations": translations.tolist(),
            "camera_centers": centers.tolist(), "num_inlier_matches": len(pair.points1),
            "num_points3D": int(np.count_nonzero(valid)),
            "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
            "min_triangulation_angle_deg": MIN_TRIANGULATION_ANGLE_DEG,
            "median_reprojection_error_px": float(np.median(np.max(result.errors[valid], axis=1))),
            "median_triangulation_angle_deg": float(np.median(result.angles[valid])),
        }
        with (staging / "cameras.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for path in staging.iterdir():
            path.replace(OUTPUT_DIR / path.name)
    for summary in summaries:
        median_angle_text = "算出不可"
        if summary["median_angle_deg"] != "":
            median_angle_text = f"{summary['median_angle_deg']:.4f}度"
        print(f"{summary['image1']} -> {summary['image2']}: "
              f"インライア{summary['num_inliers']}組、前方{summary['num_in_front']}点、"
              f"採用{summary['num_points3D']}点、三角測量角度の中央値{median_angle_text}、"
              f"{summary['status']}")
    print(f"選択した初期ペア：{pair.image1} / {pair.image2}")
    print(f"三次元点数：{len(point_ids)}")
    print(f"再投影誤差の中央値：{metadata['median_reprojection_error_px']:.4f} ピクセル")
    print(f"三角測量角度の中央値：{metadata['median_triangulation_angle_deg']:.4f} 度")
    print("世界→2枚目カメラの回転行列R：\n", result.rotation)
    print("世界→2枚目カメラの並進t：", result.translation)
    print("2枚目カメラの世界座標での位置：", centers[1])
    print(f"復元結果の出力先：{OUTPUT_DIR}")
    print("座標の単位は任意です。カメラ間距離を1として保存しています。")


def write_visualization(camera, result):
    """出力層: 三次元点群と2台のカメラの位置・向きを図に保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = set()
    for font in font_manager.fontManager.ttflist:
        available_fonts.add(font.name)
    for name in ["Meiryo", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Hiragino Sans"]:
        if name in available_fonts:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    points = result.points3D[result.valid]
    figure = plt.figure(figsize=(13, 6), layout="constrained")
    scene_axis = figure.add_subplot(1, 2, 1, projection="3d")
    camera_axis = figure.add_subplot(1, 2, 2, projection="3d")
    scene_axis.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=points[:, 2], cmap="viridis", s=9, alpha=0.8)
    scene_axis.set_title(f"復元した点群：{len(points)}点")
    camera_axis.set_title("カメラ配置の拡大（カメラ間距離＝1）")
    image_corners = np.array([[0.0, 0.0], [camera["width"] - 1.0, 0.0],
                              [camera["width"] - 1.0, camera["height"] - 1.0],
                              [0.0, camera["height"] - 1.0]])
    # 画像四隅の視線を、表示用の短い距離まで延ばしてカメラの向きを描く。
    normalized_corners = normalized_image_points(image_corners, result.K)
    local_corners = np.column_stack((normalized_corners, np.ones(4))) * 0.4
    camera_points = []
    for index, (rotation, translation, color) in enumerate([
        (np.eye(3), np.zeros(3), "#d34949"),
        (result.rotation, result.translation, "#2867b2"),
    ], start=1):
        center = -rotation.T @ translation
        corners = (local_corners - translation) @ rotation
        camera_points.extend(corners)
        camera_points.append(center)
        for axis in (scene_axis, camera_axis):
            axis.scatter(*center, color=color, s=55)
            axis.text(*center, f" カメラ{index}", color=color)
            for corner_index in range(4):
                corner = corners[corner_index]
                next_corner = corners[(corner_index + 1) % 4]
                axis.plot(*np.stack((center, corner)).T, color=color, linewidth=1.2)
                axis.plot(*np.stack((corner, next_corner)).T, color=color, linewidth=1.2)
        camera_axis.quiver(*center, *(rotation.T @ np.array([0.0, 0.0, 0.6])),
                           color=color, arrow_length_ratio=0.15)
    for axis, coordinates in [(scene_axis, np.vstack((points, camera_points))),
                              (camera_axis, np.asarray(camera_points))]:
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        center = (lower + upper) / 2.0
        radius = max(float(np.max(upper - lower)) * 0.55, 0.5)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set(xlabel="X（右）", ylabel="Y（下）", zlabel="Z（前）")
        axis.view_init(elev=20, azim=-65)
    figure.suptitle(f"{result.pair.image1} / {result.pair.image2}　（尺度は任意）", fontsize=13)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISUALIZATION_DIR / "reconstruction.png", dpi=150)
    plt.close(figure)
    print(f"点群・カメラ配置の図：{VISUALIZATION_DIR / 'reconstruction.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("IMAGE1", "IMAGE2"),
                        help="RANSACのCSVと同じ順序で画像名を指定（省略時は自動選択）")
    parser.add_argument("--visualize", action="store_true", help="点群とカメラ配置の図を保存")
    args = parser.parse_args()

    # 1. 入力
    camera, K, pairs = load_inputs(args.pair)
    # 2. 推定と三角測量
    result, summaries = reconstruct_two_views(pairs, K)
    # 3. 出力
    write_results(camera, result, summaries)
    if args.visualize:
        write_visualization(camera, result)


if __name__ == "__main__":
    main()
