"""sfm1.pyの続き: 二画像を復元し、PnP＋RANSACで三枚目を1枚追加する。

配置: COLMAP/sfm2.py（sfm1.pyのimportや事前実行は不要）
実行: python COLMAP/sfm2.py
可視化: python COLMAP/sfm2.py --visualize
初期ペア指定: python COLMAP/sfm2.py --pair frame_000001.png frame_000006.png
追加画像指定: python COLMAP/sfm2.py --image frame_000002.png --visualize
依存: NumPy, OpenCV / 可視化時のみMatplotlib。CPUで実行。

入力（このファイルのあるCOLMAPディレクトリが基準）:
    camera.json
    output/matching_first/keypoints.csv
    output/ransac/inlier_matches.csv
    output/ransac/fundamental_matrices.csv
出力: output/sfm2/ に以下を保存する。
    reconstruction.npz: 登録した全カメラ・三次元点・観測
    cameras.json: K、R、t、カメラ中心、初期ペア、設定
    points3D.csv / observations.csv: 三次元点と画像ごとの観測
    initialization_candidates.csv: sfm1.pyと同じ初期ペア評価
    registration_candidates.csv: 追加画像の順位・PnP成否・判断値
    pnp_correspondences.csv: 最後に試した画像の2D–3D対応・誤差・採否
可視化: work/sfm2/reconstruction.png

処理の順番:
    1. sfm1.pyと同じ F → E → 4候補 → 三角測量 → 前方・誤差・角度の選別
    2. 登録済み画像の特徴点番号を介し、未登録画像の2Dと既存点の3Dを結ぶ
    3. 2D–3D対応数の多い画像からPnP＋RANSACを試す
    4. 成功した画像のR、tとインライア観測を追加する

毎回CSVから二画像を復元して、三枚目を追加する。保存済みNPZは読み込まない。
二画像復元の計算関数を引き継ぎ、PnPの再投影にもproject_points()を使う。
RANSACの抽選・全対応の評価・反復数更新はPythonで明示的に記述する。
4対応からの姿勢候補（AP3P）と、姿勢だけのLM微調整はOpenCVを利用する。
三次元点の新規追加、再三角測量、バンドル調整は次の段階で扱う。

座標: X_camera = R @ X_world + t / カメラ中心 C = -R.T @ t。
一枚目を世界座標系、初期二枚のカメラ間距離を1とする。PnPのtは正規化しない。
camera.jsonの主点をOpenCV規約へ0.5ずらす処理はload_camera()で一度だけ行う。
"""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CAMERA_PATH = BASE_DIR / "camera.json"
KEYPOINTS_PATH = BASE_DIR / "output" / "matching_first" / "keypoints.csv"
MATCHES_PATH = BASE_DIR / "output" / "ransac" / "inlier_matches.csv"
MATRICES_PATH = BASE_DIR / "output" / "ransac" / "fundamental_matrices.csv"
OUTPUT_DIR = BASE_DIR / "output" / "sfm2"
VISUALIZATION_DIR = BASE_DIR / "work" / "sfm2"

# 初期化を確認するための簡略化した設定。COLMAPの既定値とは異なる。
MIN_INITIAL_POINTS = 8
MAX_REPROJECTION_ERROR_PX = 2.0  # 左右それぞれの再投影誤差の上限
MIN_TRIANGULATION_ANGLE_DEG = 1.0  # 視線がほぼ平行な不安定な点を除く
DEPTH_EPSILON = 1e-9

# ここから追加: 三枚目のPnP＋RANSACの設定。
PNP_SAMPLE_SIZE = 4  # OpenCVのAP3Pに渡す対応数。登録に必要な30点とは別。
PNP_MAX_REPROJECTION_ERROR_PX = MAX_REPROJECTION_ERROR_PX  # 自作版は2 px。
MIN_PNP_INLIERS = 30  # COLMAPのabs_pose_min_num_inliersの既定値を採用。
# 値はCOLMAPを参考にしたが、最終採否にも25%を要求するのは自作版の条件。
# COLMAPのabs_pose_min_inlier_ratioはRANSACの反復数上限の計算に使われる。
MIN_PNP_INLIER_RATIO = 0.25
RANSAC_CONFIDENCE = 0.9999
RANSAC_MIN_ITERATIONS = 30
RANSAC_MAX_ITERATIONS = 10000
RANDOM_SEED = 0
GRID_SIZE = 4  # 分布の参考表示。画像の順位付けには使わない。


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
        if key[1] < 0:
            raise ValueError(f"特徴点番号は0以上にしてください: {key}")
        if key in keypoints:
            raise ValueError(f"特徴点番号が重複しています: {key}")
        point = np.array([float(row["x"]), float(row["y"])])
        if not np.all(np.isfinite(point)):
            raise ValueError(f"特徴点座標が不正です: {key}")
        if not (0 <= point[0] < camera["width"] and 0 <= point[1] < camera["height"]):
            raise ValueError("特徴点がcamera.jsonの画像サイズから外れています。")
        keypoints[key] = point

    matches = {}
    verified_matches = set()
    for row in read_csv(MATCHES_PATH):
        pair = (row["image1"], row["image2"])
        if pair not in matches:
            matches[pair] = []
        index1 = int(row["keypoint_index1"])
        index2 = int(row["keypoint_index2"])
        if pair[0] == pair[1]:
            raise ValueError("同じ画像同士の対応が含まれています。")
        if (pair[0], index1) not in keypoints or (pair[1], index2) not in keypoints:
            raise ValueError(f"対応する特徴点が見つかりません: {pair}")
        matches[pair].append((index1, index2))
        if pair[0] < pair[1]:
            verified_matches.add((pair[0], index1, pair[1], index2))
        else:
            verified_matches.add((pair[1], index2, pair[0], index1))

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
    return camera, K, pairs, keypoints, sorted(verified_matches)


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


@dataclass
class Candidate:
    image_name: str
    point_rows: np.ndarray
    keypoint_indices: np.ndarray
    points2D: np.ndarray
    raw_links: int
    conflicting_features: int
    duplicate_point_features: int
    occupied_cells: int


@dataclass
class PoseEstimate:
    rotation: np.ndarray | None
    translation: np.ndarray | None
    errors: np.ndarray
    depths: np.ndarray
    inliers: np.ndarray
    iterations: int
    draws: int
    hypotheses: int
    refinements: int
    reason: str


def make_initial_model(camera, result):
    """推定層: 二画像の結果を、画像を追加できる配列へまとめる。"""
    valid = result.valid
    return {
        "schema_version": np.array(2),
        "coordinate_convention": np.array("opencv"),
        "stage": np.array("image_registration_only"),
        "image_names": np.array([result.pair.image1, result.pair.image2]),
        "initial_image_names": np.array([result.pair.image1, result.pair.image2]),
        "K": result.K.copy(),
        "image_size": np.array([camera["width"], camera["height"]]),
        "rotations": np.stack((np.eye(3), result.rotation)),
        "translations": np.stack((np.zeros(3), result.translation)),
        "camera_centers": np.stack((np.zeros(3), -result.rotation.T @ result.translation)),
        "point_ids": np.arange(np.count_nonzero(valid), dtype=np.int64),
        "points3D": result.points3D[valid].copy(),
        "keypoint_indices": result.pair.keypoint_indices[valid].copy(),
        "points2D": np.stack((result.pair.points1[valid], result.pair.points2[valid]), axis=1),
        "reprojection_errors": result.errors[valid].copy(),
        "triangulation_angles": result.angles[valid].copy(),
        "F": result.pair.fundamental.copy(), "E": result.essential.copy(),
    }


def build_candidates(model, keypoints, matches):
    """推定層: 登録済み画像の特徴点を介して、新画像の2Dと既存点の3Dを結ぶ。"""
    registered = set(model["image_names"].tolist())
    observations = {}
    for column, image in enumerate(model["image_names"].tolist()):
        for point_row in range(len(model["points3D"])):
            index = int(model["keypoint_indices"][point_row, column])
            if index >= 0:
                key = (image, index)
                if key in observations:
                    raise ValueError("一つの特徴点に複数の三次元点が割り当てられています。")
                observations[key] = point_row

    links = {}
    raw_counts = {}
    for image1, index1, image2, index2 in matches:
        if image1 in registered and image2 not in registered:
            known, image, index = (image1, index1), image2, index2
        elif image2 in registered and image1 not in registered:
            known, image, index = (image2, index2), image1, index1
        else:
            continue
        if known not in observations:
            continue
        raw_counts[image] = raw_counts.get(image, 0) + 1
        point_set = links.setdefault(image, {}).setdefault(index, set())
        point_set.add(observations[known])

    candidates = []
    width, height = model["image_size"]
    image_names = set()
    for image, index in keypoints:
        image_names.add(image)
    for image in sorted(image_names):
        if image in registered:
            continue
        feature_links = links.get(image, {})
        conflicts = 0
        unique_links = []
        point_counts = {}
        for index in sorted(feature_links):
            point_set = feature_links[index]
            if len(point_set) != 1:
                conflicts += 1
                continue
            point_row = next(iter(point_set))
            unique_links.append((index, point_row))
            point_counts[point_row] = point_counts.get(point_row, 0) + 1

        indices = []
        point_rows = []
        points = []
        duplicate_features = 0
        occupied = set()
        for index, point_row in unique_links:
            # 距離などの追加情報がないため、同じ3D点へ複数の特徴点が結び付く場合も除外。
            if point_counts[point_row] != 1:
                duplicate_features += 1
                continue
            point = keypoints[(image, index)]
            indices.append(index)
            point_rows.append(point_row)
            points.append(point)
            cell_x = min(GRID_SIZE - 1, int(point[0] * GRID_SIZE / width))
            cell_y = min(GRID_SIZE - 1, int(point[1] * GRID_SIZE / height))
            occupied.add((cell_x, cell_y))
        candidates.append(Candidate(
            image, np.array(point_rows, dtype=np.int64), np.array(indices, dtype=np.int64),
            np.array(points, dtype=np.float64).reshape(-1, 2), raw_counts.get(image, 0),
            conflicts, duplicate_features, len(occupied)))
    candidates.sort(key=lambda candidate: (-len(candidate.point_rows), candidate.image_name))
    return candidates


def reprojection_errors(points3D, points2D, K, rotation, translation):
    """推定層: sfm1.pyのproject_points()を使い、各対応の誤差を計算する。"""
    projected = project_points(points3D, K, rotation, translation)
    depths = (points3D @ rotation.T + translation)[:, 2]
    valid = np.all(np.isfinite(projected), axis=1) & (depths > DEPTH_EPSILON)
    errors = np.full(len(points3D), np.inf)
    errors[valid] = np.linalg.norm(projected[valid] - points2D[valid], axis=1)
    # カメラの後方にある点は、画像上の誤差が小さくてもインライアにしない。
    return errors, depths


def consensus_score(errors):
    mask = errors <= PNP_MAX_REPROJECTION_ERROR_PX
    count = int(np.count_nonzero(mask))
    median = float(np.median(errors[mask])) if count else np.inf
    return mask, (count, -median)


def required_ransac_iterations(inlier_ratio):
    """4対応すべてがインライアである確率w^4から反復上限を更新する。"""
    if inlier_ratio >= 1:
        return 1
    if inlier_ratio <= 0:
        return RANSAC_MAX_ITERATIONS
    probability = inlier_ratio ** PNP_SAMPLE_SIZE
    return math.ceil(math.log1p(-RANSAC_CONFIDENCE) / math.log1p(-probability))


def non_collinear_sample(points3D, points2D):
    """点が一直線に並ぶ最小サンプルを避ける。平面上の点は許容する。"""
    for points in (points3D, points2D):
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if singular[0] <= 0 or singular[1] <= 1e-6 * singular[0]:
            return False
    return True


def estimate_pose_ransac(points3D, points2D, K):
    """推定層: AP3Pの候補をPythonのRANSACで評価し、採用姿勢だけをLMで改善する。"""
    n = len(points3D)
    empty = PoseEstimate(None, None, np.full(n, np.inf), np.full(n, np.nan),
                         np.zeros(n, dtype=bool), 0, 0, 0, 0, "2D-3D対応不足")
    if n < MIN_PNP_INLIERS:
        return empty
    if not non_collinear_sample(points3D, points2D):
        empty.reason = "対応点がほぼ一直線に並び、姿勢を安定して求められません"
        return empty
    rng = np.random.default_rng(RANDOM_SEED)
    best = empty
    best_score = (0, -np.inf)
    limit = RANSAC_MAX_ITERATIONS
    iterations = 0
    draws = 0
    hypotheses = 0
    used_samples = set()

    # RANSAC ① 4組の2D–3D対応を無作為抽出する。
    #          一組は「既存の三次元点」と「追加画像の特徴点」の対応。
    #         ② AP3Pで(R, t)の候補を求める。
    #         ③ 全対応を再投影し、前方かつ誤差2 px以下の点を数える。
    #         ④ 最良候補と反復上限を更新する。
    while iterations < limit and draws < 10 * RANSAC_MAX_ITERATIONS:
        draws += 1
        sample = np.sort(rng.choice(n, PNP_SAMPLE_SIZE, replace=False))
        sample_key = tuple(sample.tolist())
        if sample_key in used_samples:
            continue
        used_samples.add(sample_key)
        sample3D = np.ascontiguousarray(points3D[sample])
        sample2D = np.ascontiguousarray(points2D[sample])
        if not non_collinear_sample(sample3D, sample2D):
            continue
        iterations += 1
        try:
            result = cv2.solvePnPGeneric(sample3D.reshape(-1, 1, 3),
                                         sample2D.reshape(-1, 1, 2), K, None,
                                         flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        if not result[0]:
            continue
        # AP3P自体が返す複数の姿勢候補も、それぞれ評価する。
        for rotation_vector, translation_vector in zip(result[1], result[2]):
            if not np.all(np.isfinite(rotation_vector)) or not np.all(np.isfinite(translation_vector)):
                continue
            rotation = cv2.Rodrigues(rotation_vector)[0]
            translation = translation_vector.reshape(3)
            hypotheses += 1
            errors, depths = reprojection_errors(points3D, points2D, K, rotation, translation)
            mask, score = consensus_score(errors)
            if score > best_score:
                best_score = score
                best = PoseEstimate(rotation, translation, errors, depths, mask,
                                    0, 0, 0, 0, "")
                required = required_ransac_iterations(score[0] / n)
                limit = min(limit, max(RANSAC_MIN_ITERATIONS, required))

    best.iterations = iterations
    best.draws = draws
    best.hypotheses = hypotheses
    if best.rotation is None:
        best.reason = "有効なPnP姿勢を推定できませんでした"
        return best

    # 既存点を固定した姿勢だけの微調整。カメラと点を同時に動かすBAとは別。
    for _ in range(2):
        if np.count_nonzero(best.inliers) < PNP_SAMPLE_SIZE:
            break
        rotation_vector = cv2.Rodrigues(best.rotation)[0]
        translation_vector = best.translation.reshape(3, 1).copy()
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                np.ascontiguousarray(points3D[best.inliers]),
                np.ascontiguousarray(points2D[best.inliers]), K, None,
                rotation_vector, translation_vector)
        except cv2.error:
            break
        rotation = cv2.Rodrigues(refined_rvec)[0]
        translation = refined_tvec.reshape(3)
        errors, depths = reprojection_errors(points3D, points2D, K, rotation, translation)
        mask, score = consensus_score(errors)
        if score < best_score:
            break
        old_mask = best.inliers
        best_score = score
        best.rotation, best.translation = rotation, translation
        best.errors, best.depths, best.inliers = errors, depths, mask
        best.refinements += 1
        if np.array_equal(mask, old_mask):
            break

    count = int(np.count_nonzero(best.inliers))
    if count < MIN_PNP_INLIERS:
        best.reason = f"インライア数不足: {count} < {MIN_PNP_INLIERS}"
    elif count / n < MIN_PNP_INLIER_RATIO:
        best.reason = f"インライア率不足: {count / n:.3f} < {MIN_PNP_INLIER_RATIO}"
    else:
        best.reason = "ok"
    return best


def candidate_summary(rank, candidate):
    return {
        "rank": rank, "image_name": candidate.image_name,
        "raw_links": candidate.raw_links,
        "conflicting_features": candidate.conflicting_features,
        "duplicate_point_features": candidate.duplicate_point_features,
        "num_2d_3d": len(candidate.point_rows), "occupied_cells_4x4": candidate.occupied_cells,
        "status": "not_attempted", "pnp_inliers": "", "pnp_inlier_ratio": "",
        "median_error_px": "", "max_inlier_error_px": "", "num_in_front": "",
        "ransac_iterations": "", "sample_draws": "", "pose_hypotheses": "",
        "pose_refinements": "", "reason": "",
    }


def register_next_image(model, candidates, requested_image=None):
    """処理の制御: 候補順位とPnPの判断値を表示し、成功した1枚を返す。"""
    print("\n画像の順位: 2D-3D対応数の多い順（同数なら画像名順）")
    print(f"グリッド分布は参考値です。順位には使いません（{GRID_SIZE}x{GRID_SIZE}）。")
    summaries = []
    for rank, candidate in enumerate(candidates, start=1):
        summary = candidate_summary(rank, candidate)
        summaries.append(summary)
        print(f"{rank:2d}. {candidate.image_name}: 2D-3D対応{len(candidate.point_rows)}組、"
              f"元リンク{candidate.raw_links}組、矛盾する特徴点{candidate.conflicting_features}点、"
              f"同じ3D点への重複特徴点{candidate.duplicate_point_features}点、"
              f"分布{candidate.occupied_cells}/{GRID_SIZE ** 2}セル")
    if requested_image is not None:
        if requested_image in model["image_names"]:
            raise ValueError(f"すでに登録済みです: {requested_image}")
        candidate_names = set()
        for candidate in candidates:
            candidate_names.add(candidate.image_name)
        if requested_image not in candidate_names:
            raise ValueError(f"未登録の画像候補にありません: {requested_image}")
        print(f"画像指定により {requested_image} のみ試します。")

    last_candidate, last_result = None, None
    for candidate, summary in zip(candidates, summaries):
        if requested_image is not None and candidate.image_name != requested_image:
            summary["status"] = "not_requested"
            continue
        if len(candidate.point_rows) < MIN_PNP_INLIERS:
            summary["status"] = "insufficient_correspondences"
            summary["reason"] = f"2D-3D対応が{MIN_PNP_INLIERS}組未満"
            print(f"スキップ {candidate.image_name}: {summary['reason']}")
            continue
        result = estimate_pose_ransac(model["points3D"][candidate.point_rows],
                                      candidate.points2D, model["K"])
        last_candidate, last_result = candidate, result
        count = int(np.count_nonzero(result.inliers))
        median = float(np.median(result.errors[result.inliers])) if count else None
        maximum = float(np.max(result.errors[result.inliers])) if count else None
        summary.update({
            "status": "selected" if result.reason == "ok" else "pnp_failed",
            "pnp_inliers": count, "pnp_inlier_ratio": count / len(candidate.point_rows),
            "median_error_px": median if median is not None else "",
            "max_inlier_error_px": maximum if maximum is not None else "",
            "num_in_front": int(np.count_nonzero(result.depths > DEPTH_EPSILON)),
            "ransac_iterations": result.iterations, "sample_draws": result.draws,
            "pose_hypotheses": result.hypotheses, "pose_refinements": result.refinements,
            "reason": result.reason,
        })
        error_text = "算出不可" if median is None else f"中央値{median:.4f} px、最大{maximum:.4f} px"
        print(f"\nPnP {candidate.image_name}: 対応{len(candidate.point_rows)}組 → "
              f"インライア{count}組（{count / len(candidate.point_rows):.1%}）、"
              f"前方{summary['num_in_front']}点、誤差{error_text}")
        print(f"  RANSAC反復{result.iterations}回、抽選{result.draws}回、"
              f"姿勢仮説{result.hypotheses}個、姿勢微調整{result.refinements}回、判定: {result.reason}")
        if result.reason == "ok":
            return candidate, result, summaries
    return last_candidate, last_result, summaries


def update_model(model, candidate, result):
    """推定層: 固定した既存点へ、新画像の姿勢とインライア観測を追加する。"""
    updated = dict(model)
    n = len(model["points3D"])
    updated["schema_version"] = np.array(2)
    updated["stage"] = np.array("image_registration_only")
    updated["initial_image_names"] = model.get("initial_image_names", model["image_names"][:2]).copy()
    updated["image_names"] = np.append(model["image_names"], candidate.image_name)
    updated["rotations"] = np.concatenate((model["rotations"], result.rotation[None]), axis=0)
    updated["translations"] = np.concatenate((model["translations"], result.translation[None]), axis=0)
    updated["camera_centers"] = np.empty_like(updated["translations"])
    for column, (R, t) in enumerate(zip(updated["rotations"], updated["translations"])):
        updated["camera_centers"][column] = -R.T @ t
    indices = np.full(n, -1, dtype=np.int64)
    points2D = np.full((n, 2), np.nan)
    errors = np.full(n, np.nan)
    for match_row in np.flatnonzero(result.inliers):
        point_row = candidate.point_rows[match_row]
        indices[point_row] = candidate.keypoint_indices[match_row]
        points2D[point_row] = candidate.points2D[match_row]
        errors[point_row] = result.errors[match_row]
    updated["keypoint_indices"] = np.column_stack((model["keypoint_indices"], indices))
    updated["points2D"] = np.concatenate((model["points2D"], points2D[:, None]), axis=1)
    updated["reprojection_errors"] = np.column_stack((model["reprojection_errors"], errors))
    return updated


def write_results(camera, initial, initialization_summaries, model,
                  registration_summaries, candidate, pose):
    """出力層: 計算済みの二画像復元・画像登録の結果を保存する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    added_image = None
    if pose is not None and pose.reason == "ok":
        added_image = candidate.image_name
    metadata = {
        "image_names": model["image_names"].tolist(), "input_camera": camera,
        "K": model["K"].tolist(), "image_size": model["image_size"].tolist(),
        "coordinate_convention": "opencv",
        "pose_convention": "X_camera = R @ X_world + t; C = -R.T @ t",
        "axes": "x right, y down, z forward; first camera is the world frame",
        "scale": "arbitrary; initial camera baseline = 1; PnP translation is not normalized",
        "initial_image_names": model["initial_image_names"].tolist(),
        "rotations": model["rotations"].tolist(),
        "translations": model["translations"].tolist(),
        "camera_centers": model["camera_centers"].tolist(),
        "num_points3D": len(model["point_ids"]), "added_image": added_image,
        "initial_max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
        "initial_min_triangulation_angle_deg": MIN_TRIANGULATION_ANGLE_DEG,
        "initial_median_triangulation_angle_deg": float(np.median(initial.angles[initial.valid])),
        "pnp_max_reprojection_error_px": PNP_MAX_REPROJECTION_ERROR_PX,
        "min_pnp_inliers": MIN_PNP_INLIERS,
        "min_pnp_inlier_ratio": MIN_PNP_INLIER_RATIO,
        "min_pnp_inlier_ratio_usage": "additional final acceptance rule in this educational implementation",
        "pnp_inliers": int(np.count_nonzero(pose.inliers)) if pose is not None else 0,
        "pnp_status": pose.reason if pose is not None else "not_attempted",
        "selection_method": "unique 2D-3D count descending, then image name ascending",
        "missing_observation": "keypoint index -1; points2D and reprojection error NaN in NPZ",
        "F_and_E": "initial pair only; not the geometry of the added image",
        "stage": "image registration; existing 3D points and initial poses fixed",
        "run_behavior": "reconstruct two views and add one image from CSV on every run",
    }
    with TemporaryDirectory(dir=OUTPUT_DIR) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / "reconstruction.npz", **model)
        with (staging / "cameras.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for filename, summaries in (
            ("initialization_candidates.csv", initialization_summaries),
            ("registration_candidates.csv", registration_summaries),
        ):
            with (staging / filename).open("w", encoding="utf-8", newline="") as stream:
                if summaries:
                    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
        with (staging / "points3D.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "x", "y", "z", "num_observations"])
            for row, point in enumerate(model["points3D"]):
                writer.writerow([int(model["point_ids"][row]), *point,
                                 int(np.count_nonzero(model["keypoint_indices"][row] >= 0))])
        with (staging / "observations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "image_name", "keypoint_index", "x", "y", "error_px"])
            for row, point_id in enumerate(model["point_ids"]):
                for column, image in enumerate(model["image_names"].tolist()):
                    index = int(model["keypoint_indices"][row, column])
                    if index >= 0:
                        writer.writerow([int(point_id), image, index,
                                         *model["points2D"][row, column],
                                         model["reprojection_errors"][row, column]])
        # 候補がない場合にもヘッダーを書き、前回実行のCSVを残さない。
        with (staging / "pnp_correspondences.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_name", "keypoint_index", "point_id", "X", "Y", "Z",
                             "x", "y", "error_px", "depth", "is_inlier"])
            if candidate is not None and pose is not None:
                for row, point_row in enumerate(candidate.point_rows):
                    error = pose.errors[row]
                    depth = pose.depths[row]
                    writer.writerow([candidate.image_name, int(candidate.keypoint_indices[row]),
                                     int(model["point_ids"][point_row]), *model["points3D"][point_row],
                                     *candidate.points2D[row],
                                     float(error) if np.isfinite(error) else "",
                                     float(depth) if np.isfinite(depth) else "",
                                     int(pose.inliers[row])])
        for path in staging.iterdir():
            path.replace(OUTPUT_DIR / path.name)


def write_visualization(camera, model):
    """出力層: sfm1.pyと同じ描き方で、登録した全カメラを表示する。"""
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
    points = model["points3D"]
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
    normalized_corners = normalized_image_points(image_corners, model["K"])
    local_corners = np.column_stack((normalized_corners, np.ones(4))) * 0.4
    camera_points = []
    colors = ("#d34949", "#2867b2", "#d28b20", "#318b54")
    for index, (rotation, translation) in enumerate(
        zip(model["rotations"], model["translations"]), start=1
    ):
        color = colors[(index - 1) % len(colors)]
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
    figure.suptitle(" / ".join(model["image_names"].tolist()) + "　（尺度は任意）", fontsize=13)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISUALIZATION_DIR / "reconstruction.png", dpi=150)
    plt.close(figure)
    print(f"点群・カメラ配置の図：{VISUALIZATION_DIR / 'reconstruction.png'}")


def print_initialization(initial, summaries):
    """sfm1.pyと同じ判断値を表示し、続いて三枚目を追加する。"""
    for summary in summaries:
        angle = summary["median_angle_deg"]
        angle_text = "算出不可" if angle == "" else f"{angle:.4f}度"
        print(f"{summary['image1']} -> {summary['image2']}: "
              f"インライア{summary['num_inliers']}組、前方{summary['num_in_front']}点、"
              f"採用{summary['num_points3D']}点、三角測量角度の中央値{angle_text}、"
              f"{summary['status']}")
    print(f"選択した初期ペア: {initial.pair.image1} / {initial.pair.image2}")
    print(f"初期三次元点: {np.count_nonzero(initial.valid)}点")
    print("世界→2枚目カメラのR:\n" + np.array2string(initial.rotation, precision=8))
    print("世界→2枚目カメラのt: " + np.array2string(initial.translation, precision=8))
    print("2枚目カメラの中心 C = -R.T @ t: "
          + np.array2string(-initial.rotation.T @ initial.translation, precision=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("IMAGE1", "IMAGE2"),
                        help="初期ペアをRANSACのCSVと同じ画像順で指定。省略時は自動選択")
    parser.add_argument("--image", help="三枚目に追加する画像。省略時は対応数の順に試す")
    parser.add_argument("--visualize", action="store_true", help="点群とカメラ配置の図を保存")
    args = parser.parse_args()
    try:
        # 1. 入力。sfm1.pyと同じファイルを読み、特徴点辞書と全対応も受け取る。
        camera, K, pairs, keypoints, matches = load_inputs(args.pair)

        # 2. sfm1.pyと同じ二画像復元。ここまでの計算関数はそのまま使う。
        initial, initialization_summaries = reconstruct_two_views(pairs, K)
        print_initialization(initial, initialization_summaries)

        # 3. ここから追加。既存点に結び付く特徴点を探し、画像候補を順位付けする。
        model = make_initial_model(camera, initial)
        candidates = build_candidates(model, keypoints, matches)
        print("\nK（camera.jsonから作成し、二画像復元とPnPで共通に使用）:\n"
              + np.array2string(K, precision=6))
        print(f"PnP採用条件: 前方かつ再投影誤差{PNP_MAX_REPROJECTION_ERROR_PX} px以下の対応が"
              f"{MIN_PNP_INLIERS}組以上、かつ全対応の{MIN_PNP_INLIER_RATIO:.0%}以上")
        print(f"RANSAC: AP3P・{PNP_SAMPLE_SIZE}対応、信頼度{RANSAC_CONFIDENCE}、"
              f"反復{RANSAC_MIN_ITERATIONS}〜{RANSAC_MAX_ITERATIONS}回、seed={RANDOM_SEED}")

        # 4. PnP＋RANSACで姿勢を推定。失敗したら次の候補を試す。
        candidate, pose, registration_summaries = register_next_image(model, candidates, args.image)
        success = pose is not None and pose.reason == "ok"
        if success:
            model = update_model(model, candidate, pose)

        # 5. 出力。ここでは推定せず、上で求めた結果だけを保存する。
        write_results(camera, initial, initialization_summaries, model,
                      registration_summaries, candidate, pose)
        if args.visualize:
            write_visualization(camera, model)
        print(f"\n出力先: {OUTPUT_DIR}")
        if not success:
            print("三枚目を登録できませんでした。二画像の復元結果と候補の成否を保存しました。")
            raise SystemExit(1)
        print(f"追加した画像: {candidate.image_name}")
        print("世界→追加カメラのR:\n" + np.array2string(pose.rotation, precision=8))
        print("世界→追加カメラのt: " + np.array2string(pose.translation, precision=8))
        print("カメラ中心 C = -R.T @ t: "
              + np.array2string(model["camera_centers"][-1], precision=8))
        print(f"登録画像: 2 → {len(model['image_names'])}枚 / 三次元点: {len(model['points3D'])}点")
        print("初期二画像の姿勢・点の座標・尺度を引き継いでいます。")
        print("毎回CSVから二画像復元と三枚目の追加を行います。")
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
