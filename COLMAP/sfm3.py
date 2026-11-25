"""二画像の復元に続き、三枚目について1〜6を実行する。BAはNumPyで実装する。

配置・実行: python COLMAP/sfm3.py --visualize
画像指定: python COLMAP/sfm3.py --pair frame_000001.png frame_000006.png --image frame_000005.png
依存: NumPy、共通PnP処理にOpenCV。--visualize時のみMatplotlib。
sfm2.pyの共通関数を変更せず収録しているため、この一ファイルで実行できる。

入力: camera.json、output/matching_first/keypoints.csv、
      output/ransac/inlier_matches.csv、output/ransac/fundamental_matrices.csv
      パスはこのファイルのあるCOLMAPディレクトリを基準とする。
毎回CSVから二画像を復元し、三枚目を1枚追加する。保存済みNPZは読み込まない。

【1】2D–3D対応の作成: build_candidates()
【2】三枚目の選択: register_next_image()内で対応数順に候補を試す
【3】PnP＋RANSAC・登録: estimate_pose_ransac() → update_model()
【4】新規三角測量: triangulate_new_points()
【5】バンドル調整: bundle_adjustment()
【6】観測・三次元点のフィルタリング: filter_model()
main()にも、この番号を付けて呼び出す順序を示す。

BAの数式（MVGの記号に合わせる）:
    P^i = K [R^i | t^i],  x_j^i = 画像iにおける三次元点X_jの観測。
    E = sum_(i,j) || x_j^i - pi(P^i X_j) ||^2 = epsilon.T @ epsilon
    epsilon = x - f(p),  J = df/dp
    (J.T @ J + lambda * I) @ delta = J.T @ epsilon
    更新候補を実際に再投影し、Eが減ったら採用、減らなければlambdaを増やす。
    ロバスト損失はrho(s)=s。カメラ・点を同時に更新する。
    Jは中心差分、連立方程式はnp.linalg.solveで解く。BAはNumPyのみ。

回転は3成分の局所更新からRodriguesの式で作り、回転行列の制約を保つ。
Kと一台目の姿勢を固定。二台目の並進方向を2変数で動かし、長さは初期値に固定。
したがって、世界座標系と初期カメラ間距離（=1）を保ったまま最適化する。
Schur補完を省いた密行列の教育用実装で、三枚程度の小規模な復元を対象とする。

出力: output/fsm3/
    reconstruction.npz: 【6】の後の最終結果（pickle不要）
    before_ba.npz / after_ba.npz: 同じ点・観測でBA前後を確認するためのモデル
    cameras.json: 最終姿勢・K・設定・BAとフィルタリングの統計
    points3D.csv / observations.csv: 最終的な点と観測
    initialization_candidates.csv / registration_candidates.csv: 初期化・画像追加の判断値
    triangulation_candidates.csv: 【4】のトラックごとの成否
    ba_iterations.csv: 【5】のlambda、誤差、更新量、採否
    ba_observations.csv: 同じ観測のBA前後の誤差と【6】での採否
    pnp_correspondences.csv: 【3】の時点の対応と誤差
可視化: work/fsm3/reconstruction.png / ba_comparison.png

座標規約はsfm2.pyと同じ。X_camera = R @ X_world + t、C = -R.T @ t。
画像はOpenCV規約。camera.jsonの主点をload_camera()で一度だけ0.5ずらす。
FとEは初期ペアの推定結果として保持し、BA後のカメラ行列からの再計算はしない。
【4】は既存点を含むトラックと矛盾するトラックを保守的に除外する簡略版。
【6】の後のBA再実行・再三角測量の反復は、この段階では行わない。
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
OUTPUT_DIR = BASE_DIR / "output" / "fsm3"
VISUALIZATION_DIR = BASE_DIR / "work" / "fsm3"

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

# 以下は今回の教育用BAの設定。COLMAPの既定値ではない。
BA_MAX_ITERATIONS = 40
BA_MAX_TRIALS = 12
BA_INITIAL_LAMBDA_FACTOR = 1e-3
BA_DIFFERENCE_STEP = 1e-6
BA_RELATIVE_ERROR_TOLERANCE = 1e-8
BA_STEP_TOLERANCE = 1e-8
BA_GRADIENT_TOLERANCE = 1e-6



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


# ================================================================
# 【4】登録済み画像の対応から、新しい三次元点を三角測量する
# ================================================================

def feature_tracks(matches, registered_images):
    """登録済み画像間の対応をつなぎ、同じ点の候補を一つのトラックにする。"""
    graph = {}
    for image1, index1, image2, index2 in matches:
        if image1 not in registered_images or image2 not in registered_images:
            continue
        first, second = (image1, index1), (image2, index2)
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
    visited = set()
    tracks = []
    for start in sorted(graph):
        if start in visited:
            continue
        stack, track = [start], []
        visited.add(start)
        while stack:
            node = stack.pop()
            track.append(node)
            for neighbor in sorted(graph[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        tracks.append(sorted(track))
    return tracks


def maximum_viewing_angle(point, centers):
    """観測カメラの全組について調べ、最も大きい三角測量角度を返す。"""
    maximum = 0.0
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            # sfm2.pyの角度計算を使うため、一台目の中心を原点へ移す。
            angle = triangulation_angles(
                (point - centers[first])[None], centers[second] - centers[first])[0]
            maximum = max(maximum, float(angle))
    return maximum


def triangulate_world_point(model, columns, pixels, first, second):
    """sfm2.pyのDLTを使い、任意の登録済み二台から世界座標の点を求める。"""
    a, b = columns[first], columns[second]
    Ra, Rb = model["rotations"][a], model["rotations"][b]
    ta, tb = model["translations"][a], model["translations"][b]
    relative_R = Rb @ Ra.T
    relative_t = tb - relative_R @ ta
    normalized = normalized_image_points(pixels[[first, second]], model["K"])
    in_camera_a = triangulate_points(normalized[:1], normalized[1:], relative_R, relative_t)[0]
    return (in_camera_a - ta) @ Ra


def triangulate_new_points(model, keypoints, matches, added_image):
    """【4】新画像を含むトラックから点を追加。既存点を含むトラックは作り直さない。"""
    image_columns = {}
    known_features = set()
    for column, image in enumerate(model["image_names"].tolist()):
        image_columns[image] = column
        for index in model["keypoint_indices"][:, column]:
            if index >= 0:
                known_features.add((image, int(index)))
    tracks = feature_tracks(matches, image_columns)
    m = len(image_columns)
    new_points, new_indices, new_pixels, new_errors, new_angles = [], [], [], [], []
    summaries = []
    next_id = int(np.max(model["point_ids"])) + 1
    for track_id, track in enumerate(tracks):
        summary = {"track_id": track_id, "num_features": len(track), "status": "",
                   "point_id": "", "num_inliers": 0, "max_angle_deg": "",
                   "median_error_px": ""}
        summaries.append(summary)
        images = set()
        contains_existing = False
        for feature in track:
            images.add(feature[0])
            contains_existing = contains_existing or feature in known_features
        if added_image not in images:
            summary["status"] = "without_added_image"
            continue
        if contains_existing:
            summary["status"] = "contains_existing_point"
            continue
        if len(images) != len(track):
            summary["status"] = "conflicting_features_in_same_image"
            continue
        columns, pixels = [], []
        for image, index in track:
            columns.append(image_columns[image])
            pixels.append(keypoints[(image, index)])
        pixels = np.asarray(pixels)
        new_view_row = 0
        for row, feature in enumerate(track):
            if feature[0] == added_image:
                new_view_row = row
        best, best_score = None, (-1, -np.inf)
        # 三枚の段階では最大3組。各二視点の仮説を、トラック全体で評価する。
        for first in range(len(track)):
            for second in range(first + 1, len(track)):
                point = triangulate_world_point(model, columns, pixels, first, second)
                if not np.all(np.isfinite(point)):
                    continue
                errors = np.full(len(track), np.inf)
                for row, column in enumerate(columns):
                    errors[row] = reprojection_errors(
                        point[None], pixels[row:row + 1], model["K"],
                        model["rotations"][column], model["translations"][column])[0][0]
                inliers = errors <= MAX_REPROJECTION_ERROR_PX
                count = int(np.count_nonzero(inliers))
                if count < 2 or not inliers[new_view_row]:
                    continue
                centers = []
                for row in np.flatnonzero(inliers):
                    centers.append(model["camera_centers"][columns[row]])
                angle = maximum_viewing_angle(point, centers)
                if angle < MIN_TRIANGULATION_ANGLE_DEG:
                    continue
                score = (count, -float(np.median(errors[inliers])))
                if score > best_score:
                    best_score = score
                    best = (point, errors, inliers, angle)
        if best is None:
            summary["status"] = "failed_geometry_or_reprojection"
            continue
        point, errors, inliers, angle = best
        indices = np.full(m, -1, dtype=np.int64)
        observations = np.full((m, 2), np.nan)
        point_errors = np.full(m, np.nan)
        for row in np.flatnonzero(inliers):
            column = columns[row]
            indices[column] = track[row][1]
            observations[column] = pixels[row]
            point_errors[column] = errors[row]
        summary.update({"status": "added", "point_id": next_id + len(new_points),
                        "num_inliers": best_score[0], "max_angle_deg": angle,
                        "median_error_px": -best_score[1]})
        new_points.append(point)
        new_indices.append(indices)
        new_pixels.append(observations)
        new_errors.append(point_errors)
        new_angles.append(angle)
    updated = dict(model)
    if new_points:
        updated["points3D"] = np.vstack((model["points3D"], new_points))
        updated["point_ids"] = np.concatenate((model["point_ids"],
                                               np.arange(next_id, next_id + len(new_points))))
        updated["keypoint_indices"] = np.vstack((model["keypoint_indices"], new_indices))
        updated["points2D"] = np.concatenate((model["points2D"], new_pixels), axis=0)
        updated["reprojection_errors"] = np.vstack((model["reprojection_errors"], new_errors))
        updated["triangulation_angles"] = np.concatenate((model["triangulation_angles"], new_angles))
    updated["stage"] = np.array("triangulated_before_ba")
    print(f"【4】三角測量: 既存{len(model['points3D'])}点 + 新規{len(new_points)}点"
          f" = {len(updated['points3D'])}点")
    return updated, summaries


# ================================================================
# 【5】NumPyで記述するバンドル調整（ロバスト損失なし）
# MVGに合わせ、P^i=K[R^i|t^i], X_j, x_j^i と表す。
# eps = x - f(p), J = df/dp と定義するため、正規方程式の右辺は +J.T @ eps。
# ================================================================

@dataclass
class BAProblem:
    point_rows: np.ndarray
    image_columns: np.ndarray
    measured: np.ndarray
    camera_blocks: list
    point_offset: int
    num_parameters: int
    baseline: float


def make_ba_problem(model):
    """観測を固定し、更新ベクトルdeltaの各成分の意味を決める。"""
    point_rows, image_columns = np.nonzero(model["keypoint_indices"] >= 0)
    measured = model["points2D"][point_rows, image_columns].copy()
    blocks, offset = [], 0
    for column in range(1, len(model["image_names"])):
        rotation_slice = slice(offset, offset + 3)
        offset += 3
        # カメラ2は並進の長さを固定し、方向だけを2変数で調整する。
        translation_size = 2 if column == 1 else 3
        translation_slice = slice(offset, offset + translation_size)
        offset += translation_size
        blocks.append((column, rotation_slice, translation_slice))
    if not np.allclose(model["rotations"][0], np.eye(3)) or not np.allclose(model["translations"][0], 0):
        raise ValueError("今回のBAでは一台目を原点・回転なしとしてください。")
    baseline = float(np.linalg.norm(model["translations"][1]))
    if baseline <= DEPTH_EPSILON or len(measured) == 0:
        raise ValueError("BAの基準距離または観測数が不正です。")
    return BAProblem(point_rows, image_columns, measured, blocks, offset,
                     offset + 3 * len(model["points3D"]), baseline)


def rotation_from_vector(vector):
    """NumPyによるRodriguesの式。3変数の回転更新から、正しい回転行列を作る。"""
    x, y, z = vector
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle_squared = float(vector @ vector)
    if angle_squared < 1e-12:
        a = 1.0 - angle_squared / 6.0
        b = 0.5 - angle_squared / 24.0
    else:
        angle = np.sqrt(angle_squared)
        a = np.sin(angle) / angle
        b = (1.0 - np.cos(angle)) / angle_squared
    return np.eye(3) + a * skew + b * (skew @ skew)


def translation_tangent_basis(translation):
    """並進方向と直交する二方向。単位球上で方向だけを動かすための座標軸。"""
    direction = translation / np.linalg.norm(translation)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.column_stack((first, second))


def apply_ba_update(model, delta, problem):
    """p_new = p (+) delta。回転と固定距離の制約を保って更新する。"""
    updated = dict(model)
    updated["rotations"] = model["rotations"].copy()
    updated["translations"] = model["translations"].copy()
    for column, rotation_slice, translation_slice in problem.camera_blocks:
        dR = rotation_from_vector(delta[rotation_slice])
        updated["rotations"][column] = dR @ model["rotations"][column]
        if column == 1:
            basis = translation_tangent_basis(model["translations"][column])
            direction = model["translations"][column] + basis @ delta[translation_slice]
            updated["translations"][column] = problem.baseline * direction / np.linalg.norm(direction)
        else:
            updated["translations"][column] += delta[translation_slice]
    updated["points3D"] = model["points3D"] + delta[problem.point_offset:].reshape(-1, 3)
    # カメラ1とKは更新しない。カメラ2の||t||固定により、尺度も変わらない。
    return updated


def ba_projection(model, problem):
    """f(p): 固定した観測順に、再投影座標を縦に並べる。"""
    projected = np.empty_like(problem.measured)
    positive = True
    for column in range(len(model["image_names"])):
        rows = problem.image_columns == column
        points = model["points3D"][problem.point_rows[rows]]
        projected[rows] = project_points(points, model["K"], model["rotations"][column],
                                         model["translations"][column])
        depths = (points @ model["rotations"][column].T + model["translations"][column])[:, 2]
        positive = positive and bool(np.all(depths > DEPTH_EPSILON))
    return projected.reshape(-1), positive


def numerical_jacobian(model, problem):
    """J[:, k] ≈ (f(p (+) h*e_k) - f(p (+) (-h*e_k))) / (2h)。"""
    jacobian = np.empty((2 * len(problem.measured), problem.num_parameters))
    step = BA_DIFFERENCE_STEP
    for column in range(problem.num_parameters):
        delta = np.zeros(problem.num_parameters)
        delta[column] = step
        plus = ba_projection(apply_ba_update(model, delta, problem), problem)[0]
        delta[column] = -step
        minus = ba_projection(apply_ba_update(model, delta, problem), problem)[0]
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def refresh_model_geometry(model):
    """推定層: 現在の姿勢・点で、中心・再投影誤差・最大三角測量角度を更新する。"""
    updated = dict(model)
    centers = np.empty_like(model["translations"])
    errors = np.full(model["keypoint_indices"].shape, np.nan)
    for column, (rotation, translation) in enumerate(zip(model["rotations"], model["translations"])):
        centers[column] = -rotation.T @ translation
        observed = model["keypoint_indices"][:, column] >= 0
        errors[observed, column] = reprojection_errors(
            model["points3D"][observed], model["points2D"][observed, column],
            model["K"], rotation, translation)[0]
    angles = np.zeros(len(model["points3D"]))
    for row, point in enumerate(model["points3D"]):
        observed = model["keypoint_indices"][row] >= 0
        angles[row] = maximum_viewing_angle(point, centers[observed])
    updated["camera_centers"] = centers
    updated["reprojection_errors"] = errors
    updated["triangulation_angles"] = angles
    return updated


def bundle_adjustment(model):
    """【5】eps=x-f(p), (J.T@J + lambda*I)@delta=J.T@eps をNumPyで解く。"""
    problem = make_ba_problem(model)
    projected, positive = ba_projection(model, problem)
    if not positive or not np.all(np.isfinite(projected)):
        raise ValueError("BA開始時の観測に、後方の点または不正な再投影があります。")
    measured = problem.measured.reshape(-1)
    epsilon = measured - projected
    initial_error = float(epsilon @ epsilon)
    error = initial_error
    current, history = model, []
    damping = None
    accepted_steps = 0
    status = "max_iterations"
    print(f"【5】BA: 観測{len(problem.measured)}組、変数{problem.num_parameters}個、"
          f"初期二乗和E={error:.6f}")
    identity = np.eye(problem.num_parameters)
    for iteration in range(1, BA_MAX_ITERATIONS + 1):
        # (a) J=df/dp、観測との差eps=x-f(p)を作る。
        J = numerical_jacobian(current, problem)
        if not np.all(np.isfinite(J)):
            raise ValueError("BAのヤコビ行列に非有限値が生じました。")
        JTJ = J.T @ J
        right_hand_side = J.T @ epsilon
        gradient = float(np.max(np.abs(right_hand_side)))
        if gradient < BA_GRADIENT_TOLERANCE:
            status = "gradient_converged"
            break
        if damping is None:
            damping = BA_INITIAL_LAMBDA_FACTOR * max(float(np.max(np.diag(JTJ))), 1.0)
        accepted = False
        for trial in range(1, BA_MAX_TRIALS + 1):
            # (b) LMの正規方程式。逆行列を作らず、連立一次方程式として解く。
            #     (J^T J + lambda I) delta = J^T epsilon
            try:
                delta = np.linalg.solve(JTJ + damping * identity, right_hand_side)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            # (c) p_new=p (+) delta。制約を保った更新後、近似でなく実際の誤差を評価する。
            proposed = apply_ba_update(current, delta, problem)
            prediction, positive = ba_projection(proposed, problem)
            new_epsilon = measured - prediction
            new_error = float(new_epsilon @ new_epsilon)
            valid = positive and np.all(np.isfinite(delta)) and np.isfinite(new_error)
            accepted = bool(valid and new_error < error)
            step_norm = float(np.linalg.norm(delta))
            history.append({"iteration": iteration, "trial": trial, "lambda": damping,
                            "error_before": error,
                            "error_trial": new_error if np.isfinite(new_error) else "",
                            "step_norm": step_norm, "gradient_inf": gradient,
                            "accepted": int(accepted)})
            decision = "採用" if accepted else "棄却"
            print(f"  LM {iteration:02d}.{trial:02d}: E={error:.6f} → {new_error:.6f}、"
                  f"lambda={damping:.3e}、||delta||={step_norm:.3e}、{decision}")
            if accepted:
                relative_decrease = (error - new_error) / max(error, 1e-15)
                current, projected, epsilon, error = proposed, prediction, new_epsilon, new_error
                accepted_steps += 1
                damping = max(damping / 3.0, 1e-15)
                if relative_decrease < BA_RELATIVE_ERROR_TOLERANCE:
                    status = "error_converged"
                elif step_norm < BA_STEP_TOLERANCE:
                    status = "step_converged"
                break
            # (d) 誤差が減らなければ、同じJでlambdaを増やして小さな更新を試す。
            damping = min(damping * 10.0, 1e30)
        if not accepted:
            status = "no_acceptable_step"
            break
        if status in ("error_converged", "step_converged"):
            break
    current = refresh_model_geometry(current)
    current["stage"] = np.array("after_bundle_adjustment")
    stats = {"status": status, "num_observations": len(problem.measured),
             "num_parameters": problem.num_parameters, "accepted_steps": accepted_steps,
             "error_before": initial_error, "error_after": error,
             "rms_before_px": float(np.sqrt(initial_error / len(problem.measured))),
             "rms_after_px": float(np.sqrt(error / len(problem.measured))),
             "fixed_baseline": problem.baseline}
    print(f"【5】BA完了: 同じ{len(problem.measured)}観測の距離RMS "
          f"{stats['rms_before_px']:.6f} → {stats['rms_after_px']:.6f} px、"
          f"採用更新{accepted_steps}回、終了理由={status}")
    return current, history, stats


# ================================================================
# 【6】再投影誤差・前方条件・三角測量角度で観測と点をフィルタリング
# ================================================================

def filter_model(model):
    """【6】悪い観測を外し、二視点未満または角度不足となった点を取り除く。"""
    updated = dict(model)
    updated["keypoint_indices"] = model["keypoint_indices"].copy()
    updated["points2D"] = model["points2D"].copy()
    updated["reprojection_errors"] = model["reprojection_errors"].copy()
    observed = model["keypoint_indices"] >= 0
    bad = observed & (~np.isfinite(model["reprojection_errors"])
                      | (model["reprojection_errors"] > MAX_REPROJECTION_ERROR_PX))
    updated["keypoint_indices"][bad] = -1
    updated["points2D"][bad] = np.nan
    updated["reprojection_errors"][bad] = np.nan
    keep = np.zeros(len(model["points3D"]), dtype=bool)
    angles = np.zeros(len(keep))
    too_few, too_small_angle = 0, 0
    for row, point in enumerate(model["points3D"]):
        visible = updated["keypoint_indices"][row] >= 0
        if np.count_nonzero(visible) < 2:
            too_few += 1
            continue
        angles[row] = maximum_viewing_angle(point, model["camera_centers"][visible])
        if angles[row] < MIN_TRIANGULATION_ANGLE_DEG:
            too_small_angle += 1
            continue
        keep[row] = True
    removed_with_points = int(np.count_nonzero(updated["keypoint_indices"][~keep] >= 0))
    for name in ("point_ids", "points3D", "keypoint_indices", "points2D", "reprojection_errors"):
        updated[name] = updated[name][keep].copy()
    updated["triangulation_angles"] = angles[keep]
    updated["stage"] = np.array("filtered_after_bundle_adjustment")
    stats = {"points_before": len(keep), "points_after": int(np.count_nonzero(keep)),
             "removed_bad_observations": int(np.count_nonzero(bad)),
             "removed_points_with_fewer_than_two_views": too_few,
             "removed_points_with_small_angle": too_small_angle,
             "removed_observations_with_points": removed_with_points,
             "observations_before": int(np.count_nonzero(observed)),
             "observations_after": int(np.count_nonzero(updated["keypoint_indices"] >= 0))}
    print(f"【6】フィルタリング: 誤差・前方条件で観測{stats['removed_bad_observations']}組を除外、"
          f"二視点未満{too_few}点、角度不足{too_small_angle}点を除外、"
          f"三次元点{len(keep)} → {stats['points_after']}点")
    return updated, stats


def write_summary_csv(path, rows):
    """出力層: 計算済みの辞書の列を書き出す。"""
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_fsm3_results(camera, initial, initialization_summaries, registration_summaries,
                       candidate, pose, registered, before_ba, after_ba, final,
                       triangulation_summaries, history, ba_stats, filtering_stats):
    """出力層: 各段階の計算済みデータを保存。出力中には推定・最適化を行わない。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "image_names": final["image_names"].tolist(), "input_camera": camera,
        "K": final["K"].tolist(), "rotations": final["rotations"].tolist(),
        "translations": final["translations"].tolist(),
        "camera_centers": final["camera_centers"].tolist(),
        "coordinate_convention": "opencv",
        "pose_convention": "X_camera = R @ X_world + t; C = -R.T @ t",
        "initial_image_names": final["initial_image_names"].tolist(),
        "added_image": candidate.image_name, "pnp_inliers": int(np.count_nonzero(pose.inliers)),
        "initial_points": len(registered["points3D"]),
        "new_points_before_ba": len(before_ba["points3D"]) - len(registered["points3D"]),
        "num_points3D": len(final["points3D"]), "bundle_adjustment": ba_stats,
        "filtering": filtering_stats,
        "ba_objective": "sum of squared 2D Euclidean reprojection distances; rho(s)=s",
        "ba_normal_equation": "(J.T @ J + lambda * I) @ delta = J.T @ epsilon",
        "ba_derivative": "central finite differences; J=df/d(delta) at delta=0",
        "ba_update": "local rotation increments and two-dimensional direction update for camera 2",
        "fixed_parameters": "K, pose of camera 1, initial camera baseline",
        "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
        "min_triangulation_angle_deg": MIN_TRIANGULATION_ANGLE_DEG,
        "triangulation_angle": "maximum stable angle over all observed view pairs",
        "F_and_E": "initial pair estimates before BA; not recomputed after BA",
        "pnp_csv": "coordinates and errors at step 3, before triangulation and BA",
        "run_behavior": "reconstruct two views from CSV and perform steps 1-6 for one added image",
        "missing_observation": "keypoint index=-1; 2D coordinates and errors=NaN in NPZ",
    }
    with TemporaryDirectory(dir=OUTPUT_DIR) as temporary:
        staging = Path(temporary)
        for filename, model in (("before_ba.npz", before_ba), ("after_ba.npz", after_ba),
                                ("reconstruction.npz", final)):
            np.savez_compressed(staging / filename, **model)
        with (staging / "cameras.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for filename, rows in (("initialization_candidates.csv", initialization_summaries),
                               ("registration_candidates.csv", registration_summaries),
                               ("triangulation_candidates.csv", triangulation_summaries),
                               ("ba_iterations.csv", history)):
            write_summary_csv(staging / filename, rows)
        with (staging / "points3D.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "x", "y", "z", "num_observations", "max_angle_deg"])
            for row, point_id in enumerate(final["point_ids"]):
                writer.writerow([int(point_id), *final["points3D"][row],
                                 int(np.count_nonzero(final["keypoint_indices"][row] >= 0)),
                                 final["triangulation_angles"][row]])
        final_rows = {}
        for row, point_id in enumerate(final["point_ids"]):
            final_rows[int(point_id)] = row
        with (staging / "observations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "image_name", "keypoint_index", "x", "y", "error_px"])
            for row, point_id in enumerate(final["point_ids"]):
                for column, image in enumerate(final["image_names"].tolist()):
                    index = int(final["keypoint_indices"][row, column])
                    if index >= 0:
                        writer.writerow([int(point_id), image, index, *final["points2D"][row, column],
                                         final["reprojection_errors"][row, column]])
        with (staging / "ba_observations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_id", "image_name", "keypoint_index", "error_before_px",
                             "error_after_px", "retained_after_filtering"])
            for row, point_id in enumerate(before_ba["point_ids"]):
                for column, image in enumerate(before_ba["image_names"].tolist()):
                    index = int(before_ba["keypoint_indices"][row, column])
                    if index < 0:
                        continue
                    final_row = final_rows.get(int(point_id))
                    retained = final_row is not None and final["keypoint_indices"][final_row, column] == index
                    writer.writerow([int(point_id), image, index, before_ba["reprojection_errors"][row, column],
                                     after_ba["reprojection_errors"][row, column], int(retained)])
        with (staging / "pnp_correspondences.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_name", "keypoint_index", "point_id", "X", "Y", "Z",
                             "x", "y", "error_px", "depth", "is_inlier"])
            for row, point_row in enumerate(candidate.point_rows):
                error = pose.errors[row]
                writer.writerow([candidate.image_name, int(candidate.keypoint_indices[row]),
                                 int(registered["point_ids"][point_row]), *registered["points3D"][point_row],
                                 *candidate.points2D[row], float(error) if np.isfinite(error) else "",
                                 float(pose.depths[row]) if np.isfinite(pose.depths[row]) else "",
                                 int(pose.inliers[row])])
        for path in staging.iterdir():
            path.replace(OUTPUT_DIR / path.name)


def write_ba_visualization(before, after, history):
    """同じ観測集合の誤差と、LMの目的関数を図にする。フィルタ後とは混ぜない。"""
    import matplotlib.pyplot as plt
    observed = before["keypoint_indices"] >= 0
    old_errors = before["reprojection_errors"][observed]
    new_errors = after["reprojection_errors"][observed]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    upper = max(float(np.max(old_errors)), float(np.max(new_errors)), 1e-6)
    bins = np.linspace(0.0, upper, 36)  # 同じ区間幅で観測数を比較する。
    axes[0].hist(old_errors, bins=bins, alpha=0.5, label="BA前")
    axes[0].hist(new_errors, bins=bins, alpha=0.5, label="BA後")
    axes[0].set(xlabel="再投影誤差（px）", ylabel="観測数", title="同じ観測集合での比較")
    axes[0].legend()
    if history:
        values = [history[0]["error_before"]]
        for row in history:
            if row["accepted"]:
                values.append(row["error_trial"])
        axes[1].plot(np.arange(len(values)), values, marker="o", markersize=3)
    axes[1].set(xlabel="採用した更新の回数", ylabel="再投影誤差の二乗和 E", title="LM法の誤差の変化")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(VISUALIZATION_DIR / "ba_comparison.png", dpi=150)
    plt.close(figure)
    print(f"BA前後の比較: {VISUALIZATION_DIR / 'ba_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("IMAGE1", "IMAGE2"), help="初期ペアを指定")
    parser.add_argument("--image", help="追加する三枚目を指定。省略時は対応数順に試す")
    parser.add_argument("--visualize", action="store_true", help="点群・カメラとBA前後の比較を保存")
    args = parser.parse_args()
    try:
        # 前提: sfm2.pyと同じ入力・二画像復元。共通関数を変更せずに使用する。
        camera, K, pairs, keypoints, matches = load_inputs(args.pair)
        initial, initialization_summaries = reconstruct_two_views(pairs, K)
        print_initialization(initial, initialization_summaries)
        initial_model = make_initial_model(camera, initial)

        # 【1】2D–3D対応を作る。build_candidates()は対応数による順位付けも行う。
        print("\n【1】既存の三次元点と未登録画像の特徴点を結び付ける")
        candidates = build_candidates(initial_model, keypoints, matches)

        # 【2】三枚目を選ぶ。共通関数register_next_image()が上位候補から順に試す。
        print("\n【2】2D–3D対応数の多い順に、三枚目の候補を選ぶ")
        # 【3】同関数内のestimate_pose_ransac()がPnP＋RANSACを実行する。
        print(f"【3】PnP＋RANSAC: 前方・誤差{PNP_MAX_REPROJECTION_ERROR_PX} px以下が"
              f"{MIN_PNP_INLIERS}組以上、かつ{MIN_PNP_INLIER_RATIO:.0%}以上で登録")
        candidate, pose, registration_summaries = register_next_image(initial_model, candidates, args.image)
        if pose is None or pose.reason != "ok":
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            write_summary_csv(OUTPUT_DIR / "registration_candidates.csv", registration_summaries)
            parser.exit(1, "三枚目を登録できませんでした。候補の評価CSVを確認してください。\n")
        registered = update_model(initial_model, candidate, pose)
        print(f"【3】登録した画像: {candidate.image_name}")

        # 【4】新しい三次元点を追加。既存のDLT・再投影・角度計算を再利用する。
        before_ba, triangulation_summaries = triangulate_new_points(
            registered, keypoints, matches, candidate.image_name)

        # 【5】BA。内部の式(a)〜(d)とLMの数式を対応させている。
        after_ba, history, ba_stats = bundle_adjustment(before_ba)

        # 【6】再投影誤差・前方条件・角度でフィルタリングする。
        final, filtering_stats = filter_model(after_ba)

        # 出力層: 1〜6で計算した結果だけを書き出す。
        write_fsm3_results(camera, initial, initialization_summaries, registration_summaries,
                           candidate, pose, registered, before_ba, after_ba, final,
                           triangulation_summaries, history, ba_stats, filtering_stats)
        if args.visualize:
            write_visualization(camera, final)
            write_ba_visualization(before_ba, after_ba, history)
        print(f"\n出力先: {OUTPUT_DIR}")
        print(f"最終結果: カメラ{len(final['image_names'])}台、三次元点{len(final['points3D'])}点")
        for column, image in enumerate(final["image_names"].tolist()):
            count = int(np.count_nonzero(final["keypoint_indices"][:, column] >= 0))
            print(f"  {image}: 観測{count}点、中心{final['camera_centers'][column]}")
        print("毎回、CSVから二画像を復元し、三枚目について1〜6を一回実行します。")
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
