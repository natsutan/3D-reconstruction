"""動画から多画像SfMを行い、画像付きCOLMAPテキストモデルを保存する。

配置: COLMAP/sfm5.py。従来の.pyをimportせず、この一ファイルで実行できる。
実行: python COLMAP/sfm5.py
      python COLMAP/sfm5.py --max-frames 20   # 最初の動作確認
      python COLMAP/sfm5.py --resume          # 同じ設定・動画で再開
依存: python -m pip install numpy opencv-python-headless pillow
      ffmpegをインストールし、PATHを通す。GPU不要。
      --visualize時のみmatplotlibも必要。

パスは、このスクリプトのあるディレクトリを基準にする。
入力: input/input.mov、camera.json（従来のSIMPLE_PINHOLE/PINHOLE）
画像: work/frames/frame_000001.png ...（標準2 fps、camera.jsonのサイズ）
途中結果: work/sfm5/（CSV、再開用NPZ、進捗状態、判断値）
最終出力: ../3DGS/input/colmap/images/ と sparse/0/ の三つの.txt

処理の対応:
  工程1: FFmpegで画像抽出。縦横比がcamera.jsonと異なる動画は停止する。
  工程2: 既存select_pairs / match_featuresでSIFTとシーケンシャルマッチング。
  工程3: 既存estimate_fundamental_matrixの8点法＋自作RANSAC。
  工程4: 既存reconstruct_pairで二画像を初期化する。
  工程5: sfm3の【1】対応作成【2】画像選択【3】PnP登録
         【4】新規三角測量【5】BA【6】フィルタリングを繰り返す。
  工程6: sfm4の変換・出力関数でCOLMAPモデルと登録画像を保存する。

共通関数は可能な限りそのまま収録。関数名や定数が重なる箇所のみ区別した。
追加・変更点:
  ・各工程・画像・ペア・LM反復の進捗と、15秒間隔の処理継続表示。
  ・初期ペアの評価は、採用点数を超えられない候補を上限で省略する。
  ・小規模BAは従来の中心差分＋密行列。大規模BAは同じLM方程式を
    解析的な局所ヤコビ行列とSchur補完で解く（いずれもNumPy）。
    epsilon=x-f(p), (J.T J + lambda I) delta = J.T epsilon。
  ・BAは三枚目、以後5枚追加ごと、登録が止まった時、最後に実行。
  ・長いトラックの三角測量は最大32組の二視点仮説を全観測で検証する。
    三視点以下では従来関数をそのまま使う。
  ・三角測量角度の全組評価をNumPyでまとめて計算する。
  ・失敗画像はモデルが成長するたびに再候補になる。
    最後まで登録できない画像はimage_status.csvに理由を残す。

Kは固定、歪みは無視。カメラ1の姿勢と初期二台間の尺度は固定する。
世界→カメラ: X_camera=R@X_world+t。画像内はOpenCV規約、出力時に+0.5。
標準設定は約50秒・2 fps（約100枚）の教育用。完全なCOLMAPの再実装ではない。
初期化できない独立成分や、ぼけ・視差不足などの画像は復元できない場合がある。
最終出力先はエクスポート専用。完了時にフォルダ全体を入れ替える。
同じ入力で--resumeすると、完了した工程と最後の画像追加から再開する。
設定・動画・抽出画像が変わった場合は、--resumeを外して実行し直す。
仕様: https://colmap.github.io/format.html
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

# 小さい行列を何度も解くため、BLASの過剰なスレッド起動を避ける。
# 起動前に明示された値は変更しない。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "input" / "input.mov"
CAMERA_PATH = BASE_DIR / "camera.json"
FRAME_DIR = BASE_DIR / "work" / "frames"
IMAGE_DIR = FRAME_DIR
WORK_DIR = BASE_DIR / "work" / "sfm5"
MATCHING_DIR = WORK_DIR / "matching"
INPUT_DIR = MATCHING_DIR
RANSAC_DIR = WORK_DIR / "ransac"
KEYPOINTS_PATH = MATCHING_DIR / "keypoints.csv"
MATCHES_PATH = RANSAC_DIR / "inlier_matches.csv"
MATRICES_PATH = RANSAC_DIR / "fundamental_matrices.csv"
CHECKPOINT_PATH = WORK_DIR / "reconstruction.npz"
MODEL_PATH = CHECKPOINT_PATH
LEGACY_MODEL_PATH = CHECKPOINT_PATH
VISUALIZATION_DIR = WORK_DIR / "visualizations"
OUTPUT_DIR = BASE_DIR.parent / "3DGS" / "input" / "colmap"
STATE_PATH = WORK_DIR / "run.json"
PIPELINE_VERSION = 1

# match_frames.pyと同じ設定。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FEATURES = 2000
RATIO_THRESHOLD = 0.75

# ransac_matches.pyのF推定用定数。PnP用と名前を区別する。
RANSAC_THRESHOLD_PX = 1.0
F_RANSAC_CONFIDENCE = 0.999
F_RANSAC_MAX_ITERATIONS = 10000
F_RANDOM_SEED = 0
MIN_MATCHES = 8
REFINEMENT_STEPS = 3

# sfm3.pyと同じ幾何・PnP・LMの設定。
MIN_INITIAL_POINTS = 8
MAX_REPROJECTION_ERROR_PX = 2.0
MIN_TRIANGULATION_ANGLE_DEG = 1.0
DEPTH_EPSILON = 1e-9
PNP_SAMPLE_SIZE = 4
PNP_MAX_REPROJECTION_ERROR_PX = MAX_REPROJECTION_ERROR_PX
MIN_PNP_INLIERS = 30
MIN_PNP_INLIER_RATIO = 0.25
RANSAC_CONFIDENCE = 0.9999
RANSAC_MIN_ITERATIONS = 30
RANSAC_MAX_ITERATIONS = 10000
RANDOM_SEED = 0
GRID_SIZE = 4
BA_MAX_ITERATIONS = 20  # --ba-iterationsで変更可能。従来の上限40回から短縮。
BA_MAX_TRIALS = 12
BA_INITIAL_LAMBDA_FACTOR = 1e-3
BA_DIFFERENCE_STEP = 1e-6
BA_RELATIVE_ERROR_TOLERANCE = 1e-8
BA_STEP_TOLERANCE = 1e-8
BA_GRADIENT_TOLERANCE = 1e-6
DENSE_BA_PARAMETER_LIMIT = 1200
MAX_TRIANGULATION_HYPOTHESES = 32

# === match_frames.pyから再利用（進捗通知だけ追加） ===

def select_frames(image_dir, start, count):
    # FFmpeg's zero-padded names sort in capture order.
    frames = []
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            frames.append(path)
    frames.sort()
    start_index = start - 1
    selected = frames[start_index:]
    if count is not None:
        selected = selected[:count]
    if len(selected) < 2:
        raise ValueError("Select at least two JPEG or PNG images.")
    return selected



def select_pairs(frames, *, first_only=False):
    frame_count = len(frames)
    offsets = list(range(1, 6))
    # Keep the five nearby frames and sample longer gaps more sparsely.
    offset = 10
    while offset < frame_count:
        offsets.append(offset)
        offset *= 2

    reference_frames = frames[:1] if first_only else frames
    pairs = []
    for index, frame in enumerate(reference_frames):
        for offset in offsets:
            second_index = index + offset
            if second_index >= frame_count:
                break
            pairs.append((frame, frames[second_index], offset))
    return pairs



def write_pairs(path, pairs):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image1", "image2", "frame_gap"])
        for first, second, gap in pairs:
            writer.writerow([first.name, second.name, gap])



def match_features(frames, pairs, csv_output, visualization_output, progress_callback=None):
    import cv2
    import numpy as np

    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    descriptors = {}
    color_images = {}
    image_keypoints = {}
    with (csv_output / "keypoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "keypoint_index", "x", "y"])
        for frame_number, frame in enumerate(frames, start=1):
            # imdecode supports paths with Japanese characters on Windows.
            encoded = np.fromfile(frame, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None
            if image is None:
                raise ValueError(f"Cannot read image: {frame}")
            keypoints, descriptors[frame] = sift.detectAndCompute(image, None)
            if progress_callback is not None:
                progress_callback("SIFT", frame_number, len(frames),
                                  f"{frame.name}: {len(keypoints)}点")
            for index, point in enumerate(keypoints):
                x, y = point.pt
                writer.writerow([frame.name, index, x, y])

            if visualization_output is not None:
                color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if color is None:
                    raise ValueError(f"Cannot read color image: {frame}")
                color_images[frame] = color
                image_keypoints[frame] = keypoints

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pair_counts = []
    with (csv_output / "matches.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image1", "image2", "keypoint_index1", "keypoint_index2"])
        for pair_number, (first, second, _) in enumerate(pairs, start=1):
            query, train = descriptors[first], descriptors[second]
            candidates = []
            if query is not None and train is not None and len(train) >= 2:
                for nearest, runner_up in matcher.knnMatch(query, train, k=2):
                    if nearest.distance < RATIO_THRESHOLD * runner_up.distance:
                        candidates.append(nearest)

            for match in candidates:
                writer.writerow([first.name, second.name, match.queryIdx, match.trainIdx])
            pair_counts.append(len(candidates))
            if progress_callback is not None:
                progress_callback("マッチング", pair_number, len(pairs),
                                  f"{first.name} → {second.name}: {len(candidates)}組")

            if visualization_output is not None:
                canvas = cv2.drawMatches(
                    color_images[first], image_keypoints[first],
                    color_images[second], image_keypoints[second],
                    candidates, None, matchColor=(60, 230, 60),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                name = f"match_{pair_number:04d}_{first.stem}__{second.stem}.png"
                saved, encoded = cv2.imencode(".png", canvas)
                if not saved:
                    raise ValueError(f"Cannot encode match image: {name}")
                encoded.tofile(visualization_output / name)
    return pair_counts



# === ransac_matches.pyから再利用（衝突する名前のみ変更） ===

@dataclass
class MatchingPair:
    image1: str
    image2: str
    frame_gap: int
    matches: list
    points1: np.ndarray
    points2: np.ndarray



@dataclass
class EstimationResult:
    matrix: np.ndarray | None
    inliers: np.ndarray
    status: str
    iterations: int
    errors: np.ndarray | None = None



def read_matching_csv(path, columns):
    """ヘッダーを確認してCSVを読み込む。BOM付きUTF-8にも対応する。"""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected CSV header: {path}")
        for row in reader:
            rows.append(row)
    return rows



def load_matching_inputs():
    """入力層: CSVを読み、画像ペアごとの対応座標を用意する。"""
    rows = read_matching_csv(
        INPUT_DIR / "keypoints.csv",
        ["image_name", "keypoint_index", "x", "y"],
    )
    keypoints = {}
    for row in rows:
        key = (row["image_name"], int(row["keypoint_index"]))
        if key in keypoints:
            raise ValueError(f"Duplicate keypoint: {key}")
        point = (float(row["x"]), float(row["y"]))
        if not np.all(np.isfinite(point)):
            raise ValueError(f"Invalid keypoint coordinates: {key}")
        keypoints[key] = point

    rows = read_matching_csv(INPUT_DIR / "pairs.csv", ["image1", "image2", "frame_gap"])
    pairs = []
    pair_matches = {}
    for row in rows:
        pair = (row["image1"], row["image2"])
        if pair in pair_matches:
            raise ValueError(f"Duplicate image pair: {pair}")
        pairs.append((pair, int(row["frame_gap"])))
        pair_matches[pair] = []

    rows = read_matching_csv(
        INPUT_DIR / "matches.csv",
        ["image1", "image2", "keypoint_index1", "keypoint_index2"],
    )
    for row in rows:
        pair = (row["image1"], row["image2"])
        if pair not in pair_matches:
            raise ValueError(f"Image pair missing from pairs.csv: {pair}")
        index1 = int(row["keypoint_index1"])
        index2 = int(row["keypoint_index2"])
        if (pair[0], index1) not in keypoints or (pair[1], index2) not in keypoints:
            raise ValueError(f"Keypoint missing from keypoints.csv: {pair}, {index1}, {index2}")
        pair_matches[pair].append((index1, index2))

    image_pairs = []
    for pair, frame_gap in pairs:
        matches = pair_matches[pair]
        points1 = []
        points2 = []
        for index1, index2 in matches:
            points1.append(keypoints[(pair[0], index1)])
            points2.append(keypoints[(pair[1], index2)])
        points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
        points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
        image_pairs.append(MatchingPair(
            pair[0], pair[1], frame_gap, matches, points1, points2,
        ))
    return image_pairs



def normalize_points(points):
    """重心を原点に移し、原点からの平均距離をsqrt(2)にする。"""
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    if mean_distance < 1e-12:
        return None, None
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ])
    return centered * scale, transform



def fit_fundamental_matrix(points1, points2):
    """8組以上の対応から、正規化8点法で一つのFを求める。"""
    if len(points1) < MIN_MATCHES:
        return None
    normalized1, transform1 = normalize_points(points1)
    normalized2, transform2 = normalize_points(points2)
    if normalized1 is None or normalized2 is None:
        return None

    x1, y1 = normalized1.T
    x2, y2 = normalized2.T
    design = np.column_stack((
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(len(points1)),
    ))
    try:
        # 8x9の行列では、零空間のベクトルを得るため完全なV.Tが必要。
        # 9組以上では縮約SVDを使い、大きなUの生成を避ける。
        _, singular_values, vt = np.linalg.svd(
            design, full_matrices=(len(points1) == MIN_MATCHES),
        )
        if singular_values[7] <= singular_values[0] * 1e-10:
            return None
        matrix = vt[-1].reshape(3, 3)
        u, singular_values, vt = np.linalg.svd(matrix)
        singular_values[-1] = 0.0
        matrix = u @ np.diag(singular_values) @ vt
    except np.linalg.LinAlgError:
        return None

    matrix = transform2.T @ matrix @ transform1
    norm = np.linalg.norm(matrix)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return matrix / norm



def epipolar_errors(matrix, homogeneous1, homogeneous2):
    """対応点とエピポーラ線の距離を、左右それぞれ求める。

    l2 = F x1、l1 = F.T x2 とし、点と直線の距離は
    |a*x + b*y + c| / sqrt(a*a + b*b) で計算する。
    左右の距離の大きい方を誤差とする。単位はピクセル。
    """
    lines2 = homogeneous1 @ matrix.T
    lines1 = homogeneous2 @ matrix
    numerator = np.abs(np.sum(homogeneous2 * lines2, axis=1))
    denominator1 = np.linalg.norm(lines1[:, :2], axis=1)
    denominator2 = np.linalg.norm(lines2[:, :2], axis=1)
    valid = (denominator1 > np.finfo(float).eps) & (denominator2 > np.finfo(float).eps)
    errors = np.full(len(homogeneous1), np.inf)
    errors[valid] = np.maximum(
        numerator[valid] / denominator1[valid],
        numerator[valid] / denominator2[valid],
    )
    return errors



def required_fundamental_iterations(inlier_ratio):
    """K = ceil(log(1-confidence) / log(1-w**8)) で必要反復数を見積もる。

    wはインライア率。独立な抽出を仮定すると、8組すべてが
    インライアである確率はw**8となる。反復回数は推定上の目安。
    """
    clean_sample_probability = inlier_ratio ** MIN_MATCHES
    if clean_sample_probability <= 0.0:
        return F_RANSAC_MAX_ITERATIONS
    if clean_sample_probability >= 1.0:
        return 1
    iterations = np.log1p(-F_RANSAC_CONFIDENCE) / np.log1p(-clean_sample_probability)
    return min(F_RANSAC_MAX_ITERATIONS, int(np.ceil(iterations)))



def estimate_fundamental_matrix(points1, points2):
    """推定層: RANSACとインライアでの再推定を行い、結果を返す。"""
    empty_inliers = np.zeros(len(points1), dtype=bool)
    if len(points1) < MIN_MATCHES:
        return EstimationResult(None, empty_inliers, "not_enough_matches", 0)

    # 重複座標や一直線上の点しかない場合は推定を見送る。
    for points in (points1, points2):
        if len(np.unique(points, axis=0)) < MIN_MATCHES:
            return EstimationResult(None, empty_inliers, "degenerate_points", 0)
        centered = points - points.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            return EstimationResult(None, empty_inliers, "degenerate_points", 0)

    homogeneous1 = np.column_stack((points1, np.ones(len(points1))))
    homogeneous2 = np.column_stack((points2, np.ones(len(points2))))
    rng = np.random.default_rng(F_RANDOM_SEED)
    best_matrix = None
    best_inliers = empty_inliers
    best_count = 0
    best_error = np.inf
    iteration = 0
    iteration_limit = F_RANSAC_MAX_ITERATIONS

    # ここからRANSAC。以下の4段階を繰り返す。
    while iteration < iteration_limit:
        iteration += 1

        # 1. 全対応から8組を無作為抽出する。同じ回の中では重複させない。
        sample_indices = rng.choice(len(points1), size=MIN_MATCHES, replace=False)

        # 2. 選んだ8組だけで、仮の基礎行列を求める。
        candidate_matrix = fit_fundamental_matrix(
            points1[sample_indices], points2[sample_indices],
        )
        if candidate_matrix is None:
            continue

        # 3. 全対応を評価し、しきい値以下の対応をインライアとする。
        errors = epipolar_errors(candidate_matrix, homogeneous1, homogeneous2)
        inliers = errors <= RANSAC_THRESHOLD_PX
        count = int(np.count_nonzero(inliers))
        median_error = float(np.median(errors[inliers])) if count else np.inf

        # 4. インライア数が最多のモデルを保存する。同数なら誤差で選ぶ。
        if count > best_count or (count == best_count and median_error < best_error):
            best_matrix = candidate_matrix
            best_inliers = inliers
            best_count = count
            best_error = median_error

            # 見つかったインライア率から、残りの必要反復数を更新する。
            if count >= MIN_MATCHES:
                required = required_fundamental_iterations(count / len(points1))
                iteration_limit = min(iteration_limit, required)

    # RANSACはここまで。
    if best_matrix is None or best_count < MIN_MATCHES:
        return EstimationResult(None, empty_inliers, "estimation_failed", iteration)

    # 最後に、選んだインライア全体でFを再推定する。
    # 改善した場合のみ更新し、返すFと採否の対応も保つ。
    for _ in range(REFINEMENT_STEPS):
        refined_matrix = fit_fundamental_matrix(points1[best_inliers], points2[best_inliers])
        if refined_matrix is None:
            break
        errors = epipolar_errors(refined_matrix, homogeneous1, homogeneous2)
        inliers = errors <= RANSAC_THRESHOLD_PX
        count = int(np.count_nonzero(inliers))
        median_error = float(np.median(errors[inliers])) if count else np.inf
        if count > best_count or (count == best_count and median_error < best_error):
            best_matrix = refined_matrix
            best_inliers = inliers
            best_count = count
            best_error = median_error
        else:
            break

    # 可視化用の誤差も推定層で計算する。出力層では再計算しない。
    errors = epipolar_errors(best_matrix, homogeneous1, homogeneous2)
    return EstimationResult(best_matrix, best_inliers, "ok", iteration, errors)



def write_csv_files(directory, results):
    """出力層の補助関数: 計算済みの結果をCSVに整形する。"""
    match_columns = ["image1", "image2", "keypoint_index1", "keypoint_index2"]
    matrix_columns = [
        "image1", "image2", "frame_gap", "status",
        "num_matches", "num_inliers", "inlier_ratio",
    ]
    for row_index in range(3):
        for column_index in range(3):
            matrix_columns.append(f"F{row_index + 1}{column_index + 1}")

    match_path = directory / "inlier_matches.csv"
    matrix_path = directory / "fundamental_matrices.csv"
    total_matches = 0
    total_inliers = 0
    successful_pairs = 0

    with match_path.open("w", encoding="utf-8", newline="") as match_stream, \
            matrix_path.open("w", encoding="utf-8", newline="") as matrix_stream:
        match_writer = csv.writer(match_stream)
        matrix_writer = csv.writer(matrix_stream)
        match_writer.writerow(match_columns)
        matrix_writer.writerow(matrix_columns)

        for pair, result in results:
            num_inliers = 0
            matrix_values = [""] * 9
            if result.status == "ok":
                successful_pairs += 1
                matrix_values = result.matrix.ravel().tolist()
                for match, is_inlier in zip(pair.matches, result.inliers):
                    if is_inlier:
                        index1, index2 = match
                        match_writer.writerow([pair.image1, pair.image2, index1, index2])
                        num_inliers += 1

            num_matches = len(pair.matches)
            ratio = num_inliers / num_matches if num_matches else 0.0
            record = [
                pair.image1, pair.image2, pair.frame_gap, result.status,
                num_matches, num_inliers, ratio,
            ]
            record.extend(matrix_values)
            matrix_writer.writerow(record)
            print(
                f"{pair.image1} -> {pair.image2}: "
                f"{num_matches} -> {num_inliers} ({ratio:.1%}), "
                f"{result.status}, RANSAC iterations={result.iterations}"
            )
            total_matches += num_matches
            total_inliers += num_inliers

    print(f"Estimated pairs: {successful_pairs}/{len(results)}")
    print(f"Total matches: {total_matches} -> {total_inliers}")



# === sfm3.pyから再利用（旧保存名fsm3.py、計算関数はそのまま） ===

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



# === sfm4.pyから再利用（入力関数の名前のみ変更） ===

@dataclass
class ExportImage:
    image_id: int
    name: str
    source_path: Path
    quaternion: np.ndarray
    translation: np.ndarray
    points2D: np.ndarray
    point_ids: np.ndarray



@dataclass
class ExportPoint:
    point_id: int
    position: np.ndarray
    color: np.ndarray
    error: float
    track: list



@dataclass
class ExportModel:
    camera_model: str
    width: int
    height: int
    camera_params: np.ndarray
    images: list
    points: list



def validate_model(model):
    """入力層: sfm3.pyのNPZの形状・座標規約・観測の一意性を確認する。"""
    if int(model["schema_version"]) != 2 or str(model["coordinate_convention"]) != "opencv":
        raise ValueError("sfm3.pyが出力した、OpenCV規約のschema_version=2のNPZが必要です。")
    n, m = len(model["points3D"]), len(model["image_names"])
    if n == 0 or m < 2:
        raise ValueError("三次元点があり、二枚以上の画像が登録されたモデルを指定してください。")
    shapes = {"K": (3, 3), "image_size": (2,), "points3D": (n, 3),
              "point_ids": (n,), "image_names": (m,), "rotations": (m, 3, 3),
              "translations": (m, 3), "keypoint_indices": (n, m), "points2D": (n, m, 2)}
    for key, shape in shapes.items():
        if model[key].shape != shape:
            raise ValueError(f"NPZの{key}の形状が不正です: {model[key].shape}")
    for key in ("K", "points3D", "rotations", "translations"):
        if not np.all(np.isfinite(model[key])):
            raise ValueError(f"NPZの{key}に非有限値があります。")
    K = model["K"]
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Kの焦点距離または最終行が不正です。")
    if not np.allclose([K[0, 1], K[1, 0]], 0.0, atol=1e-12):
        raise ValueError("今回のピンホールモデルではKのスキューを0としてください。")
    if not np.issubdtype(model["image_size"].dtype, np.integer) or np.any(model["image_size"] <= 0):
        raise ValueError("画像サイズは正の整数である必要があります。")
    for key in ("point_ids", "keypoint_indices"):
        if not np.issubdtype(model[key].dtype, np.integer):
            raise ValueError(f"{key}は整数である必要があります。")
    if np.any(model["point_ids"] < 0) or len(np.unique(model["point_ids"])) != n:
        raise ValueError("三次元点IDは重複のない0以上の整数である必要があります。")
    if np.any(model["keypoint_indices"] < -1):
        raise ValueError("未観測の特徴点番号には-1を使用してください。")
    observed = model["keypoint_indices"] >= 0
    if np.any(np.count_nonzero(observed, axis=1) < 2):
        raise ValueError("二視点未満の点があります。sfm3.pyの最終フィルタ後のNPZを指定してください。")
    coordinates = model["points2D"][observed]
    width, height = model["image_size"]
    if (not np.all(np.isfinite(coordinates)) or np.any(coordinates < 0)
            or np.any(coordinates[:, 0] >= width) or np.any(coordinates[:, 1] >= height)):
        raise ValueError("観測座標が画像範囲から外れているか、不正な値です。")
    for column, rotation in enumerate(model["rotations"]):
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError(f"画像{column}の回転行列が不正です。")
        indices = model["keypoint_indices"][observed[:, column], column]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("同じ画像の特徴点が、複数の三次元点へ割り当てられています。")



def load_export_inputs(model_path=None, image_dir=IMAGE_DIR):
    """入力層: 最終NPZと、登録済み画像のRGB配列を読み込む。"""
    if model_path is None:
        model_path = MODEL_PATH
        if not model_path.exists() and LEGACY_MODEL_PATH.exists():
            model_path = LEGACY_MODEL_PATH
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"復元結果がありません: {model_path}。先にsfm3.pyを実行してください。")
    if OUTPUT_DIR.resolve() in model_path.parents:
        raise ValueError("入力NPZは、作り直す出力先output/colmapの外に置いてください。")
    with np.load(model_path, allow_pickle=False) as archive:
        model = {}
        for key in archive.files:
            model[key] = archive[key].copy()
    validate_model(model)
    paths, colors, names = [], [], []
    width, height = model["image_size"]
    for stored_name in model["image_names"].tolist():
        name = str(stored_name).replace("\\", "/")
        relative = PurePosixPath(name)
        has_whitespace = False
        for character in name:
            if character.isspace():
                has_whitespace = True
                break
        if (not name or relative.is_absolute() or PureWindowsPath(name).drive
                or ".." in relative.parts or has_whitespace):
            raise ValueError(f"画像名は空白を含まない相対パスにしてください: {name}")
        path = Path(image_dir) / Path(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(f"元画像がありません: {path}。--image-dirで画像フォルダを指定できます。")
        with Image.open(path) as image:
            if image.size != (int(width), int(height)):
                raise ValueError(f"画像サイズがNPZと一致しません: {path} ({image.size})")
            colors.append(np.asarray(image.convert("RGB"), dtype=np.float64))
        paths.append(path.resolve())
        names.append(relative.as_posix())
    if len(set(names)) != len(names):
        raise ValueError("画像名が重複しています。")
    model["image_names"] = np.array(names)
    return model, paths, colors, model_path



def rotation_to_quaternion(rotation):
    """回転行列からHamilton形式の(qw,qx,qy,qz)へ変換する。180度付近にも対応。"""
    R = rotation
    trace = float(np.trace(R))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * scale, (R[2, 1] - R[1, 2]) / scale,
                      (R[0, 2] - R[2, 0]) / scale, (R[1, 0] - R[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(R)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array([(R[2, 1] - R[1, 2]) / scale, 0.25 * scale,
                          (R[0, 1] + R[1, 0]) / scale, (R[0, 2] + R[2, 0]) / scale])
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array([(R[0, 2] - R[2, 0]) / scale, (R[0, 1] + R[1, 0]) / scale,
                          0.25 * scale, (R[1, 2] + R[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array([(R[1, 0] - R[0, 1]) / scale, (R[0, 2] + R[2, 0]) / scale,
                          (R[1, 2] + R[2, 1]) / scale, 0.25 * scale])
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q *= -1.0
    return q



def sample_colors(image, points):
    """元のOpenCV規約の(x,y)でRGBを双一次補間する。出力用の+0.5はまだ加えない。"""
    height, width = image.shape[:2]
    x = np.clip(points[:, 0], 0.0, width - 1.0)
    y = np.clip(points[:, 1], 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1.0 - dx) * (1.0 - dy) * image[y0, x0]
            + dx * (1.0 - dy) * image[y0, x1]
            + (1.0 - dx) * dy * image[y1, x0]
            + dx * dy * image[y1, x1])



def build_colmap_model(model, image_paths, image_colors):
    """変換層: 姿勢・座標規約・ID・色・再投影誤差・トラックを整える。"""
    K = model["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    # NPZのKはOpenCV規約。camera.jsonを読み直さず、主点をCOLMAP規約へ戻す。
    cx, cy = float(K[0, 2] + 0.5), float(K[1, 2] + 0.5)
    if fx == fy:
        camera_model, camera_params = "SIMPLE_PINHOLE", np.array([fx, cx, cy])
    else:
        camera_model, camera_params = "PINHOLE", np.array([fx, fy, cx, cy])
    n = len(model["points3D"])
    point_ids = []
    tracks = []
    for point_id in model["point_ids"]:
        point_ids.append(int(point_id) + 1)
        tracks.append([])
    color_sum = np.zeros((n, 3))
    error_sum = np.zeros(n)
    observation_counts = np.zeros(n, dtype=np.int64)
    images = []
    for column, image_name in enumerate(model["image_names"].tolist()):
        rows = np.flatnonzero(model["keypoint_indices"][:, column] >= 0)
        order = np.argsort(model["keypoint_indices"][rows, column], kind="stable")
        rows = rows[order]
        observed = model["points2D"][rows, column]
        points = model["points3D"][rows]
        rotation, translation = model["rotations"][column], model["translations"][column]
        camera_points = points @ rotation.T + translation
        if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= DEPTH_EPSILON):
            raise ValueError(f"{image_name}の観測にカメラ後方の点があります。")
        homogeneous = camera_points @ K.T
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        errors = np.linalg.norm(projected - observed, axis=1)
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"{image_name}の再投影誤差に非有限値があります。")
        error_sum[rows] += errors
        color_sum[rows] += sample_colors(image_colors[column], observed)
        observation_counts[rows] += 1
        image_id = column + 1
        image_point_ids = []
        for local_index, point_row in enumerate(rows):
            # POINT2D_IDXは元のSIFT番号でなく、この画像で出力する観測の0始まりの番号。
            image_point_ids.append(point_ids[point_row])
            tracks[point_row].append((image_id, local_index))
        images.append(ExportImage(image_id, image_name, image_paths[column],
                                  rotation_to_quaternion(rotation), translation.copy(),
                                  observed + 0.5, np.asarray(image_point_ids, dtype=np.int64)))
    mean_colors = np.clip(np.rint(color_sum / observation_counts[:, None]), 0, 255).astype(np.uint8)
    mean_errors = error_sum / observation_counts
    points = []
    for row, point_id in enumerate(point_ids):
        points.append(ExportPoint(point_id, model["points3D"][row].copy(),
                                  mean_colors[row], float(mean_errors[row]), tracks[row]))
    width, height = model["image_size"]
    return ExportModel(camera_model, int(width), int(height), camera_params, images, points)



def write_cameras(path, model):
    """出力層: cameras.txt。内部パラメータを全画像で共有する。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Camera list with one line of data per camera:\n")
        stream.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n# Number of cameras: 1\n")
        fields = ["1", model.camera_model, str(model.width), str(model.height)]
        for value in model.camera_params:
            fields.append(format(float(value), ".17g"))
        stream.write(" ".join(fields) + "\n")



def write_images(path, model):
    """出力層: images.txt。画像一枚につき必ず二行書く。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Image list with two lines of data per image:\n")
        stream.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        stream.write("# POINTS2D[] as (X Y POINT3D_ID)\n")
        stream.write(f"# Number of images: {len(model.images)}\n")
        for image in model.images:
            fields = [str(image.image_id)]
            for value in image.quaternion:
                fields.append(format(float(value), ".17g"))
            for value in image.translation:
                fields.append(format(float(value), ".17g"))
            fields.extend(["1", image.name])
            stream.write(" ".join(fields) + "\n")
            fields = []
            for point, point_id in zip(image.points2D, image.point_ids):
                fields.extend([format(float(point[0]), ".17g"),
                               format(float(point[1]), ".17g"), str(int(point_id))])
            stream.write(" ".join(fields) + "\n")



def write_points3D(path, model):
    """出力層: points3D.txt。座標、RGB、平均誤差、画像への逆参照を保存する。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# 3D point list with one line of data per point:\n")
        stream.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)\n")
        stream.write(f"# Number of points: {len(model.points)}\n")
        for point in model.points:
            fields = [str(point.point_id)]
            for value in point.position:
                fields.append(format(float(value), ".17g"))
            for value in point.color:
                fields.append(str(int(value)))
            fields.append(format(point.error, ".17g"))
            for image_id, point2D_index in point.track:
                fields.extend([str(image_id), str(point2D_index)])
            stream.write(" ".join(fields) + "\n")



def write_results(model):
    """出力層: 一時フォルダで完成させてから、専用の出力先を入れ替える。"""
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
        raise ValueError(f"出力先がディレクトリではありません: {OUTPUT_DIR}")
    with TemporaryDirectory(prefix="sfm4_export_", dir=OUTPUT_DIR.parent) as temporary:
        temporary = Path(temporary)
        staging = temporary / "new"
        image_dir, sparse_dir = staging / "images", staging / "sparse" / "0"
        image_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        for image in model.images:
            destination = image_dir / Path(image.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image.source_path, destination)
        write_cameras(sparse_dir / "cameras.txt", model)
        write_images(sparse_dir / "images.txt", model)
        write_points3D(sparse_dir / "points3D.txt", model)
        # 古いBINやPLYを残さず、画像と三ファイルを一組として更新する。
        backup = temporary / "previous"
        if OUTPUT_DIR.exists():
            OUTPUT_DIR.replace(backup)
        try:
            staging.replace(OUTPUT_DIR)
        except OSError:
            if backup.exists():
                backup.replace(OUTPUT_DIR)
            raise



# === sfm5で追加: 進捗、動画入力、多画像制御、計算量の削減 ===

class Progress:
    """長いNumPy/OpenCVの呼び出し中も、経過時間と処理名を表示する。"""
    def __init__(self):
        self.started = time.monotonic()
        self.last_output = self.started
        self.task = "準備"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop_event.set()
        self.thread.join(timeout=1)

    def say(self, message):
        with self.lock:
            now = time.monotonic()
            self.last_output = now
            print(f"[経過 {now - self.started:8.1f}秒] {message}", flush=True)

    def stage(self, message):
        self.task = message
        self.say(message)

    def count(self, label, done, total, detail=""):
        self.task = label
        self.say(f"{label} {done}/{total} ({done / max(total, 1):.1%}) {detail}")

    def heartbeat(self):
        while not self.stop_event.wait(15):
            if time.monotonic() - self.last_output >= 14:
                self.say(f"処理継続中: {self.task}")


def json_value(value):
    """状態保存用: NumPyの値や非有限値を、通常のJSONの値へ直す。"""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[str(key)] = json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.append(json_value(item))
        return result
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_state(state):
    temporary = STATE_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_value(state), stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(STATE_PATH)


def save_checkpoint(model, state):
    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **model)
    temporary.replace(CHECKPOINT_PATH)
    state["has_checkpoint"] = True
    state["registered_images"] = model["image_names"].tolist()
    state["num_points"] = len(model["points3D"])
    save_state(state)


def read_checkpoint():
    model = {}
    with np.load(CHECKPOINT_PATH, allow_pickle=False) as archive:
        for name in archive.files:
            model[name] = archive[name].copy()
    validate_model(model)
    return model


def run_signature(args):
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"入力動画がありません: {video}")
    if (FRAME_DIR.resolve() in video.parents or WORK_DIR.resolve() in video.parents
            or OUTPUT_DIR.resolve() in video.parents):
        raise ValueError("入力動画はwork/frames・work/sfm5・最終出力先の外に置いてください。")
    stat = video.stat()
    return {"pipeline_version": PIPELINE_VERSION, "video": str(video),
            "video_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
            "camera_sha256": hashlib.sha256(CAMERA_PATH.read_bytes()).hexdigest(),
            "fps": args.fps, "max_frames": args.max_frames,
            "max_features": MAX_FEATURES, "ratio_threshold": RATIO_THRESHOLD,
            "ba_every": args.ba_every, "ba_iterations": args.ba_iterations}


def frame_signature(frames):
    rows = []
    for frame in frames:
        stat = frame.stat()
        rows.append([frame.name, stat.st_size, stat.st_mtime_ns])
    return rows


def replace_directory(staging, destination):
    """専用ディレクトリを完成後に入れ替え、失敗時は元に戻す。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sfm5_replace_", dir=destination.parent) as temporary:
        backup = Path(temporary) / "previous"
        if destination.exists():
            destination.replace(backup)
        try:
            staging.replace(destination)
        except OSError:
            if backup.exists():
                backup.replace(destination)
            raise


def extract_frames(args, camera, progress):
    """入力層: FFmpegの-progressを読みながら、CPUでPNGを抽出する。"""
    executable = shutil.which(args.ffmpeg)
    if executable is None:
        raise FileNotFoundError("ffmpegが見つかりません。インストールしてPATHを通してください。")
    width, height = int(camera["width"]), int(camera["height"])
    FRAME_DIR.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sfm5_frames_", dir=FRAME_DIR.parent) as temporary:
        temporary = Path(temporary)
        staging = temporary / "frames"
        staging.mkdir()
        command = [executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                   "-i", str(args.video.resolve()), "-map", "0:v:0", "-vf",
                   f"fps={args.fps},scale={width}:{height}:force_original_aspect_ratio=decrease,setsar=1",
                   "-an", "-sn", "-dn"]
        if args.max_frames is not None:
            command.extend(["-frames:v", str(args.max_frames)])
        command.extend(["-progress", "pipe:1", "-stats_period", "1",
                        str(staging / "frame_%06d.png")])
        with (temporary / "ffmpeg_error.txt").open("w+", encoding="utf-8") as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                       text=True, encoding="utf-8", errors="replace")
            count, movie_time = "0", "00:00:00"
            try:
                for line in process.stdout:
                    key, separator, value = line.strip().partition("=")
                    if not separator:
                        continue
                    if key == "frame":
                        count = value
                    elif key == "out_time":
                        movie_time = value
                    elif key == "progress":
                        progress.say(f"画像抽出: {count}枚、出力時刻 {movie_time}")
                return_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
            if return_code != 0:
                errors.seek(0)
                raise RuntimeError("FFmpegの画像抽出に失敗しました:\n" + errors.read()[-4000:])
        frames = select_frames(staging, 1, None)
        with Image.open(frames[0]) as image:
            if image.size != (width, height):
                raise ValueError(f"動画の縦横比がcamera.jsonと一致しません。抽出サイズ{image.size}、"
                                 f"指定サイズ{(width, height)}。動画に合うcamera.jsonを用意してください。")
        replace_directory(staging, FRAME_DIR)
    return select_frames(FRAME_DIR, 1, None)


def run_matching(frames, progress):
    pairs = select_pairs(frames)
    progress.say(f"画像{len(frames)}枚、比較ペア{len(pairs)}組。近傍1〜5枚と間隔10,20,40,…")
    with TemporaryDirectory(prefix="matching_", dir=WORK_DIR) as temporary:
        staging = Path(temporary) / "csv"
        staging.mkdir()
        write_pairs(staging / "pairs.csv", pairs)
        match_features(frames, pairs, staging, None, progress.count)
        replace_directory(staging, MATCHING_DIR)


def run_fundamental_ransac(progress):
    pairs = load_matching_inputs()
    results = []
    for number, pair in enumerate(pairs, start=1):
        progress.task = f"FのRANSAC {number}/{len(pairs)}: {pair.image1} → {pair.image2}"
        result = estimate_fundamental_matrix(pair.points1, pair.points2)
        count = int(np.count_nonzero(result.inliers))
        progress.count("FのRANSAC", number, len(pairs),
                       f"{pair.image1} → {pair.image2}: {len(pair.points1)} → {count}組、"
                       f"反復{result.iterations}、{result.status}")
        results.append((pair, result))
    with TemporaryDirectory(prefix="ransac_", dir=WORK_DIR) as temporary:
        staging = Path(temporary) / "csv"
        staging.mkdir()
        write_csv_files(staging, results)
        replace_directory(staging, RANSAC_DIR)


def initialize_many_pairs(pairs, K, progress):
    """採用点数・角度という従来の基準を保ち、上限で不要な評価を省く。"""
    indexed = list(enumerate(pairs))
    indexed.sort(key=lambda item: (-len(item[1].points1), item[0]))
    selected, best_score, best_index = None, (-1, -1.0), len(pairs)
    summaries = []
    for number, (original_index, pair) in enumerate(indexed, start=1):
        summary = {"image1": pair.image1, "image2": pair.image2,
                   "num_inliers": len(pair.points1), "num_in_front": 0,
                   "num_points3D": 0, "median_angle_deg": "",
                   "median_reprojection_error_px": "", "status": "failed"}
        if len(pair.points1) < best_score[0]:
            summary["status"] = "skipped_by_point_count_upper_bound"
            summaries.append(summary)
            continue
        progress.task = f"初期ペア評価 {number}/{len(pairs)}"
        try:
            result = reconstruct_pair(pair, K)
            count = int(np.count_nonzero(result.valid))
            summary["num_in_front"] = int(np.count_nonzero(result.positive_depth))
            summary["num_points3D"] = count
            if count:
                summary["median_angle_deg"] = float(np.median(result.angles[result.valid]))
                summary["median_reprojection_error_px"] = float(np.median(np.max(result.errors[result.valid], axis=1)))
            if count >= MIN_INITIAL_POINTS:
                summary["status"] = "ok"
                score = (count, summary["median_angle_deg"])
                if score > best_score or (score == best_score and original_index < best_index):
                    selected, best_score, best_index = result, score, original_index
            else:
                summary["status"] = "not_enough_stable_points"
        except (ValueError, np.linalg.LinAlgError):
            summary["status"] = "failed"
        summaries.append(summary)
        angle_text = "算出不可" if summary["median_angle_deg"] == "" else f"{summary['median_angle_deg']:.4f}度"
        progress.count("初期ペア", number, len(pairs),
                       f"{pair.image1} → {pair.image2}: インライア{len(pair.points1)}組、"
                       f"前方{summary['num_in_front']}点、採用{summary['num_points3D']}点、"
                       f"角度中央値={angle_text}、{summary['status']}")
    if selected is None:
        raise ValueError("初期化できるペアがありません。視差・ぼけ・内部パラメータを確認してください。")
    progress.say(f"初期ペア決定: {selected.pair.image1} / {selected.pair.image2}、"
                 f"採用{best_score[0]}点、角度中央値{best_score[1]:.4f}度")
    return selected, summaries


def maximum_viewing_angle(point, centers):
    """従来の全視線ペア評価と同じ値を、行列演算でまとめて求める。"""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    rays = np.asarray(point) - centers
    lengths = np.linalg.norm(rays, axis=1)
    nonzero = np.isfinite(lengths) & (lengths > 0.0)
    rays, lengths = rays[nonzero], lengths[nonzero]
    if len(rays) < 2:
        return 0.0
    rays = rays / lengths[:, None]
    dots = np.abs(rays @ rays.T)
    first, second = np.triu_indices(len(rays), 1)
    valid = lengths[first] * lengths[second] > DEPTH_EPSILON
    values = dots[first[valid], second[valid]]
    if len(values) == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.min(values), 0.0, 1.0))))


def triangulate_many_views(model, keypoints, matches, added_image, progress):
    """長いトラックだけ仮説数を制限し、採否は全観測で確認する。"""
    if len(model["image_names"]) <= 3:
        return triangulate_new_points(model, keypoints, matches, added_image)
    image_columns, known_features = {}, set()
    for column, image in enumerate(model["image_names"].tolist()):
        image_columns[image] = column
        for index in model["keypoint_indices"][:, column]:
            if index >= 0:
                known_features.add((image, int(index)))
    tracks = feature_tracks(matches, image_columns)
    m = len(image_columns)
    new_points, new_indices, new_pixels, new_errors, new_angles, summaries = [], [], [], [], [], []
    next_id = int(np.max(model["point_ids"])) + 1 if len(model["point_ids"]) else 0
    rng = np.random.default_rng(RANDOM_SEED)
    for track_id, track in enumerate(tracks):
        if track_id % 1000 == 0:
            progress.count("三角測量トラック", track_id, len(tracks), f"新規{len(new_points)}点")
        images, contains_existing = set(), False
        for feature in track:
            images.add(feature[0])
            contains_existing = contains_existing or feature in known_features
        if added_image not in images or contains_existing or len(images) != len(track):
            continue
        columns, pixels, new_view_row = [], [], 0
        for row, (image, index) in enumerate(track):
            columns.append(image_columns[image])
            pixels.append(keypoints[(image, index)])
            if image == added_image:
                new_view_row = row
        pixels = np.asarray(pixels)
        centers = model["camera_centers"][columns]
        combinations = []
        for first in range(len(track)):
            for second in range(first + 1, len(track)):
                baseline = float(np.linalg.norm(centers[first] - centers[second]))
                combinations.append((first, second, baseline))
        if len(combinations) > MAX_TRIANGULATION_HYPOTHESES:
            combinations.sort(key=lambda item: -item[2])
            chosen = combinations[:8]
            random_rows = rng.choice(len(combinations) - 8,
                                     size=MAX_TRIANGULATION_HYPOTHESES - 8, replace=False) + 8
            for row in random_rows:
                chosen.append(combinations[int(row)])
            combinations = chosen
        rotations = model["rotations"][columns]
        translations = model["translations"][columns]
        best, best_score = None, (-1, -np.inf)
        for first, second, unused_baseline in combinations:
            point = triangulate_world_point(model, columns, pixels, first, second)
            if not np.all(np.isfinite(point)):
                continue
            camera_points = rotations @ point + translations
            projected_h = camera_points @ model["K"].T
            with np.errstate(divide="ignore", invalid="ignore"):
                errors = np.linalg.norm(projected_h[:, :2] / projected_h[:, 2:3] - pixels, axis=1)
            errors[(camera_points[:, 2] <= DEPTH_EPSILON) | ~np.isfinite(errors)] = np.inf
            inliers = errors <= MAX_REPROJECTION_ERROR_PX
            count = int(np.count_nonzero(inliers))
            if count < 2 or not inliers[new_view_row]:
                continue
            angle = maximum_viewing_angle(point, centers[inliers])
            if angle < MIN_TRIANGULATION_ANGLE_DEG:
                continue
            score = (count, -float(np.median(errors[inliers])))
            if score > best_score:
                best_score, best = score, (point, errors, inliers, angle)
        if best is None:
            continue
        point, errors, inliers, angle = best
        indices = np.full(m, -1, dtype=np.int64)
        observations, point_errors = np.full((m, 2), np.nan), np.full(m, np.nan)
        for row in np.flatnonzero(inliers):
            column = columns[row]
            indices[column], observations[column], point_errors[column] = track[row][1], pixels[row], errors[row]
        summaries.append({"track_id": track_id, "status": "added", "point_id": next_id + len(new_points),
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
        updated["point_ids"] = np.concatenate((model["point_ids"], np.arange(next_id, next_id + len(new_points))))
        updated["keypoint_indices"] = np.vstack((model["keypoint_indices"], new_indices))
        updated["points2D"] = np.concatenate((model["points2D"], new_pixels), axis=0)
        updated["reprojection_errors"] = np.vstack((model["reprojection_errors"], new_errors))
        updated["triangulation_angles"] = np.concatenate((model["triangulation_angles"], new_angles))
    updated["stage"] = np.array("triangulated_before_ba")
    progress.say(f"【4】三角測量: 既存{len(model['points3D'])}点 + 新規{len(new_points)}点")
    return updated, summaries


def skew_batch(vectors):
    """[v]_x w = v × wとなる行列を、観測数だけ作る。"""
    result = np.zeros((len(vectors), 3, 3))
    result[:, 0, 1], result[:, 0, 2] = -vectors[:, 2], vectors[:, 1]
    result[:, 1, 0], result[:, 1, 2] = vectors[:, 2], -vectors[:, 0]
    result[:, 2, 0], result[:, 2, 1] = -vectors[:, 1], vectors[:, 0]
    return result


def schur_normal_blocks(model, problem):
    """J.T J = [[A,B],[B.T,C]]、J.T eps = [g_c,g_X]を直接集める。

    Cは点ごとの3x3ブロック対角。Bは観測のあるカメラとの交差項だけ保持する。
    D=d pi(Y)/dY、J_X=D R、J_R=-D [R X]_x、J_t=D（カメラ2はD basis）。
    ここで回転の局所更新はexp(delta_omega) R、並進は独立に更新する。
    """
    nc, n = problem.point_offset, len(model["points3D"])
    A, gc = np.zeros((nc, nc)), np.zeros(nc)
    C, gp = np.zeros((n, 3, 3)), np.zeros((n, 3))
    entries = []
    for unused in range(n):
        entries.append([])
    blocks = {}
    for column, rotation_slice, translation_slice in problem.camera_blocks:
        blocks[column] = (rotation_slice, translation_slice)
    for column in range(len(model["image_names"])):
        observed = problem.image_columns == column
        rows = problem.point_rows[observed]
        if len(rows) == 0:
            continue
        R, t = model["rotations"][column], model["translations"][column]
        rotated = model["points3D"][rows] @ R.T
        Y = rotated + t
        z = Y[:, 2]
        fx, fy = model["K"][0, 0], model["K"][1, 1]
        D = np.zeros((len(rows), 2, 3))
        D[:, 0, 0], D[:, 1, 1] = fx / z, fy / z
        D[:, 0, 2], D[:, 1, 2] = -fx * Y[:, 0] / z ** 2, -fy * Y[:, 1] / z ** 2
        Jx = D @ R
        prediction = project_points(model["points3D"][rows], model["K"], R, t)
        residual = problem.measured[observed] - prediction
        np.add.at(C, rows, np.einsum("nki,nkj->nij", Jx, Jx))
        np.add.at(gp, rows, np.einsum("nki,nk->ni", Jx, residual))
        if column == 0:
            continue
        rotation_slice, translation_slice = blocks[column]
        camera_slice = slice(rotation_slice.start, translation_slice.stop)
        Jr = -D @ skew_batch(rotated)
        Jt = D @ translation_tangent_basis(t) if column == 1 else D
        Jc = np.concatenate((Jr, Jt), axis=2)
        A[camera_slice, camera_slice] += np.einsum("nki,nkj->ij", Jc, Jc)
        gc[camera_slice] += np.einsum("nki,nk->i", Jc, residual)
        cross = np.einsum("nki,nkj->nij", Jc, Jx)
        indices = np.arange(camera_slice.start, camera_slice.stop)
        for index, point_row in enumerate(rows):
            entries[int(point_row)].append((indices, cross[index]))
    cross_blocks = []
    for point_entries in entries:
        if not point_entries:
            cross_blocks.append((np.empty(0, dtype=np.int64), np.empty((0, 3))))
            continue
        indices, matrices = [], []
        for index, matrix in point_entries:
            indices.extend(index.tolist())
            matrices.append(matrix)
        cross_blocks.append((np.array(indices, dtype=np.int64), np.vstack(matrices)))
    return A, C, cross_blocks, gc, gp


def solve_schur_step(problem, normal, damping):
    """LMの同じ正規方程式から点を消去し、カメラ→点の順に解く。

    C_l=C+lambda I, A_l=A+lambda I
    S=A_l-B C_l^-1 B.T
    S delta_c=g_c-B C_l^-1 g_X
    delta_X=C_l^-1 (g_X-B.T delta_c)
    C_l^-1は各3x3連立方程式として求め、全体の逆行列は作らない。
    """
    A, C, cross_blocks, gc, gp = normal
    eye3 = np.broadcast_to(np.eye(3), C.shape)
    inverse_blocks = np.linalg.solve(C + damping * eye3, eye3)
    reduced = A.copy()
    reduced.flat[::len(reduced) + 1] += damping
    rhs = gc.copy()
    for row, (indices, B) in enumerate(cross_blocks):
        if len(indices):
            weighted = B @ inverse_blocks[row]
            reduced[np.ix_(indices, indices)] -= weighted @ B.T
            rhs[indices] -= weighted @ gp[row]
    delta = np.empty(problem.num_parameters)
    delta[:problem.point_offset] = np.linalg.solve(reduced, rhs)
    for row, (indices, B) in enumerate(cross_blocks):
        local_rhs = gp[row] - B.T @ delta[indices]
        offset = problem.point_offset + 3 * row
        delta[offset:offset + 3] = inverse_blocks[row] @ local_rhs
    return delta


def bundle_adjustment_schur(model, progress):
    """大規模版【5】。残差・制約・lambda更新は従来BAと同じ。"""
    problem = make_ba_problem(model)
    projected, positive = ba_projection(model, problem)
    if not positive or not np.all(np.isfinite(projected)):
        raise ValueError("BA開始時に後方点または不正な再投影があります。")
    measured = problem.measured.reshape(-1)
    epsilon = measured - projected
    initial_error = error = float(epsilon @ epsilon)
    current, history, damping, accepted_steps = model, [], None, 0
    status = "max_iterations"
    progress.say(f"【5】Schur BA: カメラ{len(model['image_names'])}台、点{len(model['points3D'])}、"
                 f"観測{len(problem.measured)}組、変数{problem.num_parameters}、"
                 f"縮約後{problem.point_offset}変数、E={error:.6f}")
    for iteration in range(1, BA_MAX_ITERATIONS + 1):
        progress.task = f"BA {iteration}/{BA_MAX_ITERATIONS}: ヤコビ行列と正規方程式のブロックを集計"
        normal = schur_normal_blocks(current, problem)
        A, C, unused_cross, gc, gp = normal
        gradient = max(float(np.max(np.abs(gc))), float(np.max(np.abs(gp))))
        if gradient < BA_GRADIENT_TOLERANCE:
            status = "gradient_converged"
            break
        if damping is None:
            largest_diagonal = max(float(np.max(np.diag(A))),
                                   float(np.max(np.diagonal(C, axis1=1, axis2=2))), 1.0)
            damping = BA_INITIAL_LAMBDA_FACTOR * largest_diagonal
        accepted = False
        for trial in range(1, BA_MAX_TRIALS + 1):
            progress.task = f"BA {iteration}/{BA_MAX_ITERATIONS}, 試行{trial}: Schur補完と求解"
            try:
                delta = solve_schur_step(problem, normal, damping)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            proposed = apply_ba_update(current, delta, problem)
            prediction, positive = ba_projection(proposed, problem)
            new_epsilon = measured - prediction
            new_error = float(new_epsilon @ new_epsilon)
            accepted = bool(positive and np.all(np.isfinite(delta)) and np.isfinite(new_error) and new_error < error)
            step_norm = float(np.linalg.norm(delta))
            history.append({"iteration": iteration, "trial": trial, "lambda": damping,
                            "error_before": error, "error_trial": new_error,
                            "step_norm": step_norm, "gradient_inf": gradient, "accepted": int(accepted)})
            decision = "採用" if accepted else "棄却"
            progress.say(f"  LM {iteration:02d}.{trial:02d}: E={error:.6f} → {new_error:.6f}、"
                         f"lambda={damping:.3e}、||delta||={step_norm:.3e}、{decision}")
            if accepted:
                relative_decrease = (error - new_error) / max(error, 1e-15)
                current, epsilon, error = proposed, new_epsilon, new_error
                accepted_steps += 1
                damping = max(damping / 3.0, 1e-15)
                if relative_decrease < BA_RELATIVE_ERROR_TOLERANCE:
                    status = "error_converged"
                elif step_norm < BA_STEP_TOLERANCE:
                    status = "step_converged"
                break
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
             "rms_after_px": float(np.sqrt(error / len(problem.measured))), "fixed_baseline": problem.baseline}
    progress.say(f"【5】BA完了: 距離RMS {stats['rms_before_px']:.6f} → {stats['rms_after_px']:.6f} px、{status}")
    return current, history, stats


def refine_model(model, state, progress):
    progress.task = "【5】バンドル調整"
    problem = make_ba_problem(model)
    if problem.num_parameters <= DENSE_BA_PARAMETER_LIMIT:
        optimized, history, stats = bundle_adjustment(model)
        solver = "dense_numerical"
    else:
        optimized, history, stats = bundle_adjustment_schur(model, progress)
        solver = "schur_analytic"
    progress.task = "【6】フィルタリング"
    filtered, filtering = filter_model(optimized)
    if len(filtered["points3D"]) < MIN_INITIAL_POINTS:
        raise ValueError("BA・フィルタリング後の点が不足しました。camera.jsonや動画の視差を確認してください。")
    batch = len(state["ba_runs"]) + 1
    state["ba_runs"].append({"batch": batch, "num_images": len(model["image_names"]),
                              "solver": solver, "ba": stats, "filtering": filtering})
    for row in history:
        state["ba_history"].append(dict(batch=batch, num_images=len(model["image_names"]), solver=solver, **row))
    state["last_ba_count"] = len(model["image_names"])
    save_checkpoint(filtered, state)
    write_summary_csv(WORK_DIR / "ba_iterations.csv", state["ba_history"])
    return filtered


def write_image_status(frames, model, state, candidates):
    """出力層: 成否と、最後に確認できた対応数・PnP判定だけを書き出す。"""
    registered = set(model["image_names"].tolist())
    final_candidates, last_attempts = {}, {}
    for candidate in candidates:
        final_candidates[candidate.image_name] = candidate
    for attempt in state["registration_attempts"]:
        if attempt["status"] != "not_attempted":
            last_attempts[attempt["image_name"]] = attempt
    rows = []
    for frame in frames:
        row = {"image_name": frame.name, "status": "registered", "num_2d_3d": "", "reason": ""}
        if frame.name not in registered:
            row["status"] = "unregistered"
            candidate = final_candidates.get(frame.name)
            count = len(candidate.point_rows) if candidate is not None else 0
            row["num_2d_3d"] = count
            if count < MIN_PNP_INLIERS:
                row["reason"] = f"2D-3D対応が{MIN_PNP_INLIERS}組未満（{count}組）"
            else:
                row["reason"] = last_attempts.get(frame.name, {}).get("reason", "PnP未成功")
        rows.append(row)
    write_summary_csv(WORK_DIR / "image_status.csv", rows)
    write_summary_csv(WORK_DIR / "registration_attempts.csv", state["registration_attempts"])


def reconstruct_all(frames, state, args, progress):
    camera, K, pairs, keypoints, matches = load_inputs()
    if state["has_checkpoint"]:
        model = read_checkpoint()
        progress.say(f"再開: 登録{len(model['image_names'])}/{len(frames)}枚、点{len(model['points3D'])}")
    else:
        progress.stage("工程4/6: 二画像から初期化")
        initial, summaries = initialize_many_pairs(pairs, K, progress)
        write_summary_csv(WORK_DIR / "initialization_candidates.csv", summaries)
        model = make_initial_model(camera, initial)
        save_checkpoint(model, state)
    progress.stage("工程5/6: 全画像の登録・三角測量・BA・フィルタリング")
    if not state["sfm_complete"]:
        # 画像追加直後・BA開始前に中断した場合は、そのBAから再開する。
        current_count = len(model["image_names"])
        if (state["last_ba_count"] != current_count
                and (current_count == 3 or current_count - max(state["last_ba_count"], 2) >= args.ba_every)):
            model = refine_model(model, state, progress)
        # 同じモデルでの再試行を無限に繰り返さないためのフラグ。
        retriangulated_at_count = -1
        while len(model["image_names"]) < len(frames):
            progress.task = "【1】2D-3D対応の作成"
            candidates = build_candidates(model, keypoints, matches)
            progress.say(f"【1〜3】現在 {len(model['image_names'])}/{len(frames)}枚、"
                         f"{len(model['points3D'])}点。次の画像を選択")
            candidate, pose, summaries = register_next_image(model, candidates)
            for row in summaries:
                state["registration_attempts"].append(dict(num_registered=len(model["image_names"]), **row))
            if pose is None or pose.reason != "ok":
                count = len(model["image_names"])
                if state["last_ba_count"] != count:
                    progress.say("登録が停止したため、BA・フィルタリング後に再試行します。")
                    model = refine_model(model, state, progress)
                    continue
                if retriangulated_at_count != count:
                    retriangulated_at_count = count
                    progress.task = "BA後の再三角測量"
                    retriangulated, unused = triangulate_many_views(
                        model, keypoints, matches, str(model["image_names"][-1]), progress)
                    if len(retriangulated["points3D"]) > len(model["points3D"]):
                        model = refresh_model_geometry(retriangulated)
                        state["last_ba_count"] = -1
                        save_checkpoint(model, state)
                        continue
                progress.say("現モデルから登録できる画像がなくなりました。残りはimage_status.csvへ記録します。")
                break
            progress.task = "【3】画像の登録"
            model = update_model(model, candidate, pose)
            progress.count("画像登録", len(model["image_names"]), len(frames), candidate.image_name)
            progress.task = "【4】新しい三次元点の三角測量"
            model, unused = triangulate_many_views(model, keypoints, matches, candidate.image_name, progress)
            model = refresh_model_geometry(model)
            # 新しい点・観測が得られた時点で保存。BA中の中断でもここから再開できる。
            save_checkpoint(model, state)
            count = len(model["image_names"])
            if count == 3 or count - max(state["last_ba_count"], 2) >= args.ba_every:
                model = refine_model(model, state, progress)
        if state["last_ba_count"] != len(model["image_names"]):
            model = refine_model(model, state, progress)
        state["sfm_complete"] = True
        save_checkpoint(model, state)
    candidates = build_candidates(model, keypoints, matches)
    write_image_status(frames, model, state, candidates)
    return camera, model


def parse_arguments():
    parser = argparse.ArgumentParser(description="動画から全画像SfMを行い、3DGS用COLMAPモデルを保存")
    parser.add_argument("video", nargs="?", type=Path, default=VIDEO_PATH,
                        help="入力動画。省略時はスクリプトの隣のinput/input.mov")
    parser.add_argument("--fps", type=float, default=2.0, help="抽出する毎秒の枚数（標準2）")
    parser.add_argument("--max-frames", type=int, help="動作確認用の抽出枚数上限。省略時は全動画")
    parser.add_argument("--ba-every", type=int, default=5, help="BAの間に追加する画像枚数（標準5）")
    parser.add_argument("--ba-iterations", type=int, default=20, help="一回のBAのLM反復上限（標準20）")
    parser.add_argument("--resume", action="store_true", help="同じ入力・設定の最後の保存から再開")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpegの実行ファイル名またはパス")
    parser.add_argument("--visualize", action="store_true", help="最終点群とカメラ配置を描画（matplotlib）")
    args = parser.parse_args()
    if (not math.isfinite(args.fps) or args.fps <= 0 or args.ba_every < 1
            or args.ba_iterations < 1 or (args.max_frames is not None and args.max_frames < 2)):
        parser.error("fpsは正の有限値、枚数上限は2以上、BAの間隔と反復上限は1以上にしてください。")
    return args


def main():
    global BA_MAX_ITERATIONS
    args = parse_arguments()
    BA_MAX_ITERATIONS = args.ba_iterations
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    try:
        with Progress() as progress:
            camera, unused_K = load_camera()
            width, height = camera["width"], camera["height"]
            if int(width) != width or int(height) != height:
                raise ValueError("camera.jsonのwidthとheightは整数にしてください。")
            signature = run_signature(args)
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            if args.resume:
                if not STATE_PATH.is_file():
                    raise ValueError("再開用run.jsonがありません。--resumeを外して開始してください。")
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if state["signature"] != signature:
                    raise ValueError("動画・camera.json・設定が前回と異なります。--resumeを外して再実行してください。")
            else:
                state = {"signature": signature, "frames_complete": False, "matching_complete": False,
                         "ransac_complete": False, "has_checkpoint": False, "sfm_complete": False,
                         "export_complete": False, "last_ba_count": 0, "ba_runs": [],
                         "ba_history": [], "registration_attempts": []}
                save_state(state)
            progress.say(f"入力: {args.video.resolve()}、{args.fps:g} fps、画像サイズ{width}x{height}")
            progress.say(f"フレーム: {FRAME_DIR}\n最終出力: {OUTPUT_DIR}")
            progress.stage("工程1/6: 動画から画像抽出")
            if state["frames_complete"]:
                frames = select_frames(FRAME_DIR, 1, None)
                if frame_signature(frames) != state["frame_signature"]:
                    raise ValueError("抽出画像が変更されています。--resumeを外して再実行してください。")
                progress.say(f"抽出済み{len(frames)}枚を再利用")
            else:
                frames = extract_frames(args, camera, progress)
                state["frames_complete"] = True
                state["frame_signature"] = frame_signature(frames)
                save_state(state)
            progress.stage("工程2/6: SIFTとシーケンシャルマッチング")
            if not state["matching_complete"]:
                run_matching(frames, progress)
                state["matching_complete"] = True
                save_state(state)
            else:
                progress.say("保存済みの特徴点・対応CSVを再利用")
            progress.stage("工程3/6: Fの推定とRANSACによる幾何検証")
            if not state["ransac_complete"]:
                run_fundamental_ransac(progress)
                state["ransac_complete"] = True
                save_state(state)
            else:
                progress.say("保存済みのF・インライアCSVを再利用")
            camera, model = reconstruct_all(frames, state, args, progress)
            progress.stage("工程6/6: COLMAP形式へ変換・保存")
            final, image_paths, colors, unused_source = load_export_inputs(CHECKPOINT_PATH, FRAME_DIR)
            exported = build_colmap_model(final, image_paths, colors)
            write_results(exported)
            state["export_complete"] = True
            save_state(state)
            if args.visualize:
                progress.stage("最終点群とカメラ配置の可視化")
                write_visualization(camera, model)
            registered, total = len(model["image_names"]), len(frames)
            progress.say(f"完了: 登録画像{registered}/{total}枚、未登録{total - registered}枚、"
                         f"三次元点{len(model['points3D'])}点")
            progress.say(f"COLMAP: {OUTPUT_DIR}\n途中結果・未登録理由: {WORK_DIR}")
    except KeyboardInterrupt:
        print("\n中断しました。同じ引数に--resumeを付けると最後の保存から再開できます。", flush=True)
        raise SystemExit(130)
    except (OSError, KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"エラー: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
