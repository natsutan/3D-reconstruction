"""CSVの対応点から、RANSACで画像ペアごとの基礎行列を推定する。

配置: COLMAP/ransac_matches.py
実行: リポジトリのrootで python COLMAP/ransac_matches.py
依存: NumPy、OpenCV（opencv-python または opencv-python-headless）

入力: このファイルと同じディレクトリの output/matching_first にある
      keypoints.csv、matches.csv、pairs.csv
出力: output/ransac/inlier_matches.csv
      output/ransac/fundamental_matrices.csv

座標はkeypoints.csvに記録されたOpenCVのピクセル座標をそのまま使う。
基礎行列の向きは x2.T @ F @ x1 = 0。x1、x2は同次画像座標。
この段階では画像、SIFT記述子、camera.jsonは読み込まない。
"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output" / "matching_first"
OUTPUT_DIR = BASE_DIR / "output" / "ransac"

# エピポーラ線からの距離を判定するしきい値。単位はピクセル。
RANSAC_THRESHOLD_PX = 1.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERATIONS = 10000
RANDOM_SEED = 0
MIN_MATCHES = 8


def read_csv(path, columns):
    """ヘッダーを確認してCSVを読み込む。BOM付きUTF-8にも対応する。"""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected CSV header: {path}")
        for row in reader:
            rows.append(row)
    return rows


def load_inputs():
    """特徴点番号で座標を引けるようにし、対応を画像ペアごとにまとめる。"""
    rows = read_csv(
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

    rows = read_csv(INPUT_DIR / "pairs.csv", ["image1", "image2", "frame_gap"])
    pairs = []
    pair_matches = {}
    for row in rows:
        pair = (row["image1"], row["image2"])
        if pair in pair_matches:
            raise ValueError(f"Duplicate image pair: {pair}")
        pairs.append((pair, int(row["frame_gap"])))
        pair_matches[pair] = []

    rows = read_csv(
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

    return keypoints, pairs, pair_matches


def estimate_fundamental_matrix(points1, points2):
    """基礎行列、対応ごとの採否、処理結果を返す。"""
    if len(points1) < MIN_MATCHES:
        return None, None, "not_enough_matches"

    # 重複座標や一直線上の点しかない場合は推定を見送る。
    for points in (points1, points2):
        if len(np.unique(points, axis=0)) < MIN_MATCHES:
            return None, None, "degenerate_points"
        centered = points - points.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            return None, None, "degenerate_points"

    # 同じ入力・OpenCVバージョンで結果を再現しやすくする。
    cv2.setRNGSeed(RANDOM_SEED)
    matrix, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        RANSAC_THRESHOLD_PX,
        RANSAC_CONFIDENCE,
        RANSAC_MAX_ITERATIONS,
    )
    if matrix is None or mask is None:
        return None, None, "estimation_failed"
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None, None, "estimation_failed"
    if np.linalg.matrix_rank(matrix) != 2 or mask.size != len(points1):
        return None, None, "estimation_failed"

    inliers = mask.ravel().astype(bool)
    if np.count_nonzero(inliers) < MIN_MATCHES:
        return None, None, "not_enough_inliers"
    return matrix, inliers, "ok"


def write_results(directory, keypoints, pairs, pair_matches):
    """全ペアを処理し、採用対応と基礎行列を別々のCSVに保存する。"""
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

        for pair, frame_gap in pairs:
            matches = pair_matches[pair]
            points1 = []
            points2 = []
            for index1, index2 in matches:
                points1.append(keypoints[(pair[0], index1)])
                points2.append(keypoints[(pair[1], index2)])
            points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
            points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)

            matrix, inliers, status = estimate_fundamental_matrix(points1, points2)
            num_inliers = 0
            matrix_values = [""] * 9
            if status == "ok":
                successful_pairs += 1
                matrix_values = matrix.ravel().tolist()
                for match, is_inlier in zip(matches, inliers):
                    if is_inlier:
                        index1, index2 = match
                        match_writer.writerow([pair[0], pair[1], index1, index2])
                        num_inliers += 1

            num_matches = len(matches)
            ratio = num_inliers / num_matches if num_matches else 0.0
            record = [
                pair[0], pair[1], frame_gap, status,
                num_matches, num_inliers, ratio,
            ]
            record.extend(matrix_values)
            matrix_writer.writerow(record)
            print(
                f"{pair[0]} -> {pair[1]}: "
                f"{num_matches} -> {num_inliers} ({ratio:.1%}), {status}"
            )
            total_matches += num_matches
            total_inliers += num_inliers

    print(f"Estimated pairs: {successful_pairs}/{len(pairs)}")
    print(f"Total matches: {total_matches} -> {total_inliers}")


def main():
    keypoints, pairs, pair_matches = load_inputs()
    # 計算を完了してからCSVを出力先へ移す。
    with TemporaryDirectory(dir=BASE_DIR) as temporary:
        staging = Path(temporary)
        write_results(staging, keypoints, pairs, pair_matches)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(OUTPUT_DIR / path.name)
    print(f"CSV output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
