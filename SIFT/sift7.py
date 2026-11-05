"""sift6.pyのNPZを読み、対応候補とRANSACによる幾何検証を可視化する。

準備: python -m pip install numpy pillow matplotlib
実行: python sift7.py
スクリプトはoutputフォルダーと同じ階層に置く。

入力:
  output/descriptors/view_01_wood_descriptors.npz
  output/descriptors/view_02_wood_descriptors.npz
  output/view_01_wood.png と output/view_02_wood.png（可視化用）
出力: output/matches

処理は、同一フレームの重複・ゼロベクトル除去、RootSIFTのユークリッド距離による最近傍照合、
距離比テスト、双方向照合、同じ座標ペアの重複除去、FのRANSACの順。
RANSACは説明用のFモデルと正規化8点法を使う。COLMAP全体のE/F/H選択を
扱う場合には、幾何検証の部分を拡張する。

Fは x2.T @ F @ x1 = 0 の規約。誤差は平方根を取ったSampson誤差 [px]。
sift6のNPZは左上画素中心を(0,0)としているので、読み込んだ座標に0.5を加える。
計算・CSV・結果NPZでは、左上画素中心が(0.5,0.5)となる規約に統一する。
可視化時は0.5を引いて、画像配列の画素位置に戻す。

view_01_wood_matches.csv と view_02_wood_matches.csv は同じidを対応ペアにする。
既存のestimate_relative_pose.pyへ、この2ファイルを指定できる。
"""

import colorsys
import csv
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


INPUT_DIR = Path(__file__).resolve().parent / "output"
DESCRIPTOR_DIR = INPUT_DIR / "descriptors"
OUTPUT_DIR = INPUT_DIR / "matches"
IMAGE_NAMES = ("view_01_wood.png", "view_02_wood.png")
DESCRIPTOR_NAMES = ("view_01_wood_descriptors.npz", "view_02_wood_descriptors.npz")

RATIO_THRESHOLD = 0.8
RANSAC_THRESHOLD_PX = 1.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MIN_TRIALS = 100
RANSAC_MAX_TRIALS = 10000
RANSAC_REFIT_STEPS = 10
RANDOM_SEED = 0


def load_features(path):
    """座標とfloat版RootSIFTを読み、同じ位置・スケール・向きの重複を除く。"""
    with np.load(path, allow_pickle=False) as data:
        if "keypoints" not in data or "descriptors" not in data:
            raise ValueError(f"{path}: sift6.pyのkeypointsとdescriptorsが必要です。")
        keypoints = np.asarray(data["keypoints"], dtype=np.float64)
        descriptors = np.asarray(data["descriptors"], dtype=np.float64)

    if keypoints.ndim != 2 or keypoints.shape[1] != 4:
        raise ValueError(f"{path}: keypointsの形状は(N, 4)が必要です。")
    if descriptors.shape != (len(keypoints), 128):
        raise ValueError(f"{path}: descriptorsの形状は(N, 128)が必要です。")
    if not np.all(np.isfinite(keypoints)) or not np.all(np.isfinite(descriptors)):
        raise ValueError(f"{path}: 座標または記述子に非有限値があります。")

    seen = set()
    kept_indices = []
    zero_descriptors = 0
    for index in range(len(keypoints)):
        if not np.any(descriptors[index]):
            zero_descriptors += 1
            continue
        # 向きが違う点は別の記述子なので残す。同一フレームだけをまとめる。
        frame = tuple(keypoints[index])
        if frame not in seen:
            seen.add(frame)
            kept_indices.append(index)

    source_indices = np.asarray(kept_indices, dtype=np.int64)
    print(f"{path.name}: 入力{len(keypoints)}記述子 -> 同一フレーム・ゼロベクトル除去後{len(source_indices)}")
    if zero_descriptors > 0:
        print(f"  ゼロベクトルによる除去 {zero_descriptors}記述子")
    return keypoints[source_indices], descriptors[source_indices], source_indices


def ratio_test(source_descriptors, target_descriptors):
    """1位と2位の距離比が小さい場合に、1位の対応を採用する。"""
    accepted = {}
    if len(target_descriptors) < 2:
        return accepted

    for source_index, descriptor in enumerate(source_descriptors):
        differences = target_descriptors - descriptor
        distances = np.linalg.norm(differences, axis=1)
        order = np.argsort(distances, kind="stable")
        first_index = int(order[0])
        second_index = int(order[1])
        first_distance = float(distances[first_index])
        second_distance = float(distances[second_index])

        # 2位も距離0なら区別できないので、0/0を計算せずに除外する。
        if second_distance <= 0:
            continue
        ratio = first_distance / second_distance
        if ratio < RATIO_THRESHOLD:
            accepted[source_index] = {
                "target_index": first_index,
                "distance": first_distance,
                "ratio": ratio,
            }
    return accepted


def match_features(features1, features2):
    """両方向で距離比テストを行い、互いを選んだ対応を残す。"""
    keypoints1, descriptors1, source_indices1 = features1
    keypoints2, descriptors2, source_indices2 = features2
    forward = ratio_test(descriptors1, descriptors2)
    reverse = ratio_test(descriptors2, descriptors1)
    matches = []

    for index1, forward_match in forward.items():
        index2 = forward_match["target_index"]
        if index2 not in reverse:
            continue
        if reverse[index2]["target_index"] != index1:
            continue

        match = {
            "index1": int(source_indices1[index1]),
            "index2": int(source_indices2[index2]),
            "point1": keypoints1[index1, :2] + 0.5,
            "point2": keypoints2[index2, :2] + 0.5,
            "distance": forward_match["distance"],
            "ratio12": forward_match["ratio"],
            "ratio21": reverse[index2]["ratio"],
        }
        matches.append(match)

    def descriptor_distance(match):
        return match["distance"]

    matches.sort(key=descriptor_distance)
    unique_matches = []
    seen_pairs = set()
    for match in matches:
        # 別の主方向が同じ座標ペアへ対応した場合、幾何に渡す観測は1組にする。
        pair = tuple(match["point1"]) + tuple(match["point2"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_matches.append(match)

    print(f"距離比テスト < {RATIO_THRESHOLD}: 画像1->2 {len(forward)}組 / 画像2->1 {len(reverse)}組")
    print(f"双方向で一致 {len(matches)}組 -> 同じ座標ペアの重複除去後 {len(unique_matches)}組")
    return unique_matches


def center_and_normalize_points(points):
    """既存の姿勢推定と同じ、重心0・RMS距離sqrt(2)の座標正規化。"""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rms_distance = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if not np.isfinite(rms_distance) or rms_distance <= np.finfo(float).eps:
        raise ValueError("座標の広がりが小さすぎます。")
    scale = np.sqrt(2.0) / rms_distance
    transform = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    return centered * scale, transform


def estimate_fundamental(points1, points2):
    """正規化8点法で、元のピクセル座標に対するランク2のFを求める。"""
    if len(points1) < 8:
        raise ValueError("Fの推定には8組以上の対応点が必要です。")
    work1, transform1 = center_and_normalize_points(points1)
    work2, transform2 = center_and_normalize_points(points2)
    x, y = work1.T
    xp, yp = work2.T
    matrix = np.column_stack([
        xp * x, xp * y, xp,
        yp * x, yp * y, yp,
        x, y, np.ones_like(x),
    ])
    # 8×9の場合も、9次元の零空間を得るためにfull_matrices=Trueにする。
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    if np.count_nonzero(singular_values > tolerance) < 8:
        raise ValueError("対応点の配置が退化しており、Fを一意に決められません。")
    initial_f = vt[-1].reshape(3, 3)
    u, values, vt = np.linalg.svd(initial_f)
    values[-1] = 0.0
    normalized_f = u @ np.diag(values) @ vt
    fundamental = transform2.T @ normalized_f @ transform1
    norm = np.linalg.norm(fundamental)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("有効なFが求まりませんでした。")
    return fundamental / norm


def squared_sampson_errors(fundamental, points1, points2):
    """エピポーラ制約のずれを、勾配で補正したSampson誤差の二乗 [px^2]。"""
    ones = np.ones((len(points1), 1))
    homogeneous1 = np.column_stack([points1, ones])
    homogeneous2 = np.column_stack([points2, ones])
    lines2 = homogeneous1 @ fundamental.T
    lines1 = homogeneous2 @ fundamental
    residuals = np.sum(homogeneous2 * lines2, axis=1)
    denominator = np.sum(lines2[:, :2]**2, axis=1) + np.sum(lines1[:, :2]**2, axis=1)
    errors = np.full(len(points1), np.inf)
    valid = denominator > 0
    errors[valid] = residuals[valid]**2 / denominator[valid]
    return errors


def model_score(errors):
    """しきい値内の組数を優先し、同数なら切り詰めた誤差の合計で比較する。"""
    threshold_squared = RANSAC_THRESHOLD_PX**2
    mask = errors <= threshold_squared
    count = int(np.count_nonzero(mask))
    cost = float(np.sum(np.minimum(errors, threshold_squared)))
    return mask, count, cost


def ransac_fundamental(points1, points2):
    """8組を抽出して仮説を作り、全対応で支持されるFを探す。"""
    count = len(points1)
    best_f = None
    best_mask = np.zeros(count, dtype=bool)
    best_errors = np.full(count, np.inf)
    best_count = 0
    best_cost = np.inf
    history = []
    degenerate_samples = 0
    if count < 8:
        return best_f, best_mask, best_errors, history, degenerate_samples

    random = np.random.default_rng(RANDOM_SEED)
    trial_limit = RANSAC_MAX_TRIALS
    trial = 0
    while trial < trial_limit:
        trial += 1
        sample = random.choice(count, size=8, replace=False)
        sample1 = points1[sample]
        sample2 = points2[sample]
        try:
            if len(np.unique(sample1, axis=0)) < 8 or len(np.unique(sample2, axis=0)) < 8:
                raise ValueError("同じ画像座標を含むサンプルです。")
            fundamental = estimate_fundamental(sample1, sample2)
        except (ValueError, np.linalg.LinAlgError):
            degenerate_samples += 1
            history.append(best_count)
            continue

        errors = squared_sampson_errors(fundamental, points1, points2)
        mask, inlier_count, cost = model_score(errors)
        improved = inlier_count > best_count or (inlier_count == best_count and cost < best_cost)
        if inlier_count >= 8 and improved:
            best_f = fundamental
            best_mask = mask
            best_errors = errors
            best_count = inlier_count
            best_cost = cost

            # 全8組がインライアになる確率から、必要な試行回数を更新する。
            all_inlier_probability = (best_count / count)**8
            if all_inlier_probability >= 1.0:
                required_trials = RANSAC_MIN_TRIALS
            else:
                required_trials = int(np.ceil(
                    np.log1p(-RANSAC_CONFIDENCE) / np.log1p(-all_inlier_probability)
                ))
            trial_limit = min(trial_limit, max(RANSAC_MIN_TRIALS, required_trials))
        history.append(best_count)

    # 最良モデルの全インライアで再推定し、支持数・誤差が改善する間は繰り返す。
    for step in range(RANSAC_REFIT_STEPS):
        if best_f is None:
            break
        try:
            fundamental = estimate_fundamental(points1[best_mask], points2[best_mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        errors = squared_sampson_errors(fundamental, points1, points2)
        mask, inlier_count, cost = model_score(errors)
        improved = inlier_count > best_count or (inlier_count == best_count and cost < best_cost)
        if not improved:
            break
        same_mask = np.array_equal(mask, best_mask)
        best_f = fundamental
        best_mask = mask
        best_errors = errors
        best_count = inlier_count
        best_cost = cost
        if same_mask:
            break

    return best_f, best_mask, best_errors, history, degenerate_samples


def save_results(matches, points1, points2, fundamental, mask, errors, history):
    """全候補の診断CSV、採用ペアのCSV2個、Fと判定結果のNPZを保存する。"""
    with (OUTPUT_DIR / "matches_candidates.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "index1", "index2", "x1", "y1", "x2", "y2", "distance",
                         "ratio12", "ratio21", "inlier", "sampson_error_px"])
        for index, match in enumerate(matches):
            writer.writerow([
                index + 1, match["index1"], match["index2"],
                *points1[index], *points2[index], match["distance"],
                match["ratio12"], match["ratio21"], int(mask[index]), np.sqrt(errors[index]),
            ])

    # 画像間で同じidを持たせ、既存の姿勢推定へそのまま渡せるようにする。
    for view, points in ((1, points1), (2, points2)):
        path = OUTPUT_DIR / f"view_{view:02d}_wood_matches.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "x", "y"])
            for pair_id, xy in enumerate(points[mask], start=1):
                writer.writerow([pair_id, float(xy[0]), float(xy[1])])
        print(f"Saved: {path.name} ({np.count_nonzero(mask)}組)")

    source_indices = np.zeros((len(matches), 2), dtype=np.int64)
    for index, match in enumerate(matches):
        source_indices[index] = [match["index1"], match["index2"]]
    stored_f = fundamental
    status = "success"
    if fundamental is None:
        stored_f = np.empty((0, 0))
        status = "no_valid_model"
    np.savez_compressed(
        OUTPUT_DIR / "matches_result.npz",
        F=stored_f, points1=points1, points2=points2, source_indices=source_indices,
        inlier_mask=mask, sampson_errors_px=np.sqrt(errors), history=np.asarray(history),
        threshold_px=RANSAC_THRESHOLD_PX, ratio_threshold=RATIO_THRESHOLD,
        coordinate_convention="COLMAP: upper-left pixel center (0.5,0.5)", status=status,
    )


def save_figure(figure, filename):
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=150)
        (OUTPUT_DIR / filename).write_bytes(buffer.getvalue())
    plt.close(figure)
    print(f"Saved: {filename}", flush=True)


def save_match_preview(image1, image2, points1, points2, mask):
    """対応候補とRANSACの判定を、同じ座標配置で上下に並べる。"""
    gap = 32
    right_offset = image1.width + gap
    canvas = Image.new("RGB", (right_offset + image2.width, max(image1.height, image2.height)), "white")
    canvas.paste(image1, (0, 0))
    canvas.paste(image2, (right_offset, 0))
    figure, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

    for panel, axis in enumerate(axes):
        axis.imshow(canvas)
        # 除去点を先に描き、残る対応を見やすくする。
        for accepted in (False, True):
            for index in range(len(points1)):
                if bool(mask[index]) != accepted:
                    continue
                x1, y1 = points1[index] - 0.5
                x2, y2 = points2[index] - 0.5
                x2 += right_offset
                if panel == 0:
                    hue = (index * 0.61803398875) % 1.0
                    color = colorsys.hsv_to_rgb(hue, 0.8, 0.85)
                    line_style = "-"
                elif accepted:
                    color = "tab:green"
                    line_style = "-"
                else:
                    color = "tab:red"
                    line_style = "--"
                axis.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, linewidth=0.9, alpha=0.8)
                axis.scatter([x1, x2], [y1, y2], color=color, s=17, edgecolors="white", linewidths=0.3)
        axis.set_xlim(-0.5, canvas.width - 0.5)
        axis.set_ylim(canvas.height - 0.5, -0.5)
        axis.set_axis_off()

    axes[0].set_title(f"Descriptor candidates: {len(points1)} unique coordinate pairs | ratio < {RATIO_THRESHOLD}")
    axes[1].set_title(
        f"RANSAC F: {np.count_nonzero(mask)} inliers / {len(mask)} candidates | "
        f"Sampson error <= {RANSAC_THRESHOLD_PX} px"
    )
    axes[1].legend(handles=[
        Line2D([0], [0], color="tab:green", label="Kept (geometric inlier)"),
        Line2D([0], [0], color="tab:red", linestyle="--", label="Rejected"),
    ], loc="lower center", fontsize=10)
    save_figure(figure, "matches_comparison.png")


def save_ransac_diagnostics(mask, errors, history):
    """RANSACの最良支持数の推移と、各候補の誤差を表示する。"""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axis = axes[0]
    axis.plot(np.arange(1, len(history) + 1), history, linewidth=1.5)
    axis.set_title("Best consensus during random sampling")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Number of inliers")
    axis.grid(alpha=0.25)

    axis = axes[1]
    error_px = np.sqrt(errors)
    indices = np.arange(1, len(errors) + 1)
    finite = np.isfinite(error_px)
    axis.scatter(indices[finite & mask], error_px[finite & mask], color="tab:green", label="Kept")
    axis.scatter(indices[finite & ~mask], error_px[finite & ~mask], color="tab:red", label="Rejected")
    axis.axhline(RANSAC_THRESHOLD_PX, color="black", linestyle="--", label="Threshold")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_ylim(bottom=0)
    axis.set_title(f"Final Sampson errors | undefined: {np.count_nonzero(~finite)}")
    axis.set_xlabel("Candidate pair id")
    axis.set_ylabel("Error (pixels)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    save_figure(figure, "ransac_diagnostics.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features1 = load_features(DESCRIPTOR_DIR / DESCRIPTOR_NAMES[0])
    features2 = load_features(DESCRIPTOR_DIR / DESCRIPTOR_NAMES[1])
    matches = match_features(features1, features2)
    points1 = np.zeros((len(matches), 2))
    points2 = np.zeros((len(matches), 2))
    for index, match in enumerate(matches):
        points1[index] = match["point1"]
        points2[index] = match["point2"]

    fundamental, mask, errors, history, degenerate = ransac_fundamental(points1, points2)
    print(f"RANSAC: 試行{len(history)}回 / 退化サンプル{degenerate}回 / しきい値{RANSAC_THRESHOLD_PX} px")
    print(f"対応候補 {len(matches)}組 -> インライア {np.count_nonzero(mask)}組 / 除去 {np.count_nonzero(~mask)}組")
    if fundamental is None:
        print("有効なFを推定できませんでした。8組以上の有効な対応と、点の配置を確認してください。")
    else:
        print("F（x2.T @ F @ x1 = 0、左上画素中心は0.5,0.5）")
        print(fundamental)
        print(f"インライアのSampson誤差の中央値 {np.median(np.sqrt(errors[mask])):.4f} px")

    save_results(matches, points1, points2, fundamental, mask, errors, history)
    with Image.open(INPUT_DIR / IMAGE_NAMES[0]) as source1:
        image1 = source1.convert("RGB")
    with Image.open(INPUT_DIR / IMAGE_NAMES[1]) as source2:
        image2 = source2.convert("RGB")
    save_match_preview(image1, image2, points1, points2, mask)
    save_ransac_diagnostics(mask, errors, history)


if __name__ == "__main__":
    main()
