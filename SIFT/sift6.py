"""sift5の特徴点に、COLMAP CPU版の規約に沿う128次元記述子を加える。

準備: python -m pip install numpy pillow scipy matplotlib
実行: python sift6.py
入力: スクリプトと同じ場所の output フォルダー
出力: output/gaussian、output/dog、output/keypoints、output/keypoints_refined、output/orientations

候補の検出は、浮動小数点のDoG画像で周囲26点と比較する。
円の半径はその候補のスケールσ（表示時は最低2 px）。
位置とスケール方向の3変数 (x, y, s) を二次式で補正する。
sはDoGのレベル番号。元画像でのσは
BASE_SIGMA * 2**(s / SCALES_PER_OCTAVE) * 2**octave で求める。
補正後の応答 |D| が CONTRAST_THRESHOLD 未満の候補を除去する。
既定の0.03は、画素値を0～1に正規化したときのLowe論文の実験値。
主方向は、近いスケールのガウシアン画像から求める勾配の36ビンで決める。
最大ピークの80%以上の局所ピークを採用し、最大4方向を割り当てる。
角度は画像の右向きが0度、下向きが90度の時計回り（y軸は下向き）。
矢印の長さは表示用に3σ、最低7 px。円の半径はσ、最低2 px。
標準出力には候補数、補正例、主方向の角度、複数方向を持つ点の数を表示する。
この段階の入力は低コントラスト判定を通った候補。エッジ判定は後続の段階。
画像ごとに1点を選び、3×3の輝度・勾配と主方向の計算を詳しく可視化する。
3×3全画素の中央差分には、外側1画素を含む5×5の値を参照する。

記述子は4×4区画×8方向への三重線形補間、L2→0.2打ち切り→L2、
RootSIFT（L1正規化→平方根）、512倍・四捨五入・uint8化の順に計算する。
方向ビンはVLFeat順からCOLMAPのUBC順 (0,7,6,5,4,3,2,1) に変換する。
記述子の出力先はoutput/descriptors。npzに全点の座標と記述子を保存する。
座標はsift5と同じ、元画像の画素中心を0始まりで表す。
ガウシアン画像・候補点はsift5の教材実装。記述子はNumPyの倍精度と
exp/atan2で計算するため、VLFeatのfloat・高速近似演算との数値差がある。
参考: https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc
      https://github.com/colmap/colmap/blob/main/src/colmap/feature/utils.cc
      https://www.vlfeat.org/api/sift.html
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates


INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = INPUT_DIR / "gaussian"
DOG_DIR = INPUT_DIR / "dog"
KEYPOINT_DIR = INPUT_DIR / "keypoints"
REFINED_DIR = INPUT_DIR / "keypoints_refined"
ORIENTATION_DIR = INPUT_DIR / "orientations"
DESCRIPTOR_DIR = INPUT_DIR / "descriptors"
INPUT_NAMES = ("view_01_wood.png", "view_02_wood.png")

FIRST_OCTAVE = -1
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 3
BASE_SIGMA = 1.6
INPUT_SIGMA = 0.5  # 元画像がすでに持つと仮定するぼかし

CONTRAST_THRESHOLD = 0.03
MAX_REFINEMENT_STEPS = 5
NUM_COORDINATE_EXAMPLES = 3  # 採用・低コントラスト除去の各グループから表示する数

ORIENTATION_BINS = 36
ORIENTATION_WINDOW_FACTOR = 1.5
ORIENTATION_SMOOTHING_PASSES = 6
ORIENTATION_PEAK_RATIO = 0.8
MAX_ORIENTATIONS = 4
NUM_ORIENTATION_EXAMPLES = 5

DESCRIPTOR_SPATIAL_BINS = 4
DESCRIPTOR_ORIENTATION_BINS = 8
DESCRIPTOR_MAGNIFICATION = 3.0
DESCRIPTOR_WINDOW_SIGMA = 2.0  # 空間ビンの幅を1としたときのσ
DESCRIPTOR_CLIP = 0.2


def gaussian_pyramid(gray):
    """(オクターブ番号, 各レベルのガウシアン画像)のリストを返す。"""
    # uint8のままぼかすと途中で丸められるため、先にfloat32にする。
    image = gray.astype(np.float32) / 255.0

    # オクターブ -1 では元画像を2倍に拡大する。
    # ぼかしを拡大後の画素単位で表すと、0.5 px → 1.0 px。
    height, width = image.shape
    factor = 2 ** -FIRST_OCTAVE
    enlarged = Image.fromarray(image).resize(
        (width * factor, height * factor), Image.Resampling.BILINEAR
    )
    sigma = np.sqrt(BASE_SIGMA**2 - (factor * INPUT_SIGMA)**2)
    base = gaussian_filter(np.asarray(enlarged), sigma, mode="nearest")
    k = 2 ** (1 / SCALES_PER_OCTAVE)

    pyramid = []
    for octave in range(FIRST_OCTAVE, FIRST_OCTAVE + NUM_OCTAVES):
        levels = [base]
        for level in range(1, SCALES_PER_OCTAVE + 3):
            previous_sigma = BASE_SIGMA * k ** (level - 1)
            target_sigma = BASE_SIGMA * k**level

            # ガウシアンを重ねると分散が足し合わされる。
            # 前段に加えるぼかしは目標のσそのものではなく、分散の差で求める。
            extra_sigma = np.sqrt(target_sigma**2 - previous_sigma**2)
            blurred = gaussian_filter(levels[-1], extra_sigma, mode="nearest")
            levels.append(blurred)

        pyramid.append((octave, levels))

        # レベル3はσ=3.2 px。画素を間引いて半分の解像度にすると
        # 新しい画素単位でσ=1.6 pxとなり、次のオクターブの先頭になる。
        base = levels[SCALES_PER_OCTAVE][::2, ::2]

    return pyramid


def save_png(path, image):
    Image.fromarray(image).save(path)
    print(f"Saved: {path.name} ({image.shape[1]}×{image.shape[0]})")


def difference_of_gaussians(levels):
    """同じオクターブ内で隣り合うガウシアン画像の差を求める。"""
    dog_images = []

    for level in range(len(levels) - 1):
        current_image = levels[level]
        next_image = levels[level + 1]

        dog_image = next_image - current_image
        dog_images.append(dog_image)

    return dog_images


def dog_preview(dog):
    """DoGの符号を保ち、ゼロを中間グレーにして表示する。"""
    max_abs = np.max(np.abs(dog))
    if max_abs == 0:
        return np.full(dog.shape, 128, dtype=np.uint8)

    # 画像ごとに対称な範囲 [-max_abs, +max_abs] を 0～255 に写す。
    return np.rint((dog / max_abs + 1) * 127.5).astype(np.uint8)


def classify_extremum(dog_images, level, y, x):
    """周囲26点と比較し、極大は1、極小は-1、それ以外は0を返す。"""
    center_value = dog_images[level][y, x]
    is_maximum = True
    is_minimum = True

    # 同じスケールの8点と、前後のスケールの各9点を調べる。
    for level_offset in range(-1, 2):
        neighbor_image = dog_images[level + level_offset]

        for y_offset in range(-1, 2):
            for x_offset in range(-1, 2):
                # 中心そのものは比較から外す。
                if level_offset == 0 and y_offset == 0 and x_offset == 0:
                    continue

                neighbor_value = neighbor_image[y + y_offset, x + x_offset]

                # 同じ値の点がある場合も、厳密な極大・極小にはしない。
                if center_value <= neighbor_value:
                    is_maximum = False
                if center_value >= neighbor_value:
                    is_minimum = False

                # 一つでも大きい点と小さい点があれば、そこで判定を終える。
                # 同値の点があった場合も、この条件で判定を終える。
                if not is_maximum and not is_minimum:
                    return 0

    if is_maximum:
        return 1
    if is_minimum:
        return -1
    return 0


def find_keypoint_candidates(dog_images, octave):
    """1オクターブのDoG画像から極値の候補をリストで返す。"""
    candidates = []
    height, width = dog_images[0].shape
    k = 2 ** (1 / SCALES_PER_OCTAVE)

    # このオクターブの1画素が、元画像の何画素分に当たるか。
    pixel_size = 2.0**octave

    # Pillowの2倍拡大では、左上画素の中心が元画像の-0.25に来る。
    # その後の間引きでも左上の位置は同じなので、このずれを含めて戻す。
    first_pixel_size = 2.0**FIRST_OCTAVE
    origin_offset = (first_pixel_size - 1.0) / 2.0

    # 最初と最後のDoG画像には、比較に必要な前後の画像がそろわない。
    # DoG画像5枚なら、レベル1、2、3を調べる。
    for level in range(1, len(dog_images) - 1):
        # 画像の端の1画素も、周囲がそろわないので調べない。
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                kind = classify_extremum(dog_images, level, y, x)
                if kind == 0:
                    continue

                x_original = x * pixel_size + origin_offset
                y_original = y * pixel_size + origin_offset
                sigma_original = BASE_SIGMA * k**level * pixel_size

                candidate = {
                    "octave": octave,
                    "level": level,
                    "x_octave": x,
                    "y_octave": y,
                    "x": x_original,
                    "y": y_original,
                    "sigma": sigma_original,
                    "value": float(dog_images[level][y, x]),
                    "kind": kind,
                }
                candidates.append(candidate)

    return candidates


def save_keypoint_preview(path, color_image, candidates):
    """元のカラー画像に候補の位置とスケールを重ねて保存する。"""
    preview = color_image.copy()
    draw = ImageDraw.Draw(preview)

    for candidate in candidates:
        x = candidate["x"]
        y = candidate["y"]
        radius = max(2.0, candidate["sigma"])

        if candidate["kind"] == 1:
            color = (255, 40, 40)  # 極大は赤
        else:
            color = (30, 100, 255)  # 極小は青

        bounds = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bounds, outline=color, width=1)
        draw.point((x, y), fill=color)

    label = f"DoG extrema: {len(candidates)} | red: max / blue: min"
    text_bounds = draw.textbbox((5, 5), label)
    draw.rectangle((0, 0, preview.width, text_bounds[3] + 5), fill="white")
    draw.text((5, 5), label, fill="black")
    preview.save(path)
    print(f"Saved: {path.name} ({len(candidates)} candidates)", flush=True)


def dog_derivatives(dog_images, level, y, x):
    """(x, y, s) 方向の勾配gと3×3のヘッセ行列Hを、中央差分で求める。"""
    # 配列自体はfloat32。差分と連立方程式の計算はfloat64で行う。
    previous = dog_images[level - 1][y - 1:y + 2, x - 1:x + 2].astype(np.float64)
    current = dog_images[level][y - 1:y + 2, x - 1:x + 2].astype(np.float64)
    following = dog_images[level + 1][y - 1:y + 2, x - 1:x + 2].astype(np.float64)
    center = current[1, 1]

    # 一階微分。sはぼかしのレベル番号で、1の差が隣のDoG画像に当たる。
    dx = (current[1, 2] - current[1, 0]) / 2.0
    dy = (current[2, 1] - current[0, 1]) / 2.0
    ds = (following[1, 1] - previous[1, 1]) / 2.0
    gradient = np.array([dx, dy, ds])

    # 同じ方向で二回微分した値。
    dxx = current[1, 2] - 2.0 * center + current[1, 0]
    dyy = current[2, 1] - 2.0 * center + current[0, 1]
    dss = following[1, 1] - 2.0 * center + previous[1, 1]

    # 異なる二方向で一回ずつ微分した値。
    dxy = (
        current[2, 2] - current[2, 0]
        - current[0, 2] + current[0, 0]
    ) / 4.0
    dxs = (
        following[1, 2] - following[1, 0]
        - previous[1, 2] + previous[1, 0]
    ) / 4.0
    dys = (
        following[2, 1] - following[0, 1]
        - previous[2, 1] + previous[0, 1]
    ) / 4.0
    hessian = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss],
    ])
    return gradient, hessian


def refine_keypoint(dog_images, candidate):
    """二次式の停留点を求める。成功なら補正点、失敗なら理由を返す。"""
    x = candidate["x_octave"]
    y = candidate["y_octave"]
    level = candidate["level"]
    height, width = dog_images[0].shape

    for iteration in range(MAX_REFINEMENT_STEPS):
        # 差分を求めるため、空間・スケールの前後の点が必要。
        if (
            x < 1 or x >= width - 1
            or y < 1 or y >= height - 1
            or level < 1 or level >= len(dog_images) - 1
        ):
            return None, "out_of_bounds"

        gradient, hessian = dog_derivatives(dog_images, level, y, x)

        # D(u) ≈ D + g^T u + (1/2) u^T H u
        # 微分して0とすると g + H u = 0。これを解いて補正量uを得る。
        try:
            offset = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            return None, "singular_hessian"

        if not np.all(np.isfinite(offset)):
            return None, "non_finite"

        # 3方向とも半格子以内なら、この格子点を基準に補正を確定する。
        if np.max(np.abs(offset)) <= 0.5:
            refined_x = x + offset[0]
            refined_y = y + offset[1]
            refined_level = level + offset[2]

            # 停留点における二次式の値。符号付きのDoG応答を保持する。
            center_value = float(dog_images[level][y, x])
            refined_value = center_value + 0.5 * np.dot(gradient, offset)
            if not np.isfinite(refined_value):
                return None, "non_finite"

            pixel_size = 2.0**candidate["octave"]
            origin_offset = (2.0**FIRST_OCTAVE - 1.0) / 2.0
            k = 2 ** (1 / SCALES_PER_OCTAVE)

            refined = candidate.copy()
            # 表示で、最初に検出した整数座標と比較できるように残す。
            refined["initial_x_octave"] = candidate["x_octave"]
            refined["initial_y_octave"] = candidate["y_octave"]
            refined["initial_level"] = candidate["level"]
            refined["initial_x"] = candidate["x"]
            refined["initial_y"] = candidate["y"]
            refined["initial_sigma"] = candidate["sigma"]
            refined["initial_value"] = candidate["value"]

            refined["x_octave"] = float(refined_x)
            refined["y_octave"] = float(refined_y)
            refined["level"] = level  # 最後に二次式を求めた整数のレベル
            refined["level_float"] = float(refined_level)
            refined["x"] = float(refined_x * pixel_size + origin_offset)
            refined["y"] = float(refined_y * pixel_size + origin_offset)
            refined["sigma"] = float(BASE_SIGMA * k**refined_level * pixel_size)
            refined["value"] = float(refined_value)
            refined["offset"] = offset
            refined["iterations"] = iteration + 1
            return refined, None

        # 停留点が半格子を超えた方向へ、基準の格子点を一つ移す。
        # 移動先の周囲で二次式を求め直すので、補正量を足し続ける処理とは異なる。
        if offset[0] > 0.5:
            x += 1
        elif offset[0] < -0.5:
            x -= 1

        if offset[1] > 0.5:
            y += 1
        elif offset[1] < -0.5:
            y -= 1

        if offset[2] > 0.5:
            level += 1
        elif offset[2] < -0.5:
            level -= 1

    return None, "not_converged"


def refine_candidates(dog_images, candidates):
    """候補を補正して、採用・低コントラスト・補正失敗に分けて返す。"""
    kept = []
    weak = []
    failed = []

    for candidate in candidates:
        refined, failure_reason = refine_keypoint(dog_images, candidate)
        if refined is None:
            rejected = candidate.copy()
            rejected["reject_reason"] = failure_reason
            failed.append(rejected)
            continue

        # |D(u)|が小さい点は、濃淡の変化が弱く不安定なので除去する。
        if abs(refined["value"]) < CONTRAST_THRESHOLD:
            weak.append(refined)
        else:
            kept.append(refined)

    return kept, weak, failed


def print_candidate_counts(label, candidates, kept, weak, failed):
    """補正の成否と、低コントラストによる除去数を混ぜずに表示する。"""
    localized_count = len(kept) + len(weak)
    print(f"\n[{label}]", flush=True)
    print(f"  検出した候補数                  {len(candidates)}")
    print(f"  二次式による補正の成功数        {localized_count}")
    print(f"  補正失敗による除去数            {len(failed)}")
    print(f"  低コントラストによる除去数      {len(weak)}")
    print(f"  採用候補数                      {len(kept)}")
    print(f"  候補数の変化  {len(candidates)} -> {localized_count} -> {len(kept)}")

    # 補正失敗も、どの理由で発生したかを確認できるようにする。
    failure_counts = {}
    for candidate in failed:
        reason = candidate["reject_reason"]
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    for reason, count in failure_counts.items():
        print(f"    補正失敗の内訳  {reason} = {count}")


def print_coordinate_examples(label, keypoints):
    """補正量が読み取りやすい代表点を選び、整数座標からの変化を表示する。"""
    print(f"\n  {label}の補正例（最大{NUM_COORDINATE_EXAMPLES}点）")
    if not keypoints:
        print("    該当する点はありません。")
        return

    # 最終的な基準点からの空間方向の補正量が大きい順に選ぶ。
    # 関数を定義して、並べ替えの基準も読み取れるようにする。
    def offset_size(keypoint):
        offset = keypoint["offset"]
        return offset[0]**2 + offset[1]**2

    examples = sorted(keypoints, key=offset_size, reverse=True)
    for index, keypoint in enumerate(examples[:NUM_COORDINATE_EXAMPLES], start=1):
        print(f"    例{index}  octave={keypoint['octave']}  補正計算={keypoint['iterations']}回")
        print(
            "      各オクターブの (x, y, s) "
            f"({keypoint['initial_x_octave']}, {keypoint['initial_y_octave']}, "
            f"{keypoint['initial_level']}) -> "
            f"({keypoint['x_octave']:.4f}, {keypoint['y_octave']:.4f}, "
            f"{keypoint['level_float']:.4f})"
        )
        delta_x = keypoint["x_octave"] - keypoint["initial_x_octave"]
        delta_y = keypoint["y_octave"] - keypoint["initial_y_octave"]
        delta_s = keypoint["level_float"] - keypoint["initial_level"]
        print(f"      最初の候補からの補正量  dx={delta_x:+.4f}, dy={delta_y:+.4f}, ds={delta_s:+.4f}")
        print(
            "      元画像の (x, y) "
            f"({keypoint['initial_x']:.4f}, {keypoint['initial_y']:.4f}) -> "
            f"({keypoint['x']:.4f}, {keypoint['y']:.4f})"
        )
        print(f"      元画像の σ  {keypoint['initial_sigma']:.4f} -> {keypoint['sigma']:.4f} px")
        print(
            f"      DoG応答  {keypoint['initial_value']:+.6f} -> {keypoint['value']:+.6f}"
            f"  |補正後の応答|={abs(keypoint['value']):.6f}, しきい値={CONTRAST_THRESHOLD}"
        )


def draw_candidate_group(image, candidates, color):
    """指定色で候補の位置とスケールを描く。"""
    draw = ImageDraw.Draw(image)
    for candidate in candidates:
        x = candidate["x"]
        y = candidate["y"]
        radius = max(2.0, candidate["sigma"])
        bounds = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bounds, outline=color, width=1)
        draw.point((x, y), fill=color)


def add_preview_label(image, label):
    draw = ImageDraw.Draw(image)
    bounds = draw.textbbox((5, 5), label)
    draw.rectangle((0, 0, image.width, bounds[3] + 5), fill="white")
    draw.text((5, 5), label, fill="black")


def save_refinement_preview(stem, color_image, candidates, kept, weak, failed):
    """検出時・採用・除去の3枚を並べて、二次式の補正と除去の結果を表示する。"""
    before = color_image.copy()
    accepted = color_image.copy()
    rejected = color_image.copy()

    draw_candidate_group(before, candidates, (50, 90, 240))
    draw_candidate_group(accepted, kept, (0, 145, 55))
    draw_candidate_group(rejected, weak, (230, 35, 35))
    draw_candidate_group(rejected, failed, (240, 140, 0))

    add_preview_label(before, f"Candidates: {len(candidates)}")
    add_preview_label(accepted, f"Refined / kept: {len(kept)} (green)")
    add_preview_label(rejected, f"Rejected: weak={len(weak)} (red), fit failed={len(failed)} (orange)")

    save_png(REFINED_DIR / f"{stem}_refined.png", np.asarray(accepted))
    save_png(REFINED_DIR / f"{stem}_rejected.png", np.asarray(rejected))

    width, height = color_image.size
    comparison = Image.new("RGB", (width * 3, height), "white")
    comparison.paste(before, (0, 0))
    comparison.paste(accepted, (width, 0))
    comparison.paste(rejected, (width * 2, 0))
    save_png(REFINED_DIR / f"{stem}_comparison.png", np.asarray(comparison))


def image_gradient(image, y, x):
    """左右・上下の差分から、勾配ベクトル、強さ、角度を求める。"""
    gradient_x = float(image[y, x + 1]) - float(image[y, x - 1])
    gradient_y = float(image[y + 1, x]) - float(image[y - 1, x])
    magnitude = float(np.hypot(gradient_x, gradient_y))
    # y軸が下向きなので、角度は右向き0度、下向き90度の時計回り。
    angle = float(np.arctan2(gradient_y, gradient_x) % (2.0 * np.pi))
    return gradient_x, gradient_y, magnitude, angle


def orientation_histogram(image, keypoint, sample_bounds=None):
    """周辺の勾配を36方向に集計する。sample_boundsで可視化用の範囲も指定できる。"""
    histogram = np.zeros(ORIENTATION_BINS, dtype=np.float64)
    height, width = image.shape
    center_x = keypoint["x_octave"]
    center_y = keypoint["y_octave"]

    # 元画像単位のσを、今回のオクターブの画素単位に戻す。
    sigma_octave = keypoint["sigma"] / (2.0**keypoint["octave"])
    window_sigma = ORIENTATION_WINDOW_FACTOR * sigma_octave
    window_radius = 3.0 * window_sigma

    # 勾配には左右・上下の画素が必要なので、画像端の1画素は使わない。
    first_x = max(1, int(np.ceil(center_x - window_radius)))
    last_x = min(width - 2, int(np.floor(center_x + window_radius)))
    first_y = max(1, int(np.ceil(center_y - window_radius)))
    last_y = min(height - 2, int(np.floor(center_y + window_radius)))

    # (左端x, 上端y, 右端x, 下端y)。両端の画素を含む。
    # 通常の主方向計算は範囲指定なし。詳細図の3×3の投票だけを取り出すときに使う。
    if sample_bounds is not None:
        first_x = max(first_x, sample_bounds[0])
        first_y = max(first_y, sample_bounds[1])
        last_x = min(last_x, sample_bounds[2])
        last_y = min(last_y, sample_bounds[3])

    for y in range(first_y, last_y + 1):
        for x in range(first_x, last_x + 1):
            distance_x = x - center_x
            distance_y = y - center_y
            distance_squared = distance_x**2 + distance_y**2
            if distance_squared > window_radius**2:
                continue

            # 勾配はDoGではなく、キーポイントのスケールに近い画像から求める。
            gradient_x, gradient_y, magnitude, angle = image_gradient(image, y, x)
            if magnitude == 0:
                continue

            spatial_weight = np.exp(-distance_squared / (2.0 * window_sigma**2))
            vote = magnitude * spatial_weight

            # ビンの中心は0度、10度、20度、...。隣の2ビンへ線形補間して投票する。
            # 例として23度なら、20度のビンへ0.7、30度のビンへ0.3の割合で入れる。
            bin_position = angle * ORIENTATION_BINS / (2.0 * np.pi)
            lower_bin = int(np.floor(bin_position))
            fraction = bin_position - lower_bin
            upper_bin = (lower_bin + 1) % ORIENTATION_BINS
            lower_bin = lower_bin % ORIENTATION_BINS
            histogram[lower_bin] += vote * (1.0 - fraction)
            histogram[upper_bin] += vote * fraction

    return histogram


def smooth_orientation_histogram(histogram):
    """角度の境界をつなぎ、隣接3ビンの平均でヒストグラムを平滑化する。"""
    smoothed = histogram.copy()
    for iteration in range(ORIENTATION_SMOOTHING_PASSES):
        next_histogram = np.zeros_like(smoothed)
        for index in range(ORIENTATION_BINS):
            previous_bin = (index - 1) % ORIENTATION_BINS
            following_bin = (index + 1) % ORIENTATION_BINS
            next_histogram[index] = (
                smoothed[previous_bin] + smoothed[index] + smoothed[following_bin]
            ) / 3.0
        smoothed = next_histogram
    return smoothed


def find_orientation_peaks(histogram):
    """強い局所ピークを探し、隣のビンとの二次式で角度を補間する。"""
    peaks = []
    highest = float(np.max(histogram))
    if highest <= 0:
        return peaks
    minimum_height = ORIENTATION_PEAK_RATIO * highest

    for index in range(ORIENTATION_BINS):
        left = float(histogram[(index - 1) % ORIENTATION_BINS])
        center = float(histogram[index])
        right = float(histogram[(index + 1) % ORIENTATION_BINS])

        if center < minimum_height:
            continue
        # 隣の2ビンが同じ高さになる場合は片側だけを採用し、補間で中間を求める。
        if center <= left or center < right:
            continue

        # 3点の放物線の頂点。補間量はビン単位（1ビンは10度）。
        denominator = left - 2.0 * center + right
        offset = 0.5 * (left - right) / denominator
        angle = ((index + offset) * 2.0 * np.pi / ORIENTATION_BINS) % (2.0 * np.pi)
        # 丸め誤差で2πそのものになった場合も、0度として記録する。
        if angle >= 2.0 * np.pi:
            angle = 0.0
        interpolated_height = center - 0.25 * (left - right) * offset
        peaks.append((interpolated_height, angle))

    # タプルの先頭のピーク高さを基準に並べ、最も強い方向を先にする。
    peaks.sort(reverse=True)
    return peaks[:MAX_ORIENTATIONS]


def assign_orientations(gaussian_levels, keypoints):
    """補正済みの各点に向きを割り当て、向き付きの点と集計を返す。"""
    oriented = []
    multiple_count = 0
    unassigned_count = 0

    for keypoint_id, keypoint in enumerate(keypoints, start=1):
        # 補正したスケールに最も近いガウシアン画像を選ぶ。
        level = int(np.floor(keypoint["level_float"] + 0.5))
        image = gaussian_levels[level]
        histogram = orientation_histogram(image, keypoint)
        histogram = smooth_orientation_histogram(histogram)
        peaks = find_orientation_peaks(histogram)

        if len(peaks) == 0:
            unassigned_count += 1
            continue
        if len(peaks) > 1:
            multiple_count += 1

        # 同じ位置・スケールでも、向きごとに別のキーポイントとして記録する。
        for rank, (height, angle) in enumerate(peaks, start=1):
            point = keypoint.copy()
            point["keypoint_id"] = keypoint_id  # このオクターブ内での元の点の番号
            point["orientation_rad"] = float(angle)
            point["orientation_deg"] = float(np.rad2deg(angle))
            point["orientation_rank"] = rank
            point["orientation_peak"] = height
            oriented.append(point)

    return oriented, multiple_count, unassigned_count


def print_orientation_summary(label, keypoints, oriented, multiple_count, unassigned_count):
    """前段の候補数と、向きを含めたキーポイントの数を表示する。"""
    assigned_count = len(keypoints) - unassigned_count
    extra_count = len(oriented) - assigned_count
    print(f"\n[{label} / 主方向の割り当て]")
    print(f"  入力の補正済み候補数            {len(keypoints)}")
    print(f"  向きを割り当てた候補数          {assigned_count}")
    print(f"  複数の向きが付いた候補数        {multiple_count}")
    print(f"  追加の向きにより増えた点数      {extra_count}")
    print(f"  向きを決められなかった数        {unassigned_count}")
    print(f"  向き付きキーポイント数          {len(oriented)}", flush=True)


def print_orientation_examples(oriented):
    """同じ位置に付いた全ての主方向を、代表点について表示する。"""
    groups = {}
    for point in oriented:
        source_id = (point["octave"], point["keypoint_id"])
        if source_id not in groups:
            groups[source_id] = []
        groups[source_id].append(point)

    # 複数の向きが付いた例を先に表示する。
    examples = list(groups.values())
    examples.sort(key=len, reverse=True)
    print(f"\n  主方向の例（最大{NUM_ORIENTATION_EXAMPLES}点）")
    print("  角度は右向き0度、下向き90度の時計回りです。")
    displayed_positions = set()
    displayed_count = 0
    for group in examples:
        source = group[0]
        position = (source["octave"], source["x_octave"], source["y_octave"], source["level_float"])
        # 別の候補が補正後に同じ位置へ来た場合、表示例の重複を省く。
        if position in displayed_positions:
            continue
        displayed_positions.add(position)
        if displayed_count >= NUM_ORIENTATION_EXAMPLES:
            break
        displayed_count += 1
        angle_texts = []
        for point in group:
            angle_texts.append(f"{point['orientation_deg']:.2f}度")
        angles = ", ".join(angle_texts)
        print(
            f"    octave={source['octave']}, 点番号={source['keypoint_id']}, "
            f"元画像の(x,y)=({source['x']:.4f}, {source['y']:.4f}), "
            f"σ={source['sigma']:.4f} px -> 主方向 {angles}"
        )


def save_orientation_preview(path, color_image, oriented):
    """元画像に円と矢印で位置・スケール・主方向を重ねる。"""
    preview = color_image.copy()
    draw = ImageDraw.Draw(preview)
    source_ids = set()

    for point in oriented:
        x = point["x"]
        y = point["y"]
        angle = point["orientation_rad"]
        source_ids.add((point["octave"], point["keypoint_id"]))
        radius = max(2.0, point["sigma"])
        bounds = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bounds, outline=(60, 60, 60), width=1)

        if point["orientation_rank"] == 1:
            color = (0, 145, 55)  # 最大ピークの方向は緑
        else:
            color = (30, 90, 235)  # 追加の方向は青

        # 円の半径は従来のσ。矢印は見やすい長さを別に設定する。
        length = max(7.0, 3.0 * point["sigma"])
        end_x = x + length * np.cos(angle)
        end_y = y + length * np.sin(angle)
        draw.line((x, y, end_x, end_y), fill=color, width=2)

        # 矢印の先端から、向きの反対側へ2本の短い線を引く。
        for head_angle in (angle - np.pi / 6.0, angle + np.pi / 6.0):
            head_x = end_x - 3.0 * np.cos(head_angle)
            head_y = end_y - 3.0 * np.sin(head_angle)
            draw.line((end_x, end_y, head_x, head_y), fill=color, width=1)

    label = (
        f"Candidates: {len(source_ids)} | oriented keypoints: {len(oriented)}\n"
        "Green: strongest / Blue: additional | 0 deg: right / 90 deg: down"
    )
    text_bounds = draw.multiline_textbbox((5, 5), label)
    draw.rectangle((0, 0, preview.width, text_bounds[3] + 5), fill="white")
    draw.multiline_text((5, 5), label, fill="black")
    save_png(path, np.asarray(preview))


def select_gradient_example(gaussian, oriented):
    """3×3が画像内に収まり、中央の勾配が見やすいキーポイントを1点選ぶ。"""
    selected = None
    largest_magnitude = -1.0
    for point in oriented:
        if point["orientation_rank"] != 1:
            continue
        octave_index = point["octave"] - FIRST_OCTAVE
        levels = gaussian[octave_index][1]
        level = int(np.floor(point["level_float"] + 0.5))
        image = levels[level]
        center_x = int(np.floor(point["x_octave"] + 0.5))
        center_y = int(np.floor(point["y_octave"] + 0.5))
        height, width = image.shape

        # 3×3の端の画素の勾配には、その外側の画素も必要（合計5×5）。
        if center_x < 2 or center_x >= width - 2 or center_y < 2 or center_y >= height - 2:
            continue
        gx, gy, magnitude, angle = image_gradient(image, center_y, center_x)
        if magnitude > largest_magnitude:
            largest_magnitude = magnitude
            selected = point
    return selected


def save_gradient_detail(stem, color_image, gaussian, oriented):
    """1点の3×3の勾配と、周辺領域からの主方向計算を6パネルで可視化する。"""
    # バッチ実行・GUIのない環境でもPNGを保存できるようにする。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    point = select_gradient_example(gaussian, oriented)
    if point is None:
        print(f"{stem}: 3×3の詳細図を作れるキーポイントがありません。", flush=True)
        return

    levels = gaussian[point["octave"] - FIRST_OCTAVE][1]
    level = int(np.floor(point["level_float"] + 0.5))
    image = levels[level]
    center_x = int(np.floor(point["x_octave"] + 0.5))
    center_y = int(np.floor(point["y_octave"] + 0.5))
    patch = image[center_y - 1:center_y + 2, center_x - 1:center_x + 2]

    # 9画素の勾配。すべて実際の主方向計算と同じimage_gradientを使う。
    gx_patch = np.zeros((3, 3))
    gy_patch = np.zeros((3, 3))
    magnitudes = np.zeros((3, 3))
    angles_deg = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            y = center_y + row - 1
            x = center_x + column - 1
            gx, gy, magnitude, angle = image_gradient(image, y, x)
            gx_patch[row, column] = gx
            gy_patch[row, column] = gy
            magnitudes[row, column] = magnitude
            angles_deg[row, column] = np.rad2deg(angle) % 360.0

    # 3×3からの投票も、全領域と同じガウシアン重み・方向補間で集計する。
    full_histogram = smooth_orientation_histogram(orientation_histogram(image, point))
    bounds = (center_x - 1, center_y - 1, center_x + 1, center_y + 1)
    patch_histogram = orientation_histogram(image, point, sample_bounds=bounds)
    patch_histogram = smooth_orientation_histogram(patch_histogram)
    peaks = find_orientation_peaks(full_histogram)

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    figure.suptitle(
        f"{stem} | keypoint o={point['octave']}, id={point['keypoint_id']} | "
        f"Gaussian level {level}, sigma={point['sigma']:.3f} original pixels",
        fontsize=15,
    )

    # A. 元画像上の位置。
    axis = axes[0, 0]
    axis.imshow(color_image)
    axis.scatter(point["x"], point["y"], marker="+", color="red", s=150, linewidths=2)
    axis.add_patch(Circle((point["x"], point["y"]), max(7, 3 * point["sigma"]),
                          fill=False, edgecolor="red", linewidth=1.5))
    axis.set_title("A. Selected keypoint in the input image")
    axis.set_xlabel(f"Original (x, y) = ({point['x']:.3f}, {point['y']:.3f})")
    axis.set_ylabel("y increases downward")

    # B. 主方向の投票に使うガウシアン画像の領域と、3×3の位置。
    axis = axes[0, 1]
    sigma_octave = point["sigma"] / (2.0**point["octave"])
    window_radius = 3.0 * ORIENTATION_WINDOW_FACTOR * sigma_octave
    x0 = max(0, int(np.floor(point["x_octave"] - window_radius - 2)))
    y0 = max(0, int(np.floor(point["y_octave"] - window_radius - 2)))
    x1 = min(image.shape[1], int(np.ceil(point["x_octave"] + window_radius + 3)))
    y1 = min(image.shape[0], int(np.ceil(point["y_octave"] + window_radius + 3)))
    axis.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1,
                extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation="nearest")
    axis.add_patch(Circle((point["x_octave"], point["y_octave"]), window_radius,
                          fill=False, edgecolor="lime", linestyle="--", linewidth=1.5))
    axis.add_patch(Rectangle((center_x - 1.5, center_y - 1.5), 3, 3,
                             fill=False, edgecolor="red", linewidth=2))
    axis.scatter(point["x_octave"], point["y_octave"], marker="+", color="magenta", s=80)
    axis.set_title("B. Gaussian image used for orientation")
    axis.set_xlabel("Green circle: full voting window / red box: 3 x 3")
    axis.set_ylabel("Pixels in the selected octave")

    # C. 中央の角度を計算するための、3×3の輝度値。
    axis = axes[0, 2]
    axis.imshow(patch, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for row in range(3):
        for column in range(3):
            color = "black"
            if patch[row, column] < 0.5:
                color = "white"
            axis.text(column, row, f"{patch[row, column]:.4f}", ha="center", va="center",
                      color=color, fontsize=12)
    axis.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
    axis.scatter(point["x_octave"] - center_x + 1, point["y_octave"] - center_y + 1,
                 marker="+", color="magenta", s=100)
    axis.set_title("C. Gaussian intensities (float, range 0 to 1)")
    axis.set_xticks([0, 1, 2], labels=[str(center_x - 1), str(center_x), str(center_x + 1)])
    axis.set_yticks([0, 1, 2], labels=[str(center_y - 1), str(center_y), str(center_y + 1)])
    axis.set_xlabel("Red cell: center sample / magenta cross: subpixel keypoint")

    # D. 9画素それぞれの勾配。矢印の長さはこの3×3内の最大値でそろえる。
    axis = axes[1, 0]
    axis.imshow(np.ones((3, 3)), cmap="gray", vmin=0, vmax=1)
    max_magnitude = max(float(np.max(magnitudes)), np.finfo(float).eps)
    for row in range(3):
        for column in range(3):
            dx = 0.34 * gx_patch[row, column] / max_magnitude
            dy = 0.34 * gy_patch[row, column] / max_magnitude
            angle_text = f"{angles_deg[row, column]:.1f} deg"
            if magnitudes[row, column] == 0:
                angle_text = "undefined"
            else:
                axis.annotate("", xy=(column + dx, row - 0.12 + dy),
                              xytext=(column, row - 0.12),
                              arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 2})
            axis.text(column, row + 0.27,
                      f"m={magnitudes[row, column]:.4f}\n{angle_text}",
                      ha="center", va="center", fontsize=9)
    axis.set_xticks([-0.5, 0.5, 1.5, 2.5])
    axis.set_yticks([-0.5, 0.5, 1.5, 2.5])
    axis.grid(color="0.75")
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    axis.set_title("D. Gradient at each of the 3 x 3 samples")
    axis.set_xlabel("Arrow length is proportional to magnitude m; 90 deg points down")

    # E. 中央画素について、輝度差から角度までを数値で追う。
    axis = axes[1, 1]
    axis.axis("off")
    axis.set_title("E. Center sample calculation")
    center_gx = gx_patch[1, 1]
    center_gy = gy_patch[1, 1]
    formula = (
        "gx = L(right) - L(left)\n"
        f"   = {patch[1, 2]:.6f} - {patch[1, 0]:.6f}\n"
        f"   = {center_gx:+.6f}\n\n"
        "gy = L(down) - L(up)\n"
        f"   = {patch[2, 1]:.6f} - {patch[0, 1]:.6f}\n"
        f"   = {center_gy:+.6f}\n\n"
        f"m = sqrt(gx^2 + gy^2) = {magnitudes[1, 1]:.6f}\n"
        f"angle = atan2(gy, gx) = {angles_deg[1, 1]:.2f} deg\n\n"
        "This is one gradient vote.\n"
        "The main orientation uses the full window."
    )
    axis.text(0.02, 0.98, formula, va="top", family="monospace", fontsize=11,
              transform=axis.transAxes)

    # F. 実際の主方向ヒストグラムと、3×3の投票が占める部分を同じ尺度で比較。
    axis = axes[1, 2]
    bin_angles = np.arange(ORIENTATION_BINS + 1) * (360.0 / ORIENTATION_BINS)
    full_curve = np.append(full_histogram, full_histogram[0])
    patch_curve = np.append(patch_histogram, patch_histogram[0])
    axis.plot(bin_angles, full_curve, "o-", markersize=3, label="Full window (smoothed)")
    axis.plot(bin_angles, patch_curve, "o-", markersize=3, color="tab:orange",
              label="3 x 3 contribution (smoothed)")
    axis.axhline(ORIENTATION_PEAK_RATIO * np.max(full_histogram), color="0.5",
                 linestyle="--", label="80% peak threshold")
    for rank, (peak_height, peak_angle) in enumerate(peaks, start=1):
        angle_degrees = float(np.rad2deg(peak_angle))
        axis.axvline(angle_degrees, color="tab:green", linestyle=":")
        axis.annotate(f"{angle_degrees:.1f} deg", xy=(angle_degrees, peak_height),
                      xytext=(0, 8 + 12 * (rank - 1)), textcoords="offset points",
                      ha="center", fontsize=9, color="tab:green")
    axis.set_xlim(0, 360)
    axis.set_ylim(bottom=0, top=float(np.max(full_histogram)) * 1.35)
    axis.set_xticks([0, 90, 180, 270, 360])
    axis.set_title("F. Main orientation from all 36 bins")
    axis.set_xlabel("Clockwise angle (degrees)")
    axis.set_ylabel("Weighted gradient vote")
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)

    path = ORIENTATION_DIR / f"{stem}_orientation_detail.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path.name}", flush=True)
    print(
        f"  詳細図の点  octave={point['octave']}, 点番号={point['keypoint_id']}, "
        f"Gaussian level={level}, 3×3の中心画素=({center_x}, {center_y})"
    )
    print(f"  中央の勾配  gx={center_gx:+.6f}, gy={center_gy:+.6f}, "
          f"強さ={magnitudes[1, 1]:.6f}, 角度={angles_deg[1, 1]:.2f}度")
    angle_texts = []
    for peak_height, peak_angle in peaks:
        angle_texts.append(f"{np.rad2deg(peak_angle):.2f}度")
    print("  周辺領域全体で決めた主方向  " + ", ".join(angle_texts))


def descriptor_histogram(gaussian_levels, keypoint):
    """主方向を基準に、4×4×8の生のヒストグラムをVLFeat順で作る。"""
    spatial_bins = DESCRIPTOR_SPATIAL_BINS
    angle_bins = DESCRIPTOR_ORIENTATION_BINS
    histogram = np.zeros((spatial_bins, spatial_bins, angle_bins), dtype=np.float64)

    # VLFeatは、位置補正の最後に使った整数レベルの勾配を参照する。
    image = gaussian_levels[keypoint["level"]]
    height, width = image.shape
    center_x = keypoint["x_octave"]
    center_y = keypoint["y_octave"]
    sigma_octave = keypoint["sigma"] / (2.0**keypoint["octave"])
    bin_width = DESCRIPTOR_MAGNIFICATION * sigma_octave + np.finfo(np.float64).eps
    angle0 = keypoint["orientation_rad"]
    cosine = np.cos(angle0)
    sine = np.sin(angle0)

    # 4区画の外へ半区画分はみ出す勾配も、線形補間で端の区画へ投票する。
    # 回転した5区画幅の領域を包む正方形を調べる。
    radius = int(np.floor(np.sqrt(2.0) * bin_width * (spatial_bins + 1) / 2.0 + 0.5))
    sample_center_x = int(np.floor(center_x + 0.5))
    sample_center_y = int(np.floor(center_y + 0.5))
    first_x = max(1, sample_center_x - radius)
    last_x = min(width - 2, sample_center_x + radius)
    first_y = max(1, sample_center_y - radius)
    last_y = min(height - 2, sample_center_y + radius)

    for y in range(first_y, last_y + 1):
        for x in range(first_x, last_x + 1):
            dx = x - center_x
            dy = y - center_y

            # 主方向を横軸へそろえ、1区画の幅を1とする座標へ変換する。
            u = (cosine * dx + sine * dy) / bin_width
            v = (-sine * dx + cosine * dy) / bin_width
            bin_x_position = u + (spatial_bins - 1) / 2.0
            bin_y_position = v + (spatial_bins - 1) / 2.0
            lower_x = int(np.floor(bin_x_position))
            lower_y = int(np.floor(bin_y_position))
            if lower_x < -1 or lower_x >= spatial_bins or lower_y < -1 or lower_y >= spatial_bins:
                continue
            fraction_x = bin_x_position - lower_x
            fraction_y = bin_y_position - lower_y

            gx, gy, magnitude, angle = image_gradient(image, y, x)
            # VLFeatの勾配は中央差分の1/2。角度は同じで、強さだけ半分になる。
            magnitude *= 0.5
            if magnitude == 0:
                continue
            relative_angle = (angle - angle0) % (2.0 * np.pi)
            bin_angle_position = relative_angle * angle_bins / (2.0 * np.pi)
            lower_angle = int(np.floor(bin_angle_position))
            fraction_angle = bin_angle_position - lower_angle
            gaussian_weight = np.exp(-(u**2 + v**2) / (2.0 * DESCRIPTOR_WINDOW_SIGMA**2))
            weighted_magnitude = gaussian_weight * magnitude

            # x方向2区画×y方向2区画×角度方向2ビンの、合計8か所へ投票する。
            for x_neighbor in range(2):
                column = lower_x + x_neighbor
                if column < 0 or column >= spatial_bins:
                    continue
                weight_x = abs(1.0 - x_neighbor - fraction_x)
                for y_neighbor in range(2):
                    row = lower_y + y_neighbor
                    if row < 0 or row >= spatial_bins:
                        continue
                    weight_y = abs(1.0 - y_neighbor - fraction_y)
                    for angle_neighbor in range(2):
                        angle_bin = (lower_angle + angle_neighbor) % angle_bins
                        weight_angle = abs(1.0 - angle_neighbor - fraction_angle)
                        vote = weighted_magnitude * weight_x * weight_y * weight_angle
                        histogram[row, column, angle_bin] += vote

    return histogram


def vlfeat_to_colmap_order(vector):
    """各区画の方向を、COLMAPのTransformVLFeatToUBCと同じ順に並べ替える。"""
    result = np.zeros_like(vector)
    angle_order = (0, 7, 6, 5, 4, 3, 2, 1)
    for row in range(DESCRIPTOR_SPATIAL_BINS):
        for column in range(DESCRIPTOR_SPATIAL_BINS):
            start = 8 * (4 * row + column)
            for angle_bin in range(DESCRIPTOR_ORIENTATION_BINS):
                result[start + angle_order[angle_bin]] = vector[start + angle_bin]
    return result


def normalize_descriptor(histogram):
    """VLFeatの標準SIFTから、COLMAP既定のRootSIFTと保存用uint8を作る。"""
    raw = histogram.reshape(-1).copy()  # 方向→列→行の順で128要素を並べる。
    epsilon = np.finfo(np.float32).eps

    # VLFeat内での正規化。最初にL2正規化し、0.2で打ち切り、再びL2正規化する。
    first_l2 = raw / (np.linalg.norm(raw) + epsilon)
    clipped = np.minimum(first_l2, DESCRIPTOR_CLIP)
    standard_sift = clipped / (np.linalg.norm(clipped) + epsilon)

    # COLMAPの既定設定 L1_ROOT。要素の合計を1にして、各要素の平方根を取る。
    rootsift = standard_sift.copy()
    l1_norm = float(np.sum(rootsift))
    if l1_norm > 0:
        rootsift = np.sqrt(rootsift / l1_norm)

    # FeatureDescriptorsToUnsignedByteに合わせ、正の値を四捨五入する。
    # np.rintの「ちょうど半分なら偶数へ丸める」規約は、ここでは使わない。
    scaled = (rootsift.astype(np.float32) * np.float32(512.0)).astype(np.float64)
    unsigned_byte = np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)

    # CPU版の最後の並べ替え。空間区画の順は同じで、方向ビンだけを反転する。
    return {
        "descriptor_raw": vlfeat_to_colmap_order(raw),
        "descriptor_sift": vlfeat_to_colmap_order(standard_sift),
        "descriptor": vlfeat_to_colmap_order(rootsift).astype(np.float32),
        "descriptor_uint8": vlfeat_to_colmap_order(unsigned_byte),
    }


def compute_descriptors(gaussian_levels, oriented):
    """向き付きの全キーポイントについて、128次元記述子を計算する。"""
    described = []
    for keypoint in oriented:
        histogram = descriptor_histogram(gaussian_levels, keypoint)
        point = keypoint.copy()
        point.update(normalize_descriptor(histogram))
        described.append(point)
    return described


def save_descriptors(stem, described):
    """次の対応付けに使えるよう、同じ行順のキーポイントと記述子をnpzへ保存する。"""
    count = len(described)
    keypoints = np.zeros((count, 4), dtype=np.float64)
    descriptors = np.zeros((count, 128), dtype=np.float32)
    descriptors_uint8 = np.zeros((count, 128), dtype=np.uint8)
    for index, point in enumerate(described):
        keypoints[index] = [point["x"], point["y"], point["sigma"], point["orientation_rad"]]
        descriptors[index] = point["descriptor"]
        descriptors_uint8[index] = point["descriptor_uint8"]

    path = DESCRIPTOR_DIR / f"{stem}_descriptors.npz"
    np.savez_compressed(
        path,
        keypoints=keypoints,
        keypoint_columns=np.array(["x", "y", "sigma", "orientation_rad"]),
        descriptors=descriptors,
        descriptors_uint8=descriptors_uint8,
        coordinate_convention="original image, pixel centers start at (0,0)",
        descriptor_order="COLMAP UBC, row-column-direction, 4x4x8",
        normalization="RootSIFT (L1_ROOT)",
    )
    print(f"Saved: {path.name} | 記述子の形状={descriptors.shape}, dtype=float32 / uint8", flush=True)
    if count > 0:
        norms = np.linalg.norm(descriptors, axis=1)
        print(f"  RootSIFTのL2ノルム {np.min(norms):.6f}～{np.max(norms):.6f}")
        print(f"  保存用uint8の範囲 {np.min(descriptors_uint8)}～{np.max(descriptors_uint8)}")


def save_descriptor_detail(stem, color_image, gaussian, described):
    """選んだ1点の4×4×8ヒストグラムと、128要素への変換を可視化する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    point = select_gradient_example(gaussian, described)
    if point is None:
        print(f"{stem}: 記述子の詳細図を作れるキーポイントがありません。")
        return
    levels = gaussian[point["octave"] - FIRST_OCTAVE][1]
    image = levels[point["level"]]
    cosine = np.cos(point["orientation_rad"])
    sine = np.sin(point["orientation_rad"])
    bin_width_original = DESCRIPTOR_MAGNIFICATION * point["sigma"]
    bin_width_octave = bin_width_original / (2.0**point["octave"])
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    figure.suptitle(
        f"{stem} | keypoint o={point['octave']}, id={point['keypoint_id']} | "
        f"orientation={point['orientation_deg']:.2f} deg | 4 x 4 x 8 = 128",
        fontsize=15,
    )

    # A. 元画像に、主方向に合わせて回転した4×4の区画を重ねる。
    axis = axes[0, 0]
    axis.imshow(color_image)
    for boundary in range(-2, 3):
        for horizontal in (True, False):
            u = np.array([-2.0, 2.0])
            v = np.array([float(boundary), float(boundary)])
            if not horizontal:
                u, v = v, u
            x = point["x"] + bin_width_original * (cosine * u - sine * v)
            y = point["y"] + bin_width_original * (sine * u + cosine * v)
            axis.plot(x, y, color="lime", linewidth=1)
    axis.scatter(point["x"], point["y"], color="red", marker="+", s=90)
    axis.set_title("A. Descriptor grid in the input image")
    axis.set_xlabel(f"One spatial bin = 3 sigma = {bin_width_original:.3f} original pixels")

    # B. 表示用にだけ画像を主方向へ回転する。集計自体は元の画素上で行う。
    axis = axes[0, 1]
    coordinates = np.linspace(-2.5, 2.5, 180)
    u, v = np.meshgrid(coordinates, coordinates)
    sample_x = point["x_octave"] + bin_width_octave * (cosine * u - sine * v)
    sample_y = point["y_octave"] + bin_width_octave * (sine * u + cosine * v)
    patch = map_coordinates(image, [sample_y, sample_x], order=1, mode="constant", cval=np.nan)
    axis.imshow(patch, cmap="gray", vmin=0, vmax=1, extent=(-2.5, 2.5, 2.5, -2.5))
    for boundary in range(-2, 3):
        axis.plot([-2, 2], [boundary, boundary], color="lime", linewidth=1)
        axis.plot([boundary, boundary], [-2, 2], color="lime", linewidth=1)
    axis.annotate("main direction", xy=(1.5, 0), xytext=(0, 0), color="red",
                  arrowprops={"arrowstyle": "->", "color": "red"}, fontsize=9)
    axis.set_title("B. Local frame aligned to the main direction")
    axis.set_xlabel("Spatial-bin units; outer half-bin also contributes")
    axis.set_ylabel("Local y increases downward")

    # C. 16区画のそれぞれに、8方向の勾配の集計を矢印で描く。
    axis = axes[0, 2]
    histograms = point["descriptor_raw"].reshape(4, 4, 8)
    largest = max(float(np.max(histograms)), np.finfo(float).eps)
    for row in range(4):
        for column in range(4):
            for direction in range(8):
                # COLMAPの保存順では、方向ビンは局所座標の反時計回りに増える。
                angle = -2.0 * np.pi * direction / 8
                length = 0.43 * histograms[row, column, direction] / largest
                if length == 0:
                    continue
                axis.arrow(column, row, length * np.cos(angle), length * np.sin(angle),
                           width=0.006, head_width=min(0.05, length * 0.3),
                           head_length=min(0.06, length * 0.35),
                           length_includes_head=True, color="tab:blue")
            start = 8 * (4 * row + column)
            axis.text(column - 0.45, row - 0.39, f"{start}-{start + 7}", fontsize=8, color="0.4")
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(3.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(5) - 0.5)
    axis.set_yticks(np.arange(5) - 0.5)
    axis.grid(color="0.8")
    axis.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axis.set_title("C. 16 cells, each with 8 raw gradient directions")
    axis.set_xlabel("Numbers show the 128-vector index range in each cell")

    indices = np.arange(128)
    colors = plt.get_cmap("tab20")(np.repeat(np.arange(16), 8) / 19.0)
    axis = axes[1, 0]
    axis.bar(indices, point["descriptor_raw"], color=colors, width=1)
    axis.set_title("D. Concatenate 16 histograms: 128 raw values")
    axis.set_ylabel("Weighted gradient vote")

    axis = axes[1, 1]
    axis.plot(indices, point["descriptor_sift"], label="SIFT: L2 / clip at 0.2 / L2", linewidth=1)
    axis.plot(indices, point["descriptor"], label="RootSIFT: L1 / square root", linewidth=1)
    axis.set_title("E. Normalized 128-dimensional descriptors")
    axis.set_ylabel("Float value")
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    axis.bar(indices, point["descriptor_uint8"], color=colors, width=1)
    axis.set_title("F. COLMAP storage: round(512 x RootSIFT), uint8")
    axis.set_ylabel("Unsigned byte value")
    axis.set_ylim(0, 255)
    axis.set_yticks([0, 64, 128, 192, 255])

    for axis in axes[1]:
        axis.set_xlim(-0.5, 127.5)
        axis.set_xticks([0, 32, 64, 96, 127])
        axis.set_xlabel("Index = 8 x (4 x row + column) + direction bin")
        axis.grid(axis="y", alpha=0.2)
        for boundary in range(1, 16):
            axis.axvline(8 * boundary - 0.5, color="0.85", linewidth=0.5)

    path = DESCRIPTOR_DIR / f"{stem}_descriptor_detail.png"
    # PNG全体をメモリ上で完成させてから、ファイルへ書き出す。
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=150)
        path.write_bytes(buffer.getvalue())
    plt.close(figure)
    print(f"Saved: {path.name}", flush=True)
    print(f"  可視化した記述子 octave={point['octave']}, 点番号={point['keypoint_id']}, "
          f"主方向={point['orientation_deg']:.2f}度")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOG_DIR.mkdir(exist_ok=True)
    KEYPOINT_DIR.mkdir(exist_ok=True)
    REFINED_DIR.mkdir(exist_ok=True)
    ORIENTATION_DIR.mkdir(exist_ok=True)
    DESCRIPTOR_DIR.mkdir(exist_ok=True)
    print(f"低コントラストのしきい値 = {CONTRAST_THRESHOLD}（元の画素値は0～1）")
    print("補正例の整数座標は各オクターブ上の画素位置。sはDoGのレベル番号です。")
    print("元画像の座標は画素中心を0始まりで表し、拡大・縮小分を換算しています。")

    for name in INPUT_NAMES:
        path = INPUT_DIR / name
        with Image.open(path) as image:
            color_image = image.convert("RGB")
            gray = np.asarray(image.convert("L"))
        save_png(OUTPUT_DIR / f"{path.stem}_gray.png", gray)

        all_candidates = []
        all_kept = []
        all_weak = []
        all_failed = []
        all_oriented = []
        all_described = []
        total_multiple = 0
        total_unassigned = 0
        gaussian = gaussian_pyramid(gray)

        for octave, levels in gaussian:
            for level, image in enumerate(levels):
                preview = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
                filename = f"{path.stem}_gaussian_o{octave}_l{level:02d}.png"
                save_png(OUTPUT_DIR / filename, preview)

            dog_images = difference_of_gaussians(levels)
            for level, dog in enumerate(dog_images):
                filename = f"{path.stem}_dog_o{octave}_l{level:02d}.png"
                save_png(DOG_DIR / filename, dog_preview(dog))

            print(f"Finding candidates: {name}, octave {octave}", flush=True)
            candidates = find_keypoint_candidates(dog_images, octave)
            kept, weak, failed = refine_candidates(dog_images, candidates)
            print_candidate_counts(f"{name} / octave {octave}", candidates, kept, weak, failed)

            oriented, multiple_count, unassigned_count = assign_orientations(levels, kept)
            described = compute_descriptors(levels, oriented)
            all_described.extend(described)
            print_orientation_summary(
                f"{name} / octave {octave}", kept, oriented, multiple_count, unassigned_count
            )
            all_oriented.extend(oriented)
            total_multiple += multiple_count
            total_unassigned += unassigned_count

            all_candidates.extend(candidates)
            all_kept.extend(kept)
            all_weak.extend(weak)
            all_failed.extend(failed)

            stem = f"{path.stem}_keypoints_o{octave}"
            save_keypoint_preview(KEYPOINT_DIR / f"{stem}.png", color_image, candidates)
            save_refinement_preview(stem, color_image, candidates, kept, weak, failed)
            filename = f"{path.stem}_orientations_o{octave}.png"
            save_orientation_preview(ORIENTATION_DIR / filename, color_image, oriented)

        stem = f"{path.stem}_keypoints_all"
        save_keypoint_preview(KEYPOINT_DIR / f"{stem}.png", color_image, all_candidates)
        save_refinement_preview(stem, color_image, all_candidates, all_kept, all_weak, all_failed)
        print_candidate_counts(f"{name} / 全オクターブ合計", all_candidates, all_kept, all_weak, all_failed)
        print_coordinate_examples("採用候補", all_kept)
        print_coordinate_examples("低コントラストで除去した候補", all_weak)
        filename = f"{path.stem}_orientations_all.png"
        save_orientation_preview(ORIENTATION_DIR / filename, color_image, all_oriented)
        print_orientation_summary(
            f"{name} / 全オクターブ合計", all_kept, all_oriented, total_multiple, total_unassigned
        )
        print_orientation_examples(all_oriented)
        save_gradient_detail(path.stem, color_image, gaussian, all_oriented)
        save_descriptors(path.stem, all_described)
        save_descriptor_detail(path.stem, color_image, gaussian, all_described)


if __name__ == "__main__":
    main()


