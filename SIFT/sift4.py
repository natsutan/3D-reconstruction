"""DoGのキーポイント候補を二次式で補正し、低コントラスト候補を除く。

準備: python -m pip install numpy pillow scipy
実行: python sift5.py
入力: スクリプトと同じ場所の output フォルダー
出力: output/gaussian、output/dog、output/keypoints、output/keypoints_refined

候補の検出は、浮動小数点のDoG画像で周囲26点と比較する。
円の半径はその候補のスケールσ（表示時は最低2 px）。
位置とスケール方向の3変数 (x, y, s) を二次式で補正する。
sはDoGのレベル番号。元画像でのσは
BASE_SIGMA * 2**(s / SCALES_PER_OCTAVE) * 2**octave で求める。
補正後の応答 |D| が CONTRAST_THRESHOLD 未満の候補を除去する。
既定の0.03は、画素値を0～1に正規化したときのLowe論文の実験値。
今回の範囲は二次式の補正と低コントラスト除去。エッジ判定は後続の段階。
標準出力には各オクターブの候補数と、代表点の補正前後の座標を表示する。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = INPUT_DIR / "gaussian"
DOG_DIR = INPUT_DIR / "dog"
KEYPOINT_DIR = INPUT_DIR / "keypoints"
REFINED_DIR = INPUT_DIR / "keypoints_refined"
INPUT_NAMES = ("view_01_wood.png", "view_02_wood.png")

FIRST_OCTAVE = -1
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 3
BASE_SIGMA = 1.6
INPUT_SIGMA = 0.5  # 元画像がすでに持つと仮定するぼかし

CONTRAST_THRESHOLD = 0.03
MAX_REFINEMENT_STEPS = 5
NUM_COORDINATE_EXAMPLES = 3  # 採用・低コントラスト除去の各グループから表示する数


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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOG_DIR.mkdir(exist_ok=True)
    KEYPOINT_DIR.mkdir(exist_ok=True)
    REFINED_DIR.mkdir(exist_ok=True)
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

            all_candidates.extend(candidates)
            all_kept.extend(kept)
            all_weak.extend(weak)
            all_failed.extend(failed)

            stem = f"{path.stem}_keypoints_o{octave}"
            save_keypoint_preview(KEYPOINT_DIR / f"{stem}.png", color_image, candidates)
            save_refinement_preview(stem, color_image, candidates, kept, weak, failed)

        stem = f"{path.stem}_keypoints_all"
        save_keypoint_preview(KEYPOINT_DIR / f"{stem}.png", color_image, all_candidates)
        save_refinement_preview(stem, color_image, all_candidates, all_kept, all_weak, all_failed)
        print_candidate_counts(f"{name} / 全オクターブ合計", all_candidates, all_kept, all_weak, all_failed)
        print_coordinate_examples("採用候補", all_kept)
        print_coordinate_examples("低コントラストで除去した候補", all_weak)


if __name__ == "__main__":
    main()


