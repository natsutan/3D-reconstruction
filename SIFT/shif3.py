"""木目画像2枚からDoGの極値を探し、キーポイント候補を可視化する。

準備: python -m pip install numpy pillow scipy
実行: python shif3.py
入力: スクリプトと同じ場所の output フォルダー
出力: output/gaussian、output/dog、output/keypoints フォルダー

候補の検出は、浮動小数点のDoG画像で周囲26点と比較する。
円の半径はその候補のスケールσ（表示時は最低2 px）。
位置・スケールの補正、低コントラスト除去、エッジ除去は後続の段階。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = INPUT_DIR / "gaussian"
DOG_DIR = INPUT_DIR / "dog"
KEYPOINT_DIR = INPUT_DIR / "keypoints"
INPUT_NAMES = ("view_01_wood.png", "view_02_wood.png")

FIRST_OCTAVE = -1
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 3
BASE_SIGMA = 1.6
INPUT_SIGMA = 0.5  # 元画像がすでに持つと仮定するぼかし


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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOG_DIR.mkdir(exist_ok=True)
    KEYPOINT_DIR.mkdir(exist_ok=True)

    for name in INPUT_NAMES:
        path = INPUT_DIR / name
        with Image.open(path) as image:
            color_image = image.convert("RGB")
            gray = np.asarray(image.convert("L"))
        save_png(OUTPUT_DIR / f"{path.stem}_gray.png", gray)

        all_candidates = []
        gaussian = gaussian_pyramid(gray)

        for octave, levels in gaussian:
            for level, image in enumerate(levels):
                # PNGは閲覧用。8ビットへの変換は保存するときだけ行う。
                preview = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
                filename = f"{path.stem}_gaussian_o{octave}_l{level:02d}.png"
                save_png(OUTPUT_DIR / filename, preview)

            dog_images = difference_of_gaussians(levels)
            for level, dog in enumerate(dog_images):
                filename = f"{path.stem}_dog_o{octave}_l{level:02d}.png"
                save_png(DOG_DIR / filename, dog_preview(dog))

            print(f"Finding candidates: {name}, octave {octave}", flush=True)
            candidates = find_keypoint_candidates(dog_images, octave)
            all_candidates.extend(candidates)

            filename = f"{path.stem}_keypoints_o{octave}.png"
            save_keypoint_preview(KEYPOINT_DIR / filename, color_image, candidates)

        filename = f"{path.stem}_keypoints_all.png"
        save_keypoint_preview(KEYPOINT_DIR / filename, color_image, all_candidates)


if __name__ == "__main__":
    main()


