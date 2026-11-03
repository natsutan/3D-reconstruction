"""木目画像2枚をグレースケール化し、SIFT用のガウシアン画像を保存する。

準備: python -m pip install numpy pillow scipy
実行: python sift1.py
入力: スクリプトと同じ場所の output フォルダー
出力: output/gaussian フォルダー
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


INPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = INPUT_DIR / "gaussian"
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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in INPUT_NAMES:
        path = INPUT_DIR / name
        with Image.open(path) as color:
            gray = np.asarray(color.convert("L"))
        save_png(OUTPUT_DIR / f"{path.stem}_gray.png", gray)

        for octave, levels in gaussian_pyramid(gray):
            for level, image in enumerate(levels):
                # PNGは閲覧用。8ビットへの変換は保存するときだけ行う。
                preview = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
                filename = f"{path.stem}_gaussian_o{octave}_l{level:02d}.png"
                save_png(OUTPUT_DIR / filename, preview)


if __name__ == "__main__":
    main()
