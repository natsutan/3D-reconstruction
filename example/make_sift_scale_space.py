"""SIFTの最初の1オクターブのGaussian画像6枚とDoG画像5枚を保存する。

依存: NumPy, Pillow, SciPy
    python -m pip install numpy pillow scipy
実行:
    python make_sift_scale_space.py view_01_sift.png
    python make_sift_scale_space.py view_01_sift.png --output output/scale_space
    python make_sift_scale_space.py view_01_sift.png --no-upscale

既定では入力をグレースケール化し、縦横2倍に線形補間で拡大する。
入力に元からあるぼけをsigma=0.5と仮定し、最初の層の総sigmaを1.6にする。
これは最初の1オクターブを観察する教材用で、特徴点の検出までは行わない。

出力:
    gaussian_XX_sigma_....png / .npy: Gaussian画像6枚
    dog_XX_sigma_..._to_....png / .npy: 隣接Gaussian画像の差分5枚

PNGは閲覧用。Gaussianは[0,1]を[0,255]に変換する。
DoGは5枚共通の対称な表示範囲を使い、負=暗い、ゼロ=128、正=明るい。
NPYは丸める前のfloat64配列。後続の数値計算にはこちらを使う。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


BASE_SIGMA = 1.6
SCALES_PER_OCTAVE = 3
ASSUMED_INPUT_SIGMA = 0.5


def load_image(path, upscale=True):
    """入力を[0,1]のグレースケール配列にして、必要なら縦横2倍にする。"""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    factor = 2 if upscale else 1
    if upscale:
        # 浮動小数点画像のまま拡大し、8bitへの再丸めを避ける。
        image = Image.fromarray(gray.astype(np.float32))
        image = image.resize((image.width * 2, image.height * 2),
                             Image.Resampling.BILINEAR)
        gray = np.asarray(image, dtype=np.float64)
    return gray, ASSUMED_INPUT_SIGMA * factor


def make_gaussian_images(image, input_sigma):
    """総sigmaを等比数列にしてGaussian画像を作る。

    sigmaの単位は、このオクターブの画像上のピクセル。
    既定の2倍拡大時には、元画像上のsigmaはこの値の半分になる。
    """
    k = 2.0 ** (1.0 / SCALES_PER_OCTAVE)
    sigmas = [BASE_SIGMA * k**i for i in range(SCALES_PER_OCTAVE + 3)]
    if input_sigma > sigmas[0]:
        raise ValueError("入力の想定sigmaが最初の層のsigmaを超えています。")

    gaussian_images = []
    current = np.asarray(image, dtype=np.float64)
    current_sigma = input_sigma
    for target_sigma in sigmas:
        # Gaussianを重ねると分散が加算されるため、追加分だけをかける。
        additional_sigma = np.sqrt(max(0.0, target_sigma**2 - current_sigma**2))
        if additional_sigma > 0:
            current = gaussian_filter(current, sigma=additional_sigma,
                                      mode="reflect", truncate=4.0)
        gaussian_images.append(current.copy())
        current_sigma = target_sigma
    return sigmas, gaussian_images


def make_dog_images(gaussian_images):
    """D_i = L_(i+1) - L_i。浮動小数点のまま引き、符号を保持する。"""
    return [upper - lower
            for lower, upper in zip(gaussian_images[:-1], gaussian_images[1:])]


def dog_to_uint8(dog, display_limit):
    """5枚共通の表示範囲[-limit,+limit]を[0,255]へ写す。"""
    if display_limit <= 1e-12:
        return np.full(dog.shape, 128, dtype=np.uint8)
    normalized = np.clip(dog / display_limit, -1.0, 1.0)
    return np.rint((normalized + 1.0) * 127.5).astype(np.uint8)


def save_images(output_dir, sigmas, gaussian_images, dog_images):
    output_dir.mkdir(parents=True, exist_ok=True)
    # 画像ごとに明るさを自動調整すると、応答の強さを比較できなくなる。
    display_limit = max(float(np.max(np.abs(dog))) for dog in dog_images)

    for i, (sigma, image) in enumerate(zip(sigmas, gaussian_images)):
        stem = f"gaussian_{i:02d}_sigma_{sigma:.4f}"
        png = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(png).save(output_dir / f"{stem}.png")
        np.save(output_dir / f"{stem}.npy", image)
        print(f"Gaussian {i}: sigma={sigma:.6f} -> {stem}.png")

    for i, dog in enumerate(dog_images):
        stem = f"dog_{i:02d}_sigma_{sigmas[i]:.4f}_to_{sigmas[i+1]:.4f}"
        Image.fromarray(dog_to_uint8(dog, display_limit)).save(output_dir / f"{stem}.png")
        np.save(output_dir / f"{stem}.npy", dog)
        print(f"DoG {i}: L({sigmas[i+1]:.6f}) - L({sigmas[i]:.6f})"
              f" -> {stem}.png")

    print(f"DoG PNG共通の表示範囲: [{-display_limit:.6f}, {display_limit:.6f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="入力画像のパス")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "scale_space",
                        help="出力フォルダー（既定: ソースと同じ場所のoutput/scale_space）")
    parser.add_argument("--no-upscale", action="store_true",
                        help="入力画像の縦横2倍への拡大を省く")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"入力画像が見つかりません: {args.image}")

    image, input_sigma = load_image(args.image, upscale=not args.no_upscale)
    sigmas, gaussian_images = make_gaussian_images(image, input_sigma)
    dog_images = make_dog_images(gaussian_images)
    save_images(args.output, sigmas, gaussian_images, dog_images)
    print(f"画像サイズ: {image.shape[1]} x {image.shape[0]}")
    print(f"保存先: {args.output.resolve()}")


if __name__ == "__main__":
    main()
