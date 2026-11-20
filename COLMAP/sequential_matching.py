"""Match SIFT features in a sequence of frames and optionally draw the matches."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FEATURES = 2000
RATIO_THRESHOLD = 0.75
COLMAP_DIR = Path(__file__).resolve().parent
CSV_DIR = COLMAP_DIR / "output" / "matching_first"
VISUALIZATION_DIR = COLMAP_DIR.parent / "work" / "matching"


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


def match_features(frames, pairs, csv_output, visualization_output):
    import cv2
    import numpy as np

    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    descriptors = {}
    color_images = {}
    image_keypoints = {}
    with (csv_output / "keypoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "keypoint_index", "x", "y"])
        for frame in frames:
            # imdecode supports paths with Japanese characters on Windows.
            encoded = np.fromfile(frame, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None
            if image is None:
                raise ValueError(f"Cannot read image: {frame}")
            keypoints, descriptors[frame] = sift.detectAndCompute(image, None)
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


def publish_files(staging, output):
    output.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        path.replace(output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory of zero-padded image files")
    parser.add_argument("--start", type=int, default=1, help="Starting position, 1-based")
    parser.add_argument(
        "--count", type=int,
        help="Maximum number of images (default: all images from --start)",
    )
    parser.add_argument("--visualize", action="store_true", help="Save one match image per pair")
    parser.add_argument(
        "--first-only", action="store_true",
        help="Use only the first selected frame as the reference image",
    )
    args = parser.parse_args()
    if args.start < 1 or (args.count is not None and args.count < 2):
        parser.error("--start must be at least 1 and --count at least 2")

    try:
        frames = select_frames(args.image_dir, args.start, args.count)
        pairs = select_pairs(frames, first_only=args.first_only)
        if args.first_only:
            frames = [frames[0]]
            for _, second, _ in pairs:
                frames.append(second)
        with TemporaryDirectory(dir=COLMAP_DIR) as temporary:
            staging = Path(temporary)
            csv_output = staging / "csv"
            csv_output.mkdir()
            visualization_output = None
            if args.visualize:
                visualization_output = staging / "visualizations"
                visualization_output.mkdir()

            write_pairs(csv_output / "pairs.csv", pairs)
            pair_counts = match_features(frames, pairs, csv_output, visualization_output)
            publish_files(csv_output, CSV_DIR)
            if visualization_output is not None:
                publish_files(visualization_output, VISUALIZATION_DIR)
    except ModuleNotFoundError as error:
        parser.exit(1, f"{error}. Install OpenCV with: python -m pip install opencv-python-headless\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Selected {len(frames)} images: {frames[0].name} ... {frames[-1].name}")
    print(f"Selected {len(pairs)} image pairs")
    if args.first_only:
        for (first, second, _), count in zip(pairs, pair_counts):
            print(f"{first.name} -> {second.name}: {count} match candidates")
    print(f"SIFT match candidates after the ratio test: {sum(pair_counts)}")
    print(f"CSV output: {CSV_DIR}")
    if args.visualize:
        print(f"Image output: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
