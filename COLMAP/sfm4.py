"""sfm3.pyの最終結果を、画像付きのCOLMAPテキストモデルとして出力する。

配置: COLMAP/sfm4.py
実行: python COLMAP/sfm4.py
依存: NumPy, Pillow。CPUのみ。sfm3.pyの事前実行が必要。
入力画像フォルダの指定: python COLMAP/sfm4.py --image-dir COLMAP/frames
モデルの指定: python COLMAP/sfm4.py --model COLMAP/output/sfm3/reconstruction.npz

入力（既定のパスは、このファイルのあるCOLMAPディレクトリが基準）:
    output/sfm3/reconstruction.npz: BA・フィルタリング後の復元結果
    frames/: SfMに用いたものと同じ画像（同じファイル名・解像度）
    旧ファイル名で作ったoutput/fsm3/reconstruction.npzも自動検出する。
    両方があればoutput/sfm3を優先する。必要なら--modelで明示する。

出力: output/colmap/
    images/: モデルに登録された画像だけをコピー
    sparse/0/cameras.txt: 共有する内部パラメータ（一行）
    sparse/0/images.txt: 画像ごとの姿勢と、採用済みの二次元観測
    sparse/0/points3D.txt: 三次元点・RGB・平均再投影誤差・トラック
出力先はエクスポート専用。再実行時はこのディレクトリ全体を更新し、
古い.binやpoints3D.plyなどが新しい.txtより優先されることを防ぐ。
入力のNPZや画像、sfm3.py自体は変更しない。

変換の対応:
    1. 入力: NPZと元画像を読み込む。
    2. 変換: Kの主点と観測(x,y)に+0.5し、OpenCV規約からCOLMAP規約へ戻す。
       RをHamiltonのクォータニオン(QW,QX,QY,QZ)にする。tはそのまま。
       X_camera = R @ X_world + t。カメラ中心C=-R.T@tをtの代わりに書かない。
       CAMERA_ID=1、IMAGE_ID=登録順+1、POINT3D_ID=元のpoint_id+1。
       POINT2D_IDXは画像ごとの観測を元の特徴点番号順に並べ、0から付け直す。
       点の色はOpenCV規約の観測位置で元画像を双一次補間し、観測間で平均する。
       ERRORは、最終姿勢・点から計算した再投影距離の算術平均（px）。
    3. 出力: 変換済みデータだけを書き出す。姿勢推定やBAは行わない。

SIMPLE_PINHOLE（fx==fy）またはPINHOLE（fx!=fy）で出力する。
レンズ歪みを無視する、これまでのSfMの前提を引き継ぐ。
database.db、rigs.txt、frames.txt、バイナリ形式は作らない。

仕様: https://colmap.github.io/format.html
座標: https://colmap.github.io/database.html#keypoints-and-descriptors
"""

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "sfm3" / "reconstruction.npz"
LEGACY_MODEL_PATH = BASE_DIR / "output" / "fsm3" / "reconstruction.npz"
IMAGE_DIR = BASE_DIR / "frames"
OUTPUT_DIR = BASE_DIR / "output" / "colmap"
DEPTH_EPSILON = 1e-9


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


def load_inputs(model_path=None, image_dir=IMAGE_DIR):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="sfm3.pyの最終reconstruction.npz")
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR,
                        help="SfMと同じ画像があるフォルダ。既定: COLMAP/frames")
    args = parser.parse_args()
    try:
        # 1. 入力
        model, image_paths, image_colors, source = load_inputs(args.model, args.image_dir)
        # 2. 形式と座標規約の変換
        exported = build_colmap_model(model, image_paths, image_colors)
        # 3. 出力
        write_results(exported)
        print(f"入力モデル: {source}")
        print(f"内部パラメータ: {exported.camera_model}、CAMERA_ID=1を全画像で共有")
        print(f"登録画像: {len(exported.images)}枚 / 三次元点: {len(exported.points)}点")
        for image in exported.images:
            print(f"  IMAGE_ID={image.image_id}: {image.name}、観測{len(image.point_ids)}点")
        print("主点と画像座標: OpenCV規約からCOLMAP規約へ+0.5")
        print("RGB: 元画像の観測位置の色を平均 / ERROR: 再投影距離の平均（px）")
        print(f"出力先: {OUTPUT_DIR}")
        print("画像: images/、復元モデル: sparse/0/ のテキスト三ファイル")
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
