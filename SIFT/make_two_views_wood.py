"""木目付き三角錐を0°と15°から描き、output/にPNGを2枚保存する。


"""
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 640, 480
FOCAL_LENGTH_PX = 450.0  # 原点を向いても上の頂点まで収める
# COLMAPと同じ規約: 画像左上の外側の角が(0,0)、左上ピクセル中心が(0.5,0.5)。
# 前回の「ピクセル中心を整数で表す」規約から、主点も両軸に0.5 px加える。
PRINCIPAL_POINT_X_PX = WIDTH / 2
PRINCIPAL_POINT_Y_PX = HEIGHT / 2 + 100.0  # 投影位置を100 px下へ
# 以下の2値はカメラの配置を決める値（底面からの高さは約1.63956）。
# 基準はmake_scene()が返すorbit_center。原点への距離や俯角とは異なる。
CAMERA_DISTANCE = 5.0
ELEVATION_DEG = 12.0
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 底面の中心を向く
AZIMUTHS_DEG = (0.0, 15.0)  # カメラの方位角
OBJECT_ROTATION_DEG = 90.0  # どちらのカメラからも同じ2面が見える向き
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TEXTURE_SIZE = 512
WOOD_SEED = 25
K = np.array([
    [FOCAL_LENGTH_PX, 0.0, PRINCIPAL_POINT_X_PX],
    [0.0, FOCAL_LENGTH_PX, PRINCIPAL_POINT_Y_PX],
    [0.0, 0.0, 1.0],
])


def make_scene():
    """三角錐の頂点・面とカメラの周回中心を返す。"""
    vertices = np.array([
        [0.0, 0.0, 1.6],       # 0: 上の頂点
        [0.0, -0.65, -0.6],    # 1: 手前の底面頂点
        [1.2, 0.45, -0.6],     # 2: 右奥
        [-1.2, 0.45, -0.6],    # 3: 左奥
    ])
    # 形状を保ったまま、底面の重心を世界座標の原点に移す。
    base_center = vertices[1:].mean(axis=0)
    vertices -= base_center
    # 底面の中心を軸として、三角錐をZ軸周りに回転する。
    angle = np.deg2rad(OBJECT_ROTATION_DEG)
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    vertices = (rotation_z @ vertices.T).T
    # xyの中心は原点にそろえ、高さは前回の配置と同じにする。
    orbit_center = np.array([0.0, 0.0, -base_center[2]])
    faces = [(0, 1, 3), (0, 2, 1), (0, 3, 2), (1, 2, 3)]
    return vertices, faces, orbit_center


def normalize(vector):
    return vector / np.linalg.norm(vector)


def make_camera(azimuth_deg, orbit_center):
    """指定の方位角にカメラを置き、注視点LOOK_ATへ向ける。"""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(ELEVATION_DEG)
    center = orbit_center + CAMERA_DISTANCE * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    forward = normalize(LOOK_AT - center)
    right = normalize(np.cross(forward, [0.0, 0.0, 1.0]))
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return center, R, -R @ center


def project(points_world, R, t):
    """世界座標(Zが上)を、画像座標(右がu、下がv)へ透視投影する。"""
    # 列ベクトルでは X_camera = R @ X_world + t。
    # カメラ座標は右がx、下がy、前がz。
    points_camera = (R @ np.asarray(points_world).T).T + t
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("カメラの前方にない点があります。")
    homogeneous = (K @ points_camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3], points_camera[:, 2]


def visible_faces(vertices, faces, camera_center):
    """凸な三角錐の、カメラに面している面を判定する。"""
    object_center = vertices.mean(axis=0)
    visible = []
    for face in faces:
        triangle = vertices[list(face)]
        face_center = triangle.mean(axis=0)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.dot(normal, face_center - object_center) < 0:
            normal = -normal
        visible.append(bool(np.dot(normal, camera_center - face_center) > 0))
    return visible


def make_wood_texture(seed):
    """縞、節、粗さの異なる濃淡を重ね、面に貼る木目画像を作る。"""
    rng = np.random.default_rng(seed)
    size = TEXTURE_SIZE
    y, x = np.mgrid[:size, :size].astype(np.float32)
    field = np.zeros_like(x)
    for rows, cols, weight in [(5, 13, 1), (11, 30, 0.48), (25, 62, 0.20)]:
        noise = rng.normal(size=(rows, cols)).astype(np.float32)
        field += weight * cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)

    # 縦に流れる縞を、濃淡のむらと節の周囲で曲げる。
    phase = x / 18 + 0.34 * field + 0.20 * np.sin(y / 68)
    knots = np.zeros_like(x)
    for _ in range(11):
        knot_x, knot_y = rng.uniform(15, 485, 2)
        dx = (x - knot_x) / rng.uniform(9, 23)
        dy = (y - knot_y) / rng.uniform(18, 52)
        radius = np.sqrt(dx**2 + dy**2)
        phase += 0.85 * np.exp(-0.34 * radius**2) * np.tanh(dx)
        knots += np.exp(-0.75 * radius**2) * (0.55 + 0.45 * np.cos(radius * 7))

    grain = (
        0.34 * np.cos(phase * 6.283)
        + 0.12 * np.cos(phase * 12.566)
        + 0.65 * field
        - 1.75 * knots
    )
    base = np.zeros_like(x)
    base_rng = np.random.default_rng(seed)
    for grid_size, weight in [(8, 1), (17, 0.6), (37, 0.25)]:
        noise = base_rng.normal(size=(grid_size, grid_size)).astype(np.float32)
        base += weight * cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)

    grain = 0.58 * grain + 1.1 * base
    base_color = np.array([133, 181, 218])  # OpenCVのBGR順
    color_variation = np.array([37, 45, 48])
    return np.uint8(np.clip(base_color + grain[:, :, None] * color_variation, 12, 250))


def draw_view(vertices, faces, textures, camera):
    center, rotation, translation = camera
    image = np.full((HEIGHT, WIDTH, 3), 244, dtype=np.uint8)
    last = TEXTURE_SIZE - 1
    texture_corners = np.float32([[0, 0], [last, 0], [0, last], [last, last]])

    for face, texture, visible in zip(faces, textures, visible_faces(vertices, faces, center)):
        if not visible:
            continue

        # 三角形の3頂点に、同じ平面上の補助点を足して射影変換を求める。
        a, b, c = vertices[list(face)]
        corners = np.array([a, b, c, b + c - a])
        pixels, _ = project(corners, rotation, translation)
        # project()の画素中心(0.5, 0.5)をOpenCVの(0, 0)へ変換。
        pixels -= 0.5
        homography = cv2.getPerspectiveTransform(texture_corners, pixels.astype(np.float32))
        warped = cv2.warpPerspective(
            texture, homography, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE
        )

        # 四角形全体を投影し、実際の三角形の内側だけを画像に合成する。
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(pixels[:3]).astype(np.int32), 255)
        image[mask != 0] = warped[mask != 0]

    return image


def main():
    vertices, faces, orbit_center = make_scene()
    textures = [make_wood_texture(WOOD_SEED + 1000 * i) for i in range(len(faces))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, angle in enumerate(AZIMUTHS_DEG, start=1):
        camera = make_camera(angle, orbit_center)
        image = draw_view(vertices, faces, textures, camera)
        path = OUTPUT_DIR / f"view_{i:02d}_wood.png"
        # imwriteのパス処理に依存せず、Windowsの日本語パスでも保存する。
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise OSError(f"PNGへの変換に失敗しました: {path}")
        path.write_bytes(encoded.tobytes())
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
