"""三角錐を2方向から描画し、画像4枚と特徴点座標のCSV4個を保存する。

Python 3.13 / Windows 11・Ubuntu。依存: NumPy, Pillow
実行: python make_two_views.py

view_XX_points.csv: 従来どおり画素中心へ丸めた座標。
view_XX_points_ideal.csv: project()の出力を丸めずに保存した理想座標。
どちらもCOLMAPと同じ座標規約で、列は id,x,y。
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
AZIMUTHS_DEG = (0.0, 30.0)  # +x方向から始め、+y方向へ30度移動
OBJECT_ROTATION_DEG = 90.0  # どちらのカメラからも同じ2面が見える向き
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SCALE = 3  # 大きく描いて縮小し、輪郭を滑らかにする
MARKER_RADIUS_PX = 4  # 奥行きによらず、画像上の半径をそろえる
MARKER_COLOR = (220, 35, 35)
K = np.array([
    [FOCAL_LENGTH_PX, 0.0, PRINCIPAL_POINT_X_PX],
    [0.0, FOCAL_LENGTH_PX, PRINCIPAL_POINT_Y_PX],
    [0.0, 0.0, 1.0],
])


def make_scene():
    """頂点4点、辺の中点6点、面の重心4点を作る。"""
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
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    colors = [(185, 215, 244), (248, 216, 165),
              (207, 224, 204), (216, 216, 224)]

    markers = []
    for vertex_index, point in enumerate(vertices):
        markers.append({
            "id": len(markers) + 1, "xyz": point,
            "faces": [i for i, face in enumerate(faces) if vertex_index in face],
        })
    for a, b in edges:
        markers.append({
            "id": len(markers) + 1, "xyz": (vertices[a] + vertices[b]) / 2,
            "faces": [i for i, face in enumerate(faces) if a in face and b in face],
        })
    for face_index, face in enumerate(faces):
        markers.append({
            "id": len(markers) + 1, "xyz": vertices[list(face)].mean(axis=0),
            "faces": [face_index],
        })
    return vertices, faces, colors, markers, orbit_center


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


def scaled_uv(uv):
    """COLMAPの座標を、拡大描画するPillowのピクセル中心座標に変換する。"""
    return tuple(np.asarray(uv) * SCALE - 0.5)


def select_visible_markers(markers, visible):
    """見える面に属する特徴点を、番号の昇順で返す。"""
    return sorted((m for m in markers if any(visible[i] for i in m["faces"])),
                  key=lambda m: m["id"])


def save_points_csv(path, vertices, faces, markers, camera, *, ideal=False):
    """画素中心へ丸めた座標、または丸めない理想座標を保存する。

    ideal=False: 従来どおりfloorで画素番号を求め、0.5を足す。
    ideal=True: project()の出力をそのまま保存する。
    """
    center, R, t = camera
    visible = visible_faces(vertices, faces, center)
    shown = select_visible_markers(markers, visible)
    uv, _ = project([m["xyz"] for m in shown], R, t)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "x", "y"])
        for marker, (x, y) in zip(shown, uv):
            if ideal:
                # Pythonのfloat表記で保存し、読み戻すと同じfloat64値になる。
                # 桁数指定による丸めや、画素中心へ移すための+0.5は行わない。
                writer.writerow([marker["id"], repr(float(x)), repr(float(y))])
            else:
                # project()は左上の角が原点なので、floorで画素の整数番号を得る。
                # その番号に0.5を足すと、COLMAPでの画素中心の座標になる。
                pixel_x = int(np.floor(x))
                pixel_y = int(np.floor(y))
                writer.writerow([marker["id"], f"{pixel_x + 0.5:.1f}",
                                 f"{pixel_y + 0.5:.1f}"])


def add_numbers(draw, markers, marker_uv, font):
    """目印や別の番号に重ならない位置を選んで番号を付ける。"""
    occupied = []
    for uv in marker_uv:
        x, y = scaled_uv(uv)
        occupied.append((x-6*SCALE, y-6*SCALE, x+6*SCALE, y+6*SCALE))
    for marker, uv in zip(markers, marker_uv):
        x, y = scaled_uv(uv)
        label = str(marker["id"])
        bbox = draw.textbbox((0, 0), label, font=font, anchor="lt")
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        offsets = [(7*SCALE, -height-4*SCALE), (7*SCALE, 5*SCALE),
                   (-width-7*SCALE, -height-4*SCALE), (-width-7*SCALE, 5*SCALE),
                   (-width/2, -height-10*SCALE), (-width/2, 10*SCALE)]
        for dx, dy in offsets:
            box = (x+dx-SCALE, y+dy-SCALE,
                   x+dx+width+SCALE, y+dy+height+SCALE)
            if all(box[2] < other[0] or box[0] > other[2] or
                   box[3] < other[1] or box[1] > other[3] for other in occupied):
                occupied.append(box)
                draw.text((x+dx, y+dy), label, font=font, anchor="lt",
                          fill=(20, 20, 20), stroke_width=SCALE,
                          stroke_fill=(250, 250, 250))
                break
        else:
            raise ValueError(f"特徴点 {marker['id']} の番号を置く空間がありません。")


def draw_view(vertices, faces, colors, markers, camera, with_numbers):
    center, R, t = camera
    vertices_uv, depths = project(vertices, R, t)
    visible = visible_faces(vertices, faces, center)
    # 頂点や辺上の点は、どれか一つでも見える面に属せば描画する。
    shown = select_visible_markers(markers, visible)
    marker_uv, _ = project([m["xyz"] for m in shown], R, t)
    image = Image.new("RGB", (WIDTH*SCALE, HEIGHT*SCALE), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13*SCALE)  # OS固有のフォントは使わない
    order = sorted(range(len(faces)),
                   key=lambda i: depths[list(faces[i])].mean(), reverse=True)
    for i in order:
        if visible[i]:
            polygon = [scaled_uv(uv) for uv in vertices_uv[list(faces[i])]]
            draw.polygon(polygon, fill=colors[i])
            draw.line(polygon + [polygon[0]], fill=(60, 65, 70), width=2*SCALE)
    for uv in marker_uv:
        x, y = scaled_uv(uv)
        radius = MARKER_RADIUS_PX*SCALE
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=MARKER_COLOR)
    if with_numbers:
        add_numbers(draw, shown, marker_uv, font)
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vertices, faces, colors, markers, orbit_center = make_scene()
    for with_numbers in (False, True):
        suffix = "numbered" if with_numbers else "plain"
        for index, angle in enumerate(AZIMUTHS_DEG, start=1):
            camera = make_camera(angle, orbit_center)
            image = draw_view(vertices, faces, colors, markers, camera, with_numbers)
            path = OUTPUT_DIR / f"view_{index:02d}_{suffix}.png"
            image.save(path)
            print(f"Saved: {path}")
    # 各視点について、従来の座標CSVに加えて理想座標CSVも保存する。
    for index, angle in enumerate(AZIMUTHS_DEG, start=1):
        camera = make_camera(angle, orbit_center)
        path = OUTPUT_DIR / f"view_{index:02d}_points.csv"
        save_points_csv(path, vertices, faces, markers, camera)
        print(f"Saved: {path}")
        ideal_path = OUTPUT_DIR / f"view_{index:02d}_points_ideal.csv"
        save_points_csv(ideal_path, vertices, faces, markers, camera, ideal=True)
        print(f"Saved: {ideal_path}")


if __name__ == "__main__":
    main()
