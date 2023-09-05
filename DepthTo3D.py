import sys
import cv2
import json
import numpy as np
import pandas as pd
import open3d as o3d
import time
import os


### 内部パラメータ保持構造体
class intrinsic_strucst:
    def setParam(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

intstruct = intrinsic_strucst()
framename = "frameDataSet.json"
depthname = "depthData.csv"
exportpcd = "depthPCD.pcd"
exportcolorpcd = "depthColorPCD.pcd"
exportcolorply = "depthColorPCD.ply"

exportply = "depthPCD.ply"
png_image = "rgbImgData.png"
#png_image = "rgbImgData_mark.png"

ownpath = ""
export_dir = ""

### jsonからRGBデータのwidthとheightを取得する ###
# jsonpath : 読込対象ファイルパス
def readJSON(jsonpath):
    search_file = jsonpath + "/" + framename
    if os.path.exists(search_file):
        with open(search_file, "r") as file:
            dict = json.load(file)
    
        rgb_width = dict["rgb"]["width"]
        rgb_height = dict["rgb"]["height"]

        intstruct.setParam(dict["intrinsic"]["fx"], 
                           dict["intrinsic"]["fy"],
                           dict["intrinsic"]["cx"],
                           dict["intrinsic"]["cy"])
        #print(intstruct.fx)
        return rgb_width, rgb_height
    else:
        return False

### Depthmapの読込 ###
# depthpath : 読込対象Depthmapファイルパス
# rgb_width : RGB画像データの横サイズ
# rgb_height : RGB画像データの縦サイズ
def readDepth(depthpath, rgb_width, rgb_height):
    depthmap = pd.read_csv(depthpath + "/" + depthname, header=None).values
    new_size = (rgb_width, rgb_height)
    return cv2.resize(depthmap, new_size, interpolation = cv2.INTER_LINEAR)


def depthToCP(depthmap):
    height, width = depthmap.shape
    totalSize = width * height * 3
    result = [0.0] * totalSize

    index = 0
    for y in range(height):
        for x in range(width):
            ZZ = depthmap[y][x]
            XX = ZZ * (float(width) - intstruct.cx) / intstruct.fx
            YY = ZZ * (float(height) - intstruct.cy) / intstruct.fy
            result[index] = float(XX)
            result[index + 1] = float(YY)
            result[index + 2] = float(ZZ)
            index += 3
    
    X = np.reshape(XX, -1)
    Y = np.reshape(YY, -1)
    Z = np.reshape(ZZ, -1)

    # 座標を結合
    # pcdの形式にするため、[X, Y, Z]に変換する
    points = np.vstack((X, Y, Z)).transpose()
    # 軸は結果が変わらないのでどちらでも良いが横方向
    points = points[np.isfinite(points).all(axis=1)]

    # open3dのPointCloudオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    o3d.io.write_point_cloud(export_dir + "/" + exportply, pcd)
    o3d.io.write_point_cloud(export_dir + "/" + exportpcd, pcd)

    AddColorPCD(pcd, export_dir)

    return result

### Depthmapと内部パラメータによる3次元再構成
# depthmap : Depthmap
def depthTo3D(depthmap):

    # 深度画像のサイズ
    height, width = depthmap.shape
    print(f"height : {height}, width : {width}")
    # 画像の各ピクセルの座標を生成
    # linspace : {0}から{width - 1}まで、{width}個の要素の1次元配列を作る
    x = np.linspace(0, width - 1, width)
    # linspace : {0}から{height - 1}まで、{height}個の要素の1次元配列を作る
    y = np.linspace(0, height - 1, height)
    # 2次元配列を作る
    u, v = np.meshgrid(x, y)


    print(f"depthmap.shape : {depthmap.shape}")
    print(f"x : {x.shape}")
    print(f"y : {y.shape}")
    # 画像座標系からカメラ座標系に変換
    #X = (intstruct.cx - u) / intstruct.fx * depthmap
    X = depthmap * (x - intstruct.cx)/intstruct.fx
    #Y = (intstruct.cy - v) / intstruct.fy * depthmap
    Y = depthmap * (y[:, np.newaxis] - intstruct.cy) / intstruct.fy
    
    
    #X = depthmap * (u - intstruct.fx) / intstruct.cx
    #Y = depthmap * (v - intstruct.fy) / intstruct.cy
    #Y = (intstruct.cy - v) / intstruct.fy * depthmap
    Z = depthmap

    # XYZ座標を1次元に変換
    X = np.reshape(X, -1)
    Y = np.reshape(Y, -1)
    Z = np.reshape(Z, -1)

    # 座標を結合
    # pcdの形式にするため、[X, Y, Z]に変換する
    points = np.vstack((X, Y, Z)).transpose()
    # 軸は結果が変わらないのでどちらでも良いが横方向
    points = points[np.isfinite(points).all(axis=1)]

    # open3dのPointCloudオブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    o3d.io.write_point_cloud(export_dir + "/" + exportply, pcd)
    o3d.io.write_point_cloud(export_dir + "/" + exportpcd, pcd)

    AddColorPCD(pcd, export_dir)

    """
    # 3D可視化
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # 回転角度（ラジアン）
    angle = np.pi / 180
    # Z軸に対する回転行列
    print(f"angle : {angle}")
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0.174533, 0])
    for _ in range(360):  # 360度回転
        # 回転
        pcd.rotate(R, center=pcd.get_center())

        # 更新
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 一時停止（回転速度の制御）
        #time.sleep(0.001)

    # 終了
    vis.destroy_window()
    """

def AddColorPCD(pcd, target_dir):
    from PIL import Image
    image_path = AppendPath(target_dir, png_image)
    image = Image.open(image_path)
    color_array = np.array(image)
    color_array = color_array / 255.0
    colors = color_array.reshape(-1, 3)
    #print(f"color_array : {color_array}")
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(export_dir + "/" + exportcolorply, pcd)
    o3d.io.write_point_cloud(export_dir + "/" + exportcolorpcd, pcd)

def AppendPath(own, child):
    return own + "/" + child

def GetDirList():
    dir_array = []
    for root, dirs, files in os.walk(ownpath):
        for dir in dirs:
            dir_array.append(os.path.join(root, dir))
    return dir_array

    #dirs = os.listdir(ownpath)
    #dir_array = [AppendPath(ownpath, dir) for dir in dirs if not dir.startswith('.')]
    #return dir_array

if __name__ == "__main__":
    args = sys.argv
    ownpath = args[1]

    for target_dir in GetDirList():
        export_dir = target_dir
        print(f"export_dir : {export_dir}")
        # 内部パラメータを読み込む
        if readJSON(target_dir) != False:
            rgb_width, rgb_height = readJSON(target_dir)
            depthTo3D(readDepth(target_dir, rgb_width, rgb_height))
            #depthToCP(readDepth(target_dir, rgb_width, rgb_height))
