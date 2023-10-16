import bpy
import cv2
import bpycv
import numpy as np

# 立方体を形成する頂点と面を定義する
h = 3
z = 1
verts = [(0, 0, 0), (0, h, 0), (h, h, 0), (h, 0, 0), (0, 0, z), (0, h, z), (h, h, z), (h, 0, z)]
faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
]

# メッシュを定義する
mesh = bpy.data.meshes.new("Cube_mesh")
# 頂点と面のデータからメッシュを生成する
mesh.from_pydata(verts, [], faces)
mesh.update(calc_edges=True)

# メッシュのデータからオブジェクトを定義する
obj = bpy.data.objects.new("Cube", mesh)
# オブジェクトの生成場所をカーソルに指定する
obj.location = bpy.context.scene.cursor.location
# オブジェクトをシーンにリンク(表示)させる
bpy.context.scene.collection.objects.link(obj)

obj["inst_id"] = 150

result = bpycv.render_data()
instance_map = result["inst"]
instance_map = instance_map.astype(np.uint16)
print(instance_map.shape)
print(np.unique(instance_map))
cv2.imshow("a", instance_map[..., ::-1])
