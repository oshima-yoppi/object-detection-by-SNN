# import bpycv
import cv2
import bpy
import bpycv
import random
import numpy as np

# remove all MESH objects
[bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]

for obj in bpy.data.objects:
    if obj.type == "MESH":
        bpy.data.objects.remove(obj)

# for index in range(1, 5):
#     # create cube and sphere as instance at random location
#     location = [random.uniform(-2, 2) for _ in range(3)]
#     if index % 2:
#         bpy.ops.mesh.primitive_cube_add(size=0.5, location=location)
#         categories_id = 1
#     else:
#         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location)
#         categories_id = 2
#     obj = bpy.context.active_object
#     # set each instance a unique inst_id, which is used to generate instance annotation.
#     obj["inst_id"] = categories_id * 100
verts = []
faces = []
for i in range(2):
    a = 1 * i
    verts.append(((a, 0, 0), (a + 1, 0, 0), (a + 1, 1, 0), (a, 1, 0)))
    faces.append((4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
    # verts = [(0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0)]
    # faces = [(0, 1, 2, 3)]

    # メッシュを定義する
mesh = bpy.data.meshes.new("Plane_mesh")
# 頂点と面のデータからメッシュを生成する
mesh.from_pydata(verts, [], faces)
mesh.update(calc_edges=True)

# メッシュのデータからオブジェクトを定義する
obj = bpy.data.objects.new("Plane", mesh)  ##
# オブジェクトの生成場所をカーソルに指定する
obj.location = bpy.context.scene.cursor.location
obj["inst_id"] = 150
# オブジェクトをシーンにリンク(表示)させる
bpy.context.scene.collection.objects.link(obj)
# render image, instance annoatation and depth in one line code
result = bpycv.render_data()
print(bpy.context.scene)
# result["ycb_meta"] is 6d pose GT

# save result
# cv2.imshow("rgb", result["image"][..., ::-1])  # transfer RGB image to opencv's BGR
path = "D://research//object-detection-by-SNN//dem_stereo//"
cv2.imwrite(path + "demo-rgb.jpg", result["image"][..., ::-1])  # transfer RGB image to opencv's BGR

# save instance map as 16 bit png
instance_map = result["inst"]
print(instance_map.shape)
print(type(instance_map))
print(np.unique(instance_map))
print(np.max(instance_map), np.min(instance_map))
cv2.imwrite(path + "demo-inst.png", result["inst"])
