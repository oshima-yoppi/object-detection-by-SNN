import numpy as np
import cv2



def draw_circle(center, radius,time=10, pixel=128):
    img = np.zeros((pixel, pixel, 2), dtype=np.uint8)
    cv2.circle(img, center, radius, color=255, thickness=1, lineType=cv2.LINE_8, shift=0)
    return img
def youtube(imgs, pixel=128):
    CLIP_FPS = 20.0
    filepath = 'youtube/test.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filepath, codec, CLIP_FPS, (pixel, pixel))

    for img in imgs:
        video.write(img)
    video.release()
    return
    
if __name__ == "__main__":
    img = draw_circle((0, 50), 20)
    imgs = []
    for i in range(10):
        img = draw_circle((0, 50), 20 + i)
        imgs.append(img)
    youtube(imgs)
    # 画像の表示
    # cv2.imshow("Image", img)
    # cv2.waitKey()