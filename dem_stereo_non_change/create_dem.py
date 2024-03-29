import math
import numpy as np
from module.const import *
from module import convert_label, hazard
import random
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm import tqdm

# random seed
random.seed(123)
np.random.seed(123)


class LunarDEMGenerator(hazard.LunarHazardMapper):
    def __init__(self, shape, max_crater, max_boulder, sigma, harst, rough, theta):
        self.shape = shape
        self.dem = np.zeros((self.shape, self.shape), dtype="float32")
        self.side = self.shape - 1
        self.max_crater = max_crater
        self.max_boulder = max_boulder

        super().__init__(shape=shape, rough=rough, theta=theta)
        self.sigma0 = sigma
        self.harst = harst

    def calculate_sigma(self, n):
        sigma_n = (
            self.sigma0 * (1 - 2 ** (2 * self.harst - 2)) / (2**n) ** (2 * self.harst)
        )
        return sigma_n

    def put_fractal(self):
        self.dem[0 :: self.shape - 1, 0 :: self.shape - 1] = np.random.uniform(
            -1, 1, (2, 2)
        )
        nsquares = 1
        step = 1
        while self.side > 1:
            sideo2 = self.side // 2

            # Diamond step
            for ix in range(nsquares):
                for iy in range(nsquares):
                    x0, x1, y0, y1 = (
                        ix * self.side,
                        (ix + 1) * self.side,
                        iy * self.side,
                        (iy + 1) * self.side,
                    )
                    xc, yc = x0 + sideo2, y0 + sideo2
                    # Set this pixel to the mean of its "diamond" neighbours plus
                    # a random offset.
                    self.dem[yc, xc] = (
                        self.dem[y0, x0]
                        + self.dem[y0, x1]
                        + self.dem[y1, x0]
                        + self.dem[y1, x1]
                    ) / 4
                    # self.dem[yc,xc] += f * np.random.uniform(-1,1)
                    self.dem[yc, xc] += np.random.normal(0, self.calculate_sigma(step))
            step += 1
            # Square step: NB don't do this step until the pixels from the preceding
            # diamond step have been set.
            for iy in range(2 * nsquares + 1):
                yc = sideo2 * iy
                for ix in range(nsquares + 1):
                    xc = self.side * ix + sideo2 * (1 - iy % 2)
                    if not (0 <= xc < self.shape and 0 <= yc < self.shape):
                        continue
                    tot, ntot = 0.0, 0
                    # Set this pixel to the mean of its "square" neighbours plus
                    # a random offset. At the edges, it has only three neighbours
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        xs, ys = xc + dx * sideo2, yc + dy * sideo2
                        if not (0 <= xs < self.shape and 0 <= ys < self.shape):
                            continue
                        else:
                            tot += self.dem[ys, xs]
                            ntot += 1
                    self.dem[yc, xc] += tot / ntot
                    self.dem[yc, xc] += np.random.normal(0, self.calculate_sigma(step))
            self.side = sideo2
            nsquares *= 2
            # f /= 2
            step += 1
        return

    def put_crater(self):
        eps = 1e-6
        n_crater = random.randint(0, self.max_crater)
        for i in range(n_crater):
            center_x, center_y = (
                random.uniform(-5, self.shape + 5),
                random.uniform(-5, self.shape + 5),
            )
            minimum_detection_crater_size = 0.3  # 見つけたい障害物の最小の大きさ[m]
            min_pix_of_crater = minimum_detection_crater_size / METER_PER_GRID  # [pix]

            radius = random.uniform(min_pix_of_crater, self.shape // 8)

            # print(center_x, center_y, radius)

            #    % H_r = 150 + abs(5*randn())
            H_ro = 0.036 * (2 * radius) ** 1.014
            H_r = H_ro
            H_c = 0.196 * (2 * radius) ** 1.010 - H_ro
            W_r = 0.257 * (2 * radius) ** 1.011
            alpha = (H_c + H_r) * radius / (H_c + H_ro + eps)
            beta = radius + (1 - (H_c + H_r) / (H_c + H_ro + eps)) * W_r
            A = (
                -3 * radius**3
                + 2 * radius**2 * beta
                + 2 * radius * beta**2
                + 2 * beta**3
            )

            for i in range(self.shape):
                for j in range(self.shape):
                    h = 0
                    r = math.sqrt(abs(i - center_x) ** 2 + abs(j - center_y) ** 2)

                    if r <= alpha:
                        h = (H_c + H_ro) * (r**2 / radius**2) - H_c
                    elif r <= radius:
                        h = ((H_c + H_ro) ** 2 / (H_r - H_ro + eps) - H_c + eps) * (
                            (r / radius) - 1 + eps
                        ) ** 2 + H_r
                    elif r <= beta:
                        h = (H_r * (radius + W_r) ** 3 * A) / (
                            W_r
                            * beta**4
                            * (radius - beta) ** 2
                            * (3 * radius**2 + 3 * radius * W_r + W_r**2)
                            + eps
                        ) * (r - radius) ** 2 * (
                            r - beta * (1 + (beta**3 - radius**3) / A)
                        ) + H_r
                    elif r <= radius + W_r:
                        h = H_r * (radius + W_r) ** 3 / (
                            (radius + W_r) ** 3 - radius**3 + eps
                        ) * (r / radius) ** (-3) - (H_r * radius**3) / (
                            (radius + W_r) ** 3 - radius**3
                        )

                    self.dem[i, j] += h
        return self.dem

    def put_boulder(self):
        for i in range(self.max_boulder):
            center_x, center_y = (
                random.uniform(0, self.shape),
                random.uniform(0, self.shape),
            )

            min_detection_boulder_size = 0.15  # 見つけたい障害物の最小の大きさ[m]
            min_pix_of_boulder = min_detection_boulder_size / METER_PER_GRID  # [pix]
            max_detection_boulder_size = 0.5  # 見つけたい障害物の最大の大きさ[m]
            max_pix_of_boulder = max_detection_boulder_size / METER_PER_GRID  # [pix]

            x_axis = random.uniform(min_pix_of_boulder, max_pix_of_boulder)
            # x_axis = random.uniform(min_pix_of_boulder, min_pix_of_boulder)
            y_axis = random.uniform(min_pix_of_boulder, max_pix_of_boulder)
            # print(x_axis, y_axis)
            long_bool = random.uniform(0, 1)
            if long_bool >= 0.5:
                y_axis = x_axis * 0.75  # by kariya
            else:
                x_axis = y_axis * 0.75
            # x_axis = 2
            # y_axis = 2
            z_axis = max(x_axis, y_axis) * 0.5  # by kariya
            for i in range(self.shape):
                for j in range(self.shape):
                    # h = 0
                    x_ = i - center_x
                    y_ = j - center_y

                    # 楕円方程式無いに存在するか
                    # print(x_axis)
                    # print(x_/x_axis)
                    if y_**2 <= y_axis**2 * (1 - (x_ / x_axis) ** 2):
                        self.dem[i, j] += self.generate_elllipsoid(
                            i, j, center_x, center_y, x_axis, y_axis, z_axis
                        )
        return

    def generate_elllipsoid(self, x, y, xc, yc, xr, yr, zr):
        return zr * math.sqrt(1 - ((x - xc) ** 2 / xr**2) - ((y - yc) ** 2 / yr**2))

    def generate_dem(self):
        self.put_fractal()
        self.put_crater()
        self.put_boulder()
        return self.dem

    def generate_hazard(self):
        self.label = super().map_hazard()
        # self.converted_label = self.label_converter(self.label)
        return self.label

    def save_dem_and_label(self, path):
        # np.save(path, self.dem)
        np.savez(path, dem=self.dem, raw_label=self.label)  # demという名前で保存される


n = 8
shape = 2**n + 1  # The array must be square with edge length 2**n + 1
max_crater = 0
max_boulder = 3

# harst = 0.18
# sigma0 =3  # 3 now
# rough = 0.1
# theta = 20
harst = 0.18
sigma0 = 5  # 3 now
rough = 0.2
theta = 20

save_label_dir = LABEL_PATH
save_dem_dir = DEM_NP_PATH
if os.path.exists(save_label_dir):
    shutil.rmtree(save_label_dir)
os.mkdir(save_label_dir)
if os.path.exists(save_dem_dir):
    shutil.rmtree(save_dem_dir)
os.mkdir(save_dem_dir)
num_data = 3000
for i in tqdm(range(num_data)):
    dem_generator = LunarDEMGenerator(
        shape=shape,
        max_crater=max_crater,
        max_boulder=max_boulder,
        sigma=sigma0,
        harst=harst,
        rough=rough,
        theta=theta,
    )
    dem_generator.generate_dem()
    dem_generator.generate_hazard()

    number = str(i).zfill(5)
    dem_filename = f"{number}"
    save_dem_path = os.path.join(
        save_dem_dir, dem_filename
    )  # npzで保存する際のパスは拡張子をつけなくていいらしい,,,
    dem_generator.save_dem_and_label(save_dem_path)

    # plt.subplot(1, 2, 1)
    # plt.imshow(dem_generator.dem)
    # plt.subplot(1, 2, 2)
    # plt.imshow(dem_generator.label)
    # plt.show()
