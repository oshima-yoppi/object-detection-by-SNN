import numpy as np
import math
import cv2


class LunarHazardMapper:
    def __init__(self, shape, rough, theta, window=3):
        self.shape = shape
        self.rough = rough
        self.theta = theta
        self.window = window

    def Get_Slope(self, roi):
        middle = self.window // 2
        N = roi[middle - 1, middle]
        S = roi[middle + 1, middle]
        E = roi[middle, middle + 1]
        W = roi[middle, middle - 1]
        SE = roi[middle + 1, middle + 1]
        NE = roi[middle - 1, middle + 1]
        SW = roi[middle + 1, middle - 1]
        NW = roi[middle - 1, middle - 1]
        fx = (SE - SW + np.sqrt(2) * (E - W) + NE - NW) / (4 + 2 * np.sqrt(2))
        fy = (NW - SW + np.sqrt(2) * (N - S) + NE - SE) / (4 + 2 * np.sqrt(2))
        theta = np.arctan(math.sqrt((fx ** 2 + fy ** 2)))
        return theta

    def Get_Roughness(self, cropped):
        roughness = np.var(cropped)
        return roughness

    def map_hazard(self):
        # ウィンドウ大きさ
        half_window = self.window // 2
        self.dem_padding = np.pad(self.dem, half_window)

        S = np.zeros((self.shape, self.shape))  # slope for each pixel
        R = np.zeros((self.shape, self.shape))  # roughness for each pixel

        for i in range(half_window, half_window + self.shape):
            for j in range(half_window, half_window + self.shape):
                cropped_window = self.dem_padding[
                    i - half_window : i + half_window + 1,
                    j - half_window : j + half_window + 1,
                ]
                suiheido = self.Get_Slope(cropped_window)
                S[i - half_window, j - half_window] = max(
                    S[i - half_window, j - half_window], suiheido
                )

                heitando = self.Get_Roughness(cropped_window)
                R[i - half_window, j - half_window] = max(
                    R[i - half_window, j - half_window], heitando
                )

        S = S > math.radians(self.theta)
        R = R > self.rough

        hazard = S | R
        return hazard
