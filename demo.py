# coding=utf-8
# ******************************************************************************
# Copyright (C), 2024-2031, Huawei Tech. Co., Ltd.
# ******************************************************************************
# File Name     : cluster_img
# Description   : 
# History       :
# Date          : 2024/10/31
# Author        : x00450452
# Modification  : Created file
# Version       : 1.0
# ******************************************************************************/
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


def get_color_name(lab):
    colors = {
        "red": [53.23, 80.11, 67.22],
        "green": [87.74, -86.18, 83.18],
        "blue": [32.30, 79.19, -107.86],
        "yellow": [97.14, -21.55, 94.48],
        "cyan": [91.11, -48.09, -14.13],
        "magenta": [60.32, 98.24, -60.83],
        "black": [0, 0, 0],
        "white": [100, 0.00, -0.01],
        "gray": [53.59, 0, 0],
        "orange": [67.79, 43.30, 74.93],
        "purple": [29.78, 58.94, -36.50],
        "brown": [38.91, 19.36, 22.29],
        "pink": [88.22, 17.75, 3.18],
        "dark red": [39.35, 62.75, 49.91],
        "light blue": [79.19, -11.03, -26.23],
        "dark green": [35.49, -46.47, 35.45],
        "light green": [88.72, -42.89, 57.40],
        "navy": [16.73, 37.09, -65.49],
        "burgundy": [28.71, 49.35, 26.27],
        "beige": [89.02, -1.39, 11.09],
        "olive": [51.87, -12.93, 56.67],
        "teal": [49.31, -28.83, -8.48],
        "maroon": [25.64, 45.52, 20.77],
        "forest green": [36.23, -37.96, 30.20],
    }

    min_dist = float('inf')
    closest_color = None
    for color_name, color_lab in colors.items():
        dist = np.sqrt(sum((lab - color_lab) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    return closest_color


def extract_colors(image_path, num_colors, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im_w = image.shape[1]
    im_h = image.shape[0]
    start_x = (im_w - dst_w) // 2
    end_x = start_x + dst_w
    start_y = (im_h - dst_h) // 2
    end_y = start_y + dst_h
    img_crop = image[start_x:end_x, start_y:end_y]

    cv2.imwrite("remove.jpg", cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))

    pixels = img_crop.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get the colors
    colors = kmeans.cluster_centers_
    rgblist = np.array(colors).tolist()
    hexlist = ['#%02x%02x%02x' % tuple(int(val) for val in rgblist[i]) for i in range(len(rgblist))]
    print(hexlist)

    numLabels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
    (hist, _) = np.histogram(kmeans.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    print(hist)

    ClusterAnalysis(hexlist, hist, img_crop, os.path.join(save_dir, os.path.basename(image_path)))


def ClusterAnalysis(hexlist, histogram, img, save_path=None):
    if hexlist is None:
        raise TypeError("Hex Color List Not Found")
    else:
        filtered_data = [(color, percent) for color, percent in zip(hexlist, histogram) if percent >= 0.01]
        if not filtered_data:
            raise ValueError("No colors with more than 0.1% presence found.")

        filtered_hexlist, filtered_hist = zip(*filtered_data)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(aspect="equal"), facecolor='#FAFAFA')
        ax[0].imshow(img)
        data = filtered_hist
        color = filtered_hexlist

        wedges, texts, autopcts = ax[1].pie(data, colors=color, autopct=lambda pct: "{:.1f}%".format(pct) if pct > 5 else '',
                                         pctdistance=0.85, wedgeprops=dict(width=0.3, linewidth=3, edgecolor='white'), startangle=-40)

        plt.setp(autopcts, **{'color': 'black', 'weight': 'bold', 'fontsize': 6})
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            # Only annotate larger wedges with labels
            if data[i] > 5:
                ax[1].annotate(f"{filtered_hexlist[i]}: {data[i] * 100:.1f}%", xy=(x, y), xytext=(1.5*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)

        # Create a legend for smaller wedges
        ax[1].legend(wedges, [f"{h}: {d*100:.1f}%" for h, d in zip(filtered_hexlist, data)],
                  title="Color Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        extract_colors(image_path, extract_colors_num_colors, save_dir)


if __name__ == '__main__':
    # Example usage
    image_dir = r'/home/x00450452/xifenlei/training_data_20241027/cropped_objs/Delivery_rider'
    save_dir = r'/home/x00450452/mnt/delivery_rider_cluster'
    dst_w = 96
    dst_h = 96
    remove_background_threshold = 100
    extract_colors_num_colors = 8
    is_similar_color_threshold = 30
    main()
