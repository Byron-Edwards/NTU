import json
import os
from pathlib import Path

import numpy as np

from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html

floor_data_dir = './data/site1/F1'
path_data_dir = floor_data_dir + '/path_data_files'
floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'

save_dir = './output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir


# 根据真实值校准磁力和wifi数据


def calibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
    mwi_datas = {}
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')

        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint

        # 使用加速度数据和角度数据，计算每一步的位置信息
        step_positions = compute_step_positions(
            acce_datas, ahrs_datas, posi_datas)
        # visualize_trajectory(posi_datas[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Position', show=True)

        if wifi_datas.size != 0:
            # 对wifi数据时间戳进行去重和排序
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            # 得到按时间戳分组后的所有wifi数据
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                # 找出所有位置时间戳与当前wifi数据时间戳的差值
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                # 找出差值最小的下标
                index = np.argmin(diff)
                # 得到最小值的xy坐标计做（x,y）
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    # 如果当前位置已有wifi信息，则增加一条记录
                    mwi_datas[target_xy_key]['wifi'] = np.append(
                        mwi_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                else:
                    # 记录当前位置的wifi信息
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': wifi_ds,
                        'ibeacon': np.zeros((0, 3))
                    }

        # 以同样的逻辑得到不同位置的磁力信息
        sep_tss = np.unique(magn_datas[:, 0].astype(float))
        magn_datas_list = split_ts_seq(magn_datas, sep_tss)
        for magn_ds in magn_datas_list:
            diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]['magnetic'] = np.append(
                    mwi_datas[target_xy_key]['magnetic'], magn_ds, axis=0)
            else:
                mwi_datas[target_xy_key] = {
                    'magnetic': magn_ds,
                    'wifi': np.zeros((0, 5)),
                    'ibeacon': np.zeros((0, 3))
                }

    return mwi_datas


# 计算磁力值用于绘制热力图


def extract_magnetic_strength(mwi_datas):
    magnetic_strength = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        magnetic_data = mwi_datas[position_key]['magnetic']
        # 每个点的磁力数据的值，用L2距离来表示，后续用于绘制热力图
        magnetic_s = np.mean(
            np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_strength[position_key] = magnetic_s

    return magnetic_strength


# 计算wifi的rssi的值用于绘制热力图


def extract_wifi_rssi(mwi_datas):
    wifi_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            # wifi的物理id
            bssid = wifi_d[2]
            # rssi的值
            rssi = int(wifi_d[3])

            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (
                                                             old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            # 整理获得不同id在不同位置的信号值
            wifi_rssi[bssid] = position_rssi

    return wifi_rssi


if __name__ == "__main__":
    Path(path_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(magn_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(wifi_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(ibeacon_image_save_dir).mkdir(parents=True, exist_ok=True)

    # 读取楼层地图信息（宽高）
    with open(floor_info_filename) as f:
        floor_info = json.load(f)
    width_meter = floor_info["map_info"]["width"]
    height_meter = floor_info["map_info"]["height"]

    # 读取所有文件名
    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))

    # Task1. Visualize way points (ground-truth locations)
    # 这一问直接读数据绘制即可，不需要再处理数据
    print('Visualizing ground truth positions...')
    for path_filename in path_filenames:
        print(f'Processing file: {path_filename}...')

        path_data = read_data_file(path_filename)
        path_id = path_filename.name.split(".")[0]
        fig = visualize_trajectory(
            path_data.waypoint[:, 1:3], floor_plan_filename, width_meter, height_meter, title=path_id, show=False)
        html_filename = f'{path_image_save_dir}/{path_id}.html'
        html_filename = str(Path(html_filename).resolve())
        save_figure_to_html(fig, html_filename)

    # Task2: Visualize geomagnetic heat map
    print('Visualizing more information...')
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)

    magnetic_strength = extract_magnetic_strength(mwi_datas)
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter,
                            height_meter, colorbar_title='mu tesla', title='Magnetic Strength', show=True)
    html_filename = f'{magn_image_save_dir}/magnetic_strength.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)

    # Task3:Visualize RSS heat maps of 3 Wi-Fi APs
    wifi_rssi = extract_wifi_rssi(mwi_datas)
    print(f'This floor has {len(wifi_rssi.keys())} wifi aps')
    ten_wifi_bssids = list(wifi_rssi.keys())[0:10]
    print('Example 10 wifi ap bssids:\n')
    for bssid in ten_wifi_bssids:
        print(bssid)
    target_wifi = input(f"Please input target wifi ap bssid:\n")
    # target_wifi = '1e:74:9c:a7:b2:e4'
    heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
    heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter,
                            height_meter, colorbar_title='dBm', title=f'Wifi: {target_wifi} RSSI', show=True)
    html_filename = f'{wifi_image_save_dir}/{target_wifi.replace(":", "-")}.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)
