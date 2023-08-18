import argparse
import datetime
from pathlib import Path
import json
import numpy as np

DATASETS = [
    "Vitens-Coliform-coco",
    "Vitens-Coliform-coco-24",
    "Vitens-Aeromonas-coco",
    "Vitens-Kiemgetal-coco-full",
    "dota_v1_coco",
    "wgisd-coco",
    "alumunium-coco-roboflow",
    "Chicken-Real-Time-coco-roboflow",
    "fashion-categories-coco-roboflow",
    "skindetect-roboflow",
    "pool-danger-coco-roboflow",
    "blueberries-roboflow"
]


def read_e2e_time(log):
    time_info = []
    with open(log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "mmengine - INFO" in line:
                time_info.append(line)
    line_breakpoint = time_info[0].find(" - mmengine")
    begin_time = time_info[0][:line_breakpoint]
    end_time = time_info[-1][:line_breakpoint]

    begin_time = datetime.datetime.strptime(begin_time, "%Y/%m/%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")
    duration = end_time - begin_time
    return duration.total_seconds() / 60.0


def read_train_json(json_path):
    iter_lines = []
    with open(json_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            iter_lines.append(json.loads(line))

    total_memory = []
    iter_times = []
    data_times = []
    epochs = []
    for line in iter_lines:
        if 'time' in line:
            iter_times.append(line['time'])
        if 'data_time' in line:
            data_times.append(line['data_time'])
        if 'memory' in line:
            total_memory.append(line['memory'])
        if 'epoch' in line:
            epochs.append(line['epoch'])

    avg_iter_time = np.average(iter_times)
    avg_data_time = np.average(data_times)
    avg_gpu_meomry = np.average(total_memory)
    total_epoch = np.max(epochs)

    return avg_iter_time, avg_data_time, avg_gpu_meomry, total_epoch


def read_performance(perf_json):
    with open(perf_json, 'r') as f:
        perf_line = json.load(f)
    return perf_line['coco/segm_mAP_50']


def collect_results(folder):
    model_folder = Path(folder)
    print(model_folder.name)
    total_time = 0
    full_string = ""
    for i, dataset_name in enumerate(DATASETS):
        dataset_folder = model_folder / f"{dataset_name}"
        try:
            train_folder = list(dataset_folder.glob("20*"))[0]
            eval_folder = list(dataset_folder.glob("test/20*"))[0]

            eval_json = list(eval_folder.glob("20*.json"))[0]
            train_log = list(train_folder.glob("20*.log"))[0]
            train_json = list(train_folder.glob("vis_data/20*.json"))[0]

            test_mAP = read_performance(eval_json)
            avg_iter_time, avg_data_time, avg_gpu_meomry, total_epoch = read_train_json(train_json)
            elapsed_time = read_e2e_time(train_log)
            total_time += elapsed_time
            full_string += f"{test_mAP},{elapsed_time},{avg_iter_time},{avg_gpu_meomry},{total_epoch},"
        except:
            print(f"{dataset_name} error")
            full_string += f"{0.0},{0.0},{0.0},{0.0},{0.0},"
    print(full_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    collect_results(args.folder)