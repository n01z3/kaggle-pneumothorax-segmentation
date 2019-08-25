__author__ = 'n01z3'

import os.path as osp
import socket

import yaml

PATHS = {
    "n01z3-dl0": "dl0",
    "n01z3-dl1": "dl1",
    "n01z3-dl2": "dl2",
    "n01z3-dl4": "dl4",
    "n01z3-dl6": "dl6",
    "n01z3-work": "work",
    "mn-dgx01.x5.ru": "dgx",
    "nizhib-dl1": "nizhib",
    "docker": "docker",
    "n01z3-extreme": "x1"
}


def get_paths(correction='./'):
    pcname = socket.gethostname()
    if pcname not in PATHS.keys():
        print('shit, seems its docker')
        pcname = "docker"

    base, diff, path = '../', '', None
    for n in range(4):
        yaml_path = f"{diff}configs/paths/{PATHS[pcname]}.yml"
        if osp.exists(yaml_path):
            path = osp.join(correction, yaml_path)
        diff += base

    with open(path, "r") as stream:
        data_config = yaml.safe_load(stream)
    return data_config


if __name__ == '__main__':
    print(get_paths())
