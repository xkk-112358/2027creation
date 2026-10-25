"""RAF/L4DR demo on K-Radar seq 1: run inference, draw BEV + camera overlay, report speed.

Usage:
    cd /root/2027creation/RAF_repo
    PYTHONPATH=$PWD/ops:$PWD python3 ../tools/demo_l4dr.py --frames 30
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, '/root/2027creation/RAF_repo')
os.chdir('/root/2027creation/RAF_repo')

from utils.util_config import cfg, cfg_from_yaml_file          # noqa: E402
from datasets.kradar_detection_v2_2 import KRadarDetection_v2_2  # noqa: E402
from models.skeletons import build_skeleton                     # noqa: E402

OUT = '/root/2027creation/demo'
CKPT = '/root/2027creation/weights/l4dr_kradar_v1.1_model_34.pt'


def build(net_cfg='./configs/L4DR.yml'):
    c = cfg_from_yaml_file('./configs/RAF_v2_2_local.yml', cfg)
    net_c = cfg_from_yaml_file(net_cfg, EasyDict())
    c.MODEL = net_c.MODEL
    ds = KRadarDetection_v2_2(cfg=c, split='train')
    net = build_skeleton(c).cuda().eval()
    net.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
    return c, ds, net


def draw_bev(gt, pred, score, size=(720, 640)):
    h, w = size
    img = np.full((h, w, 3), 255, np.uint8)
    x_min, x_max, y_min, y_max = 0.0, 72.0, -8.0, 8.0

    def xy(x, y):
        return int((x - x_min) / (x_max - x_min) * (w - 1)), int((y_max - y) / (y_max - y_min) * (h - 1))

    for x in range(0, 73, 12):
        p0, p1 = xy(x, y_min), xy(x, y_max)
        cv2.line(img, p0, p1, (235, 235, 235), 1)
        cv2.putText(img, f'{x}m', (p0[0] + 2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.line(img, xy(0, 0), xy(72, 0), (210, 210, 210), 1)

    def box(x, y, l, wd, yaw, color, thick=2):
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[l / 2, wd / 2], [l / 2, -wd / 2], [-l / 2, -wd / 2], [-l / 2, wd / 2]])
        rot = np.array([[c, -s], [s, c]])
        poly = (pts @ rot.T) + np.array([x, y])
        cv2.polylines(img, [np.array([xy(*p) for p in poly], np.int32)], True, color, thick)

    for g in gt:
        box(g[0], g[1], g[3], g[4], g[6], (60, 160, 60), 2)
    for b, s in zip(pred, score):
        box(b[0], b[1], b[3], b[4], b[6], (40, 40, 220), 2)
        cv2.putText(img, f'{s:.2f}', xy(b[0], b[1] + 1.2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 220), 1)
    cv2.putText(img, 'GT: green   Pred: red (>=0.2)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--score', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    c, ds, net = build()
    idxs = [i for i, d in enumerate(ds.list_dict_item) if str(d['meta']['seq']) == '1'][:args.frames]
    print(f'frames: {len(idxs)}  ckpt: {os.path.basename(CKPT)}')

    stats = []
    t0 = time.time()
    for n, i in enumerate(idxs, 1):
        item = ds[i]
        batch = ds.collate_fn([item])
        g = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts = time.time()
        with torch.no_grad():
            out = net(g)
        torch.cuda.synchronize()
        dt = time.time() - ts

        p = out['pred_dicts'][0]
        sc = p['pred_scores'].cpu().numpy()
        bx = p['pred_boxes'].cpu().numpy()
        keep = sc >= args.score
        sc, bx = sc[keep], bx[keep]

        gt = batch['gt_boxes'][0, :int(batch['num_objs'][0])].numpy()
        bev = draw_bev(gt, bx, sc)
        cv2.imwrite(os.path.join(OUT, f'bev_{n:03d}.jpg'), bev)
        stats.append((n, len(gt), len(bx), sc.max() if len(sc) else 0.0, dt))

        if n % 5 == 0:
            print(f'[{n}/{len(idxs)}] gt={len(gt)} pred={len(bx)} max_score={stats[-1][3]:.2f} {dt*1000:.0f}ms')

    total = time.time() - t0
    fps = len(idxs) / total
    print(f'\ndone: {len(idxs)} frames in {total:.1f}s = {fps:.2f} FPS'
          f' (avg inference {np.mean([s[4] for s in stats]) * 1000:.0f} ms)')
    print(f'pred/frame: mean {np.mean([s[2] for s in stats]):.2f}, GT/frame: mean {np.mean([s[1] for s in stats]):.2f}')
    print(f'images -> {OUT}')


if __name__ == '__main__':
    main()
