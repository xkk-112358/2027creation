"""② Full evaluation on K-Radar seq 1 (597 frames) with the L4DR v1.1 checkpoint."""
import json
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, '/root/2027creation/RAF_repo')
os.chdir('/root/2027creation/RAF_repo')

from utils.util_config import cfg, cfg_from_yaml_file            # noqa: E402
from datasets.kradar_detection_v2_2 import KRadarDetection_v2_2  # noqa: E402
from models.skeletons import build_skeleton                      # noqa: E402
from ops.iou3d_nms import iou3d_nms_utils                        # noqa: E402

CKPT = '/root/2027creation/weights/l4dr_kradar_v1.1_model_34.pt'
OUT = '/hy-tmp/eval_seq1.json'
THRS = [0.1, 0.2, 0.3]
IOUS = [0.3, 0.5]


def main():
    c = cfg_from_yaml_file('./configs/RAF_v2_2_local.yml', cfg)
    net_c = cfg_from_yaml_file('./configs/L4DR.yml', EasyDict())
    c.MODEL = net_c.MODEL
    ds = KRadarDetection_v2_2(cfg=c, split='train')
    net = build_skeleton(c).cuda().eval()
    net.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)

    idxs = [i for i, d in enumerate(ds.list_dict_item) if str(d['meta']['seq']) == '1']
    print(f'seq1 frames: {len(idxs)}', flush=True)

    res = {(t, iou): {'tp': 0, 'fp': 0, 'fn': 0} for t in THRS for iou in IOUS}
    n_gt = 0
    n_pred = {t: 0 for t in THRS}
    times = []
    t_start = time.time()

    for k, i in enumerate(idxs, 1):
        item = ds[i]
        batch = ds.collate_fn([item])
        g = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()}

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = net(g)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        p = out['pred_dicts'][0]
        sc = p['pred_scores'].detach().cpu()
        bx = p['pred_boxes'].detach().cpu()
        gt = batch['gt_boxes'][0, :int(batch['num_objs'][0])].cpu()
        n_gt += len(gt)

        for t in THRS:
            keep = sc >= t
            pred = bx[keep]
            n_pred[t] += len(pred)
            for iou in IOUS:
                r = res[(t, iou)]
                if len(pred) == 0:
                    r['fn'] += len(gt)
                    continue
                if len(gt) == 0:
                    r['fp'] += len(pred)
                    continue
                iou_mat = iou3d_nms_utils.boxes_iou_bev(pred[:, :7].cuda(), gt[:, :7].cuda()).cpu().numpy()
                matched_gt = set()
                for pi in range(len(pred)):
                    gi = int(np.argmax(iou_mat[pi]))
                    if iou_mat[pi, gi] >= iou and gi not in matched_gt:
                        r['tp'] += 1
                        matched_gt.add(gi)
                    else:
                        r['fp'] += 1
                r['fn'] += len(gt) - len(matched_gt)

        if k % 25 == 0 or k == len(idxs):
            el = time.time() - t_start
            print(f'[{k}/{len(idxs)}] {el:.0f}s  gt={n_gt}  pred@0.2={n_pred[0.2]}', flush=True)

    summary = {
        'frames': len(idxs), 'gt_total': n_gt, 'det_per_frame': {str(t): n_pred[t] / len(idxs) for t in THRS},
        'infer_ms_mean': float(np.mean(times) * 1000), 'infer_fps': float(1 / np.mean(times)),
        'metrics': {}
    }
    for (t, iou), r in res.items():
        tp, fp, fn = r['tp'], r['fp'], r['fn']
        summary['metrics'][f'score>={t},iou={iou}'] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'recall': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1)
        }
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== seq1 evaluation ===')
    print(f"frames={summary['frames']}  GT={summary['gt_total']}  pred/frame={summary['det_per_frame']}")
    print(f"inference: {summary['infer_ms_mean']:.0f} ms/frame ({summary['infer_fps']:.1f} FPS)")
    for k_, v in summary['metrics'].items():
        print(f"{k_:<22s} recall={v['recall']:.3f}  precision={v['precision']:.3f}  (TP={v['tp']} FP={v['fp']} FN={v['fn']})")
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
