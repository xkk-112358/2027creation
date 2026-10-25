"""① Radar statistics across the 7 weather conditions (all 58 sequences)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

RADAR_DIR = '/hy-tmp/kradar_dataset/rtnh_wider_1p_1'
SEQ_DIR = '/hy-tmp/kradar_dataset/seqs'
ROI = (0.0, 72.0, -6.4, 6.4, -2.0, 6.0)
MAX_FRAMES_PER_SEQ = 200


def weather(seq):
    p = os.path.join(SEQ_DIR, seq, 'description.txt')
    return open(p).readline().split(',')[2].strip()


def main():
    stats = defaultdict(lambda: {'frames': 0, 'pts': [], 'roi': [], 'pw': [], 'rng': []})
    seqs = sorted([d for d in os.listdir(RADAR_DIR) if d.isdigit()], key=int)
    for seq in seqs:
        files = sorted(glob.glob(os.path.join(RADAR_DIR, seq, '*.npy')))[:MAX_FRAMES_PER_SEQ]
        if not files:
            continue
        w = weather(seq)
        for f in files:
            a = np.load(f)
            m = ((a[:, 0] >= ROI[0]) & (a[:, 0] <= ROI[1]) &
                 (a[:, 1] >= ROI[2]) & (a[:, 1] <= ROI[3]) &
                 (a[:, 2] >= ROI[4]) & (a[:, 2] <= ROI[5]))
            rng = np.linalg.norm(a[:, :3], axis=1)
            s = stats[w]
            s['frames'] += 1
            s['pts'].append(len(a))
            s['roi'].append(int(m.sum()))
            s['pw'].append(float(a[:, 3].mean()))
            s['rng'].append(float(np.percentile(rng, 95)))
        print(f'{seq:>3s} {w:<10s} frames={len(files)}', flush=True)

    order = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
    rows = []
    print('\n%-10s %8s %12s %12s %12s %10s' % ('weather', 'frames', 'pts/frame', 'roi/frame', 'mean_pw', 'r95(m)'))
    for w in order:
        s = stats[w]
        if not s['frames']:
            continue
        row = dict(weather=w, frames=s['frames'],
                   pts=float(np.mean(s['pts'])), roi=float(np.mean(s['roi'])),
                   pw=float(np.mean(s['pw'])), r95=float(np.mean(s['rng'])))
        rows.append(row)
        print('%-10s %8d %12.0f %12.0f %12.4f %10.1f' % (w, row['frames'], row['pts'], row['roi'], row['pw'], row['r95']))

    with open('/hy-tmp/radar_weather_stats.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print('\nsaved -> /hy-tmp/radar_weather_stats.json')


if __name__ == '__main__':
    main()
