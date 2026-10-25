"""Convert pc1p.zip (K-Radar 1% polar point clouds) into RAF layout:

    <out>/<seq>/sprdr_<frame>.npy    float32 [N, 4] = x, y, z, power
"""
import io
import os
import sys
import zipfile
import numpy as np

ZIP = '/hy-tmp/kradar_dataset/gdrive/pc1p.zip'
OUT = '/hy-tmp/kradar_dataset/rtnh_wider_1p_1'
ROI = (0.0, 72.0, -6.4, 6.4, -2.0, 6.0)


def main():
    zf = zipfile.ZipFile(ZIP)
    members = [m for m in zf.namelist() if m.endswith('.npy')]
    print(f'{len(members)} frames', flush=True)
    os.makedirs(OUT, exist_ok=True)
    tot_pts = tot_in = 0
    for i, m in enumerate(members):
        seq = m.split('/')[1]
        name = os.path.basename(m)[len('rpc_'):]
        dst = os.path.join(OUT, seq, f'sprdr_{name}')
        if os.path.exists(dst):
            continue
        with zf.open(m) as f:
            arr = np.load(io.BytesIO(f.read()))
        pc = arr[:, :4].astype(np.float32)
        tot_pts += len(pc)
        inroi = ((pc[:, 0] >= ROI[0]) & (pc[:, 0] <= ROI[1]) &
                 (pc[:, 1] >= ROI[2]) & (pc[:, 1] <= ROI[3]) &
                 (pc[:, 2] >= ROI[4]) & (pc[:, 2] <= ROI[5]))
        tot_in += int(inroi.sum())
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, pc)
        if (i + 1) % 2000 == 0:
            print(f'[{i+1}/{len(members)}] roi_ratio={tot_in/max(tot_pts,1):.3f}', flush=True)
    print(f'done: frames={len(members)}, roi_ratio={tot_in/max(tot_pts,1):.3f}', flush=True)


if __name__ == '__main__':
    main()
