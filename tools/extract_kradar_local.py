"""Extract only the needed folders of K-Radar seq 1 from the remote 218GB zip.

Local usage (Windows / macOS / Linux):
    pip install remotezip requests
    python extract_kradar_local.py --url "<下载直链>" --cookie "id=<cookie值>" --out ./kradar_seq1

Options:
    --cubes 20      how many radar_zyx_cube frames to fetch (default 20)
    --no-cam        skip cam-front images
    --no-lpc        skip os2-64 lidar
"""
import argparse
import os
import sys
import time

import requests
from remotezip import RemoteZip

PREFIX = '1/'


def make_zip(url, cookie):
    sess = requests.Session()
    headers = {'Cookie': cookie} if cookie else {}
    return RemoteZip(url, headers=headers, session=sess, support_suffix_range=True)


def fetch(z, name, size, out, rel):
    tgt = os.path.join(out, rel)
    if os.path.exists(tgt) and os.path.getsize(tgt) == size:
        return 'skip'
    os.makedirs(os.path.dirname(tgt) or out, exist_ok=True)
    with z.open(name) as s, open(tgt + '.part', 'wb') as d:
        while True:
            b = s.read(1 << 20)
            if not b:
                break
            d.write(b)
    os.replace(tgt + '.part', tgt)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--cookie', default='')
    ap.add_argument('--out', default='./kradar_seq1')
    ap.add_argument('--cubes', type=int, default=20)
    ap.add_argument('--no-cam', action='store_true')
    ap.add_argument('--no-lpc', action='store_true')
    args = ap.parse_args()

    z = make_zip(args.url, args.cookie)
    infos = [i for i in z.infolist() if not i.is_dir()]
    print(f'zip entries: {len(infos)}')

    want_prefix = ['1/info_label/', '1/info_calib/', '1/description.txt']
    if not args.no_cam:
        want_prefix.append('1/cam-front/')
    if not args.no_lpc:
        want_prefix.append('1/os2-64/')

    todo = [(i.filename, i.file_size) for i in infos if i.filename.startswith(tuple(want_prefix))]

    # pick radar cubes that actually have labels (seq1 frame ids start from 00033, not 00001)
    label_ids = sorted({os.path.basename(i.filename).split('_')[0]
                        for i in infos if i.filename.startswith('1/info_label/') and i.filename.endswith('.txt')})
    cube_map = {os.path.basename(i.filename).split('_')[1].split('.')[0]: i
                for i in infos if i.filename.startswith('1/radar_zyx_cube/') and not i.is_dir()}
    picked = [cube_map[fid] for fid in label_ids if fid in cube_map][:args.cubes]
    print(f'labeled frames: {len(label_ids)}, cubes available: {len(cube_map)}, picked: {len(picked)}')
    if picked:
        print('cube range:', os.path.basename(picked[0].filename), '->', os.path.basename(picked[-1].filename))
    todo += [(i.filename, i.file_size) for i in picked]

    total = sum(s for _, s in todo)
    print(f'targets: {len(todo)} files, {total / 1e9:.2f} GB')
    t0 = time.time()
    ok = 0
    for idx, (name, size) in enumerate(todo, 1):
        rel = name[len(PREFIX):]
        for attempt in range(6):
            try:
                fetch(z, name, size, args.out, rel)
                ok += 1
                break
            except Exception as e:
                print(f'  retry {attempt} {rel}: {type(e).__name__}')
                time.sleep(3 * (attempt + 1))
                try:
                    z.close()
                except Exception:
                    pass
                z = make_zip(args.url, args.cookie)
        if idx % 20 == 0 or size > 5e7:
            print(f'[{idx}/{len(todo)}] {rel} ({size/1e6:.1f}MB) {time.time()-t0:.0f}s')
    print(f'done: {ok}/{len(todo)} -> {args.out}')


if __name__ == '__main__':
    main()
