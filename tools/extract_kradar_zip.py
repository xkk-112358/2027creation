"""Resumable + parallel extraction of selected folders from the remote K-Radar zip (seq 1)."""
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip

URL = os.environ['ZIP_URL']
COOKIE = os.environ['NAS_COOKIE']
PROXY = 'http://127.0.0.1:7890'
OUT = os.environ.get('OUT_DIR', '/hy-tmp/kradar_dataset/seqs/1')
PREFIX = '1/'
CUBE_LIMIT = int(os.environ.get('CUBE_LIMIT', '20'))
WORKERS = int(os.environ.get('WORKERS', '6'))


def make_zip():
    sess = requests.Session()
    sess.proxies = {'http': PROXY, 'https': PROXY}
    return RemoteZip(URL, headers={'Cookie': COOKIE}, session=sess, support_suffix_range=True)


def fetch_one(z, name, size, rel):
    tgt = os.path.join(OUT, rel)
    if os.path.exists(tgt) and os.path.getsize(tgt) == size:
        return 'skip'
    os.makedirs(os.path.dirname(tgt) or OUT, exist_ok=True)
    with z.open(name) as s, open(tgt + '.part', 'wb') as d:
        while True:
            b = s.read(1 << 20)
            if not b:
                break
            d.write(b)
    os.replace(tgt + '.part', tgt)
    return 'ok'


def worker(task_list):
    z = make_zip()
    ok = 0
    for name, size, rel in task_list:
        for attempt in range(5):
            try:
                st = fetch_one(z, name, size, rel)
                ok += 1
                if st == 'ok' and size > 5e7:
                    print(f'{rel} {size/1e6:.1f}MB ok', flush=True)
                break
            except Exception as e:
                print(f'retry {attempt} {rel}: {type(e).__name__} {str(e)[:60]}', flush=True)
                time.sleep(3 * (attempt + 1))
                try:
                    z.close()
                except Exception:
                    pass
                z = make_zip()
        else:
            print(f'FAILED {rel}', flush=True)
    try:
        z.close()
    except Exception:
        pass
    return ok


def main():
    z = make_zip()
    infos = [i for i in z.infolist() if not i.is_dir()]
    z.close()
    cubes = sorted([i for i in infos if i.filename.startswith('1/radar_zyx_cube/')], key=lambda x: x.filename)[:CUBE_LIMIT]
    others = [i for i in infos if i.filename.startswith(('1/cam-front/', '1/os2-64/'))]
    todo = [(i.filename, i.file_size, i.filename[len(PREFIX):]) for i in others + cubes]
    print(f'targets: {len(todo)} files, {sum(t[1] for t in todo)/1e9:.2f} GB, workers={WORKERS}', flush=True)

    chunks = [todo[i::WORKERS] for i in range(WORKERS)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        total = sum(ex.map(worker, chunks))
    print(f'done {total}/{len(todo)} in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
