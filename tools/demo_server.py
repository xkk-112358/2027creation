"""审计报告 + 交互式推理 Demo 的统一服务。

同源提供：/ 审计报告，/demo 交互页，/api/* 推理接口。
只用 Python 标准库做 HTTP，不引入额外依赖；模型常驻显存，请求串行执行。

启动：
    cd /root/2027creation/RAF_repo
    PYTHONPATH=$PWD/ops:$PWD python3 ../tools/demo_server.py --port 8080
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

sys.path.insert(0, '/root/2027creation/RAF_repo')
os.chdir('/root/2027creation/RAF_repo')
torch.set_grad_enabled(False)

import cv2                                                        # noqa: E402
from easydict import EasyDict                                     # noqa: E402
from utils.util_config import cfg, cfg_from_yaml_file             # noqa: E402
from datasets.kradar_detection_v2_2 import KRadarDetection_v2_2    # noqa: E402
from models.skeletons import build_skeleton                       # noqa: E402
from ops.iou3d_nms import iou3d_nms_utils                         # noqa: E402

WEB_DIR = '/root/2027creation/demo_web'
AUDIT_DIR = '/root/raf-audit'
CKPT = '/root/2027creation/weights/l4dr_kradar_v1.1_model_34.pt'
MAX_UPLOAD_BYTES = 16 * 1024 * 1024     # 单个 base64 字段上限
MAX_POINTS = 400_000                    # 上传点云点数上限

PAGES = {
    '/': AUDIT_DIR + '/raf-audit.html',
    '/index.html': AUDIT_DIR + '/raf-audit.html',
    '/demo': WEB_DIR + '/index.html',
    '/demo/': WEB_DIR + '/index.html',
}
MOUNTS = (
    ('/assets/', AUDIT_DIR + '/assets/'),
    ('/_shared/', AUDIT_DIR + '/_shared/'),
    ('/fonts/', WEB_DIR + '/fonts/'),
)
MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.ttf': 'font/ttf',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.json': 'application/json; charset=utf-8',
}

LOCK = threading.Lock()
STATE = {}

# ---------- BEV 画布几何 ----------
CANVAS_W, CANVAS_H = 1120, 330
PAD_L, PAD_T, PAD_B = 62, 52, 49
X_MIN, X_MAX = 0.0, 72.0
Y_MIN, Y_MAX = -8.0, 8.0
SCALE = (CANVAS_W - PAD_L - 28) / (X_MAX - X_MIN)      # px / m（等比）
COL_GT = (84, 124, 28)          # #1C7C54
COL_PRED = (48, 34, 166)        # #A62230
COL_PT = (196, 176, 150)
COL_RULE = (226, 227, 226)
COL_INK = (29, 24, 22)
COL_MUTED = (114, 100, 93)


def _xy(x, y):
    return int(round(PAD_L + (x - X_MIN) * SCALE)), int(round(PAD_T + (Y_MAX - y) * SCALE))


# ---------- 启动：加载配置 / 数据集 / 模型 ----------
def build():
    c = cfg_from_yaml_file('./configs/RAF_v2_2_local.yml', cfg)
    net_c = cfg_from_yaml_file('./configs/L4DR.yml', EasyDict())
    c.MODEL = net_c.MODEL
    ds = KRadarDetection_v2_2(cfg=c, split='train')
    net = build_skeleton(c).cuda().eval()
    miss, unexp = net.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
    idxs = [i for i, d in enumerate(ds.list_dict_item) if str(d['meta']['seq']) == '1']
    STATE.update(ds=ds, net=net, idxs=idxs, cfg=c.DATASET,
                 weather=_weather(), missing=len(miss), unexpected=len(unexp))
    print(f'[demo] ready: {len(idxs)} frames, missing={len(miss)}, unexpected={len(unexp)}', flush=True)

    # 预热：首次前向会触发 CUDA kernel 初始化，先跑一遍，避免用户第一次点击等 3 秒
    t = time.time()
    try:
        infer(0, 0.2)
        print(f'[demo] warmup {((time.time() - t) * 1000):.0f} ms', flush=True)
    except Exception as e:
        print(f'[demo] warmup failed: {e}', flush=True)


def _weather():
    try:
        line = open('/hy-tmp/kradar_dataset/seqs/1/description.txt').readline()
        road, when, climate = line.split(',')
        return {'road': road, 'time': when, 'climate': climate.strip()}
    except Exception:
        return {}


# ---------- 输入替换 ----------
def apply_radar(ds, pts):
    pts = np.asarray(pts, np.float32)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError('雷达点云必须是 [N, 4]（x, y, z, power）')
    x0, y0, z0, x1, y1, z1 = ds.roi.xyz
    m = ((pts[:, 0] > x0) & (pts[:, 0] < x1) &
         (pts[:, 1] > y0) & (pts[:, 1] < y1) &
         (pts[:, 2] > z0) & (pts[:, 2] < z1))
    pts = pts[m][:, :4]
    if len(pts) == 0:
        raise ValueError('上传的雷达点云在 ROI 内没有有效点')
    return ds.sample_points(pts), len(pts)


def apply_lidar(ds, item, pts):
    pts = np.asarray(pts, np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError('LiDAR 点云必须是 [N, >=3]（x, y, z[, intensity]）')
    pc = np.zeros((len(pts), ds.ldr64.n_attr), np.float32)
    k = min(pts.shape[1], ds.ldr64.n_attr)
    pc[:, :k] = pts[:, :k]
    m = (pc[:, 0] > 0.01) | (pc[:, 0] < -0.01) | (pc[:, 1] > 0.01) | (pc[:, 1] < -0.01)
    pc = pc[m]
    if len(pc) == 0:
        raise ValueError('上传的 LiDAR 点云为空')
    dx, dy, dz = item['meta']['calib']
    pc[:, :3] += np.array([dx, dy, dz], np.float32)
    return pc


def apply_image(ds, bgr):
    ds_ = ds
    img = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    if getattr(ds_, 'map_x', None) is not None:
        img = cv2.remap(img, ds_.map_x, ds_.map_y, cv2.INTER_LINEAR)
    img = img[ds_.y_crop:, :, :]
    img = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
    f = img.astype(np.float32)
    f = (f - np.array(ds_.mean, np.float32)) / np.array(ds_.std, np.float32)
    return np.transpose(f, (2, 0, 1))


# ---------- BEV 渲染 ----------
def draw_bev(gt, pred, scores, radar, thr):
    W, H = CANVAS_W, CANVAS_H
    img = np.full((H, W, 3), 250, np.uint8)

    # 雷达点（底图）
    if radar is not None and len(radar):
        p = radar[:, :3]
        m = (p[:, 0] >= X_MIN) & (p[:, 0] <= X_MAX) & (p[:, 1] >= Y_MIN) & (p[:, 1] <= Y_MAX)
        for x, y, _ in p[m][::2]:
            cv2.circle(img, _xy(x, y), 1, COL_PT, -1)

    # 网格
    for x in range(0, 73, 12):
        cv2.line(img, _xy(x, Y_MIN), _xy(x, Y_MAX), COL_RULE, 1)
        cv2.putText(img, f'{x}m', (_xy(x, Y_MIN)[0] - 8, H - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_MUTED, 1, cv2.LINE_AA)
    cv2.line(img, _xy(0, 0), _xy(72, 0), (205, 206, 208), 1)

    # 自车
    ego = np.array([_xy(0.0, 0.9), _xy(0.0, -0.9), _xy(1.6, 0.0)])
    cv2.fillPoly(img, [ego], (90, 70, 40))
    cv2.putText(img, 'ego', (_xy(0.0, 0.0)[0] - 6, _xy(0.0, 0.0)[1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_MUTED, 1, cv2.LINE_AA)

    def box(b, color, thick=2):
        x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
        cs, sn = np.cos(yaw), np.sin(yaw)
        pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        rot = np.array([[cs, -sn], [sn, cs]])
        poly = (pts @ rot.T) + np.array([x, y])
        cv2.polylines(img, [np.array([_xy(px, py) for px, py in poly], np.int32)], True,
                      color, thick, cv2.LINE_AA)
        # 朝向刻线
        cv2.line(img, _xy(x, y), _xy(x + l / 2 * cs, y + l / 2 * sn), color, 1, cv2.LINE_AA)

    for g in gt:
        box(g, COL_GT)
    for b, s in zip(pred, scores):
        box(b, COL_PRED)
        if s >= thr:
            px, py = _xy(b[0], b[1])
            cv2.putText(img, f'{s:.2f}', (px + 4, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, COL_PRED, 1, cv2.LINE_AA)

    # 图例
    ly = 24
    cv2.line(img, (PAD_L, ly - 4), (PAD_L + 18, ly - 4), COL_GT, 2)
    cv2.putText(img, f'GT  {len(gt)}', (PAD_L + 24, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (PAD_L + 110, ly - 4), (PAD_L + 128, ly - 4), COL_PRED, 2)
    cv2.putText(img, f'Pred  {len(pred)}  (score >= {thr})', (PAD_L + 134, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_MUTED, 1, cv2.LINE_AA)
    cv2.circle(img, (PAD_L + 340, ly - 4), 2, COL_PT, -1)
    cv2.putText(img, f'Radar points  {0 if radar is None else len(radar)}', (PAD_L + 348, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, 'x: forward (m)   y: lateral (m)', (CANVAS_W - 250, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_MUTED, 1, cv2.LINE_AA)
    return img


def metrics(gt, pred, thr=0.3):
    """贪心匹配，BEV IoU >= thr，与 eval_seq1.py 同口径。"""
    n_gt, n_pred = len(gt), len(pred)
    if n_pred == 0:
        return {'tp': 0, 'fp': 0, 'fn': n_gt, 'recall': 0.0, 'precision': 0.0}
    if n_gt == 0:
        return {'tp': 0, 'fp': n_pred, 'fn': 0, 'recall': 0.0, 'precision': 0.0}
    iou = iou3d_nms_utils.boxes_iou_bev(
        torch.tensor(pred[:, :7], dtype=torch.float32).cuda(),
        torch.tensor(gt[:, :7], dtype=torch.float32).cuda()).cpu().numpy()
    used, tp = set(), 0
    for i in range(n_pred):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= thr and j not in used:
            tp += 1
            used.add(j)
    fp = n_pred - tp
    fn = n_gt - len(used)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'recall': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1)}


def _roi_count_of_sample(ds, item):
    """样例雷达在 ROI 内的真实回波点数（采样前），用于说明密度。"""
    try:
        p = os.path.join(ds.rdr_sparse.dir, item['meta']['seq'],
                         f"sprdr_{item['meta']['idx']['rdr']}.npy")
        a = np.load(p)
        x0, y0, z0, x1, y1, z1 = ds.roi.xyz
        m = ((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) &
             (a[:, 1] < y1) & (a[:, 2] > z0) & (a[:, 2] < z1))
        return int(m.sum())
    except Exception:
        return None


def infer(idx, score_thr, radar=None, lidar=None, image=None, lidar_fmt=None):
    ds, net, idxs = STATE['ds'], STATE['net'], STATE['idxs']
    if idx < 0 or idx >= len(idxs):
        raise ValueError(f'帧序号越界：{idx}（有效范围 0..{len(idxs) - 1}）')

    t_all = time.time()
    item = ds[idxs[idx]]
    src = {'radar': 'sample', 'lidar': 'sample', 'image': 'sample'}

    if radar is not None:
        item['rdr_sparse'], n_raw = apply_radar(ds, radar)
        radar_roi = n_raw
        src['radar'] = f'upload({n_raw} pts in ROI)'
    else:
        radar_roi = _roi_count_of_sample(ds, item)
    if lidar is not None:
        item['ldr64'] = apply_lidar(ds, item, lidar)
        src['lidar'] = f'upload({len(item["ldr64"])} pts' + (f' · {lidar_fmt}' if lidar_fmt else '') + ')'
    if image is not None:
        item['front0'] = apply_image(ds, image)
        src['image'] = 'upload'

    n_radar = len(item['rdr_sparse'])
    n_lidar = len(item['ldr64'])

    batch = ds.collate_fn([item])
    g = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    torch.cuda.synchronize()
    t0 = time.time()
    out = net(g)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t0) * 1000

    p = out['pred_dicts'][0]
    sc_all = p['pred_scores'].detach().cpu().numpy()
    bx_all = p['pred_boxes'].detach().cpu().numpy()
    keep = sc_all >= score_thr
    sc, bx = sc_all[keep], bx_all[keep]
    order = np.argsort(-sc)
    sc, bx = sc[order], bx[order]

    n_obj = int(batch['num_objs'][0])
    gt = batch['gt_boxes'][0, :n_obj].numpy()

    m = metrics(gt, bx)
    bev = draw_bev(gt, bx, sc, item['rdr_sparse'], score_thr)
    ok_png, buf = cv2.imencode('.png', bev)
    if not ok_png:
        raise RuntimeError('BEV 编码失败')

    cam_jpg = _camera_b64(item['meta']['path']['front'])

    return {
        'ok': True,
        'frame': {'index': idx, 'frame_id': item['meta']['idx']['rdr'],
                  'lidar_id': item['meta']['idx']['ldr64'],
                  'cam_id': item['meta']['idx']['camf'],
                  'total': len(idxs), 'climate': item['meta']['desc']['climate']},
        'score_thr': score_thr,
        'input': {'source': src, 'radar_points': n_radar, 'radar_roi_points': radar_roi,
                  'lidar_points': n_lidar},
        'timing': {'infer_ms': round(infer_ms, 1), 'total_ms': round((time.time() - t_all) * 1000, 1)},
        'metrics': m,
        'det': [{'x': round(float(b[0]), 2), 'y': round(float(b[1]), 2), 'z': round(float(b[2]), 2),
                 'l': round(float(b[3]), 2), 'w': round(float(b[4]), 2), 'h': round(float(b[5]), 2),
                 'yaw': round(float(b[6]), 3), 'score': round(float(s), 3)} for b, s in zip(bx, sc)],
        'gt': [{'x': round(float(b[0]), 2), 'y': round(float(b[1]), 2),
                'l': round(float(b[3]), 2), 'w': round(float(b[4]), 2),
                'yaw': round(float(b[6]), 3)} for b in gt],
        'bev_png': 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode(),
        'camera_jpg': cam_jpg,
    }


def _camera_b64(path, w=640, h=360):
    img = cv2.imread(path)
    if img is None:
        return None
    img = img[:, :1280, :]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode() if ok else None


def frame_list():
    ds, idxs = STATE['ds'], STATE['idxs']
    out = []
    for k, i in enumerate(idxs):
        meta = ds.list_dict_item[i]['meta']
        idx = meta['idx']
        out.append({'index': k, 'frame_id': idx['rdr'], 'lidar_id': idx['ldr64'],
                    'cam_id': idx['camf'], 'n_gt': meta['num_obj']})
    return out


# ---------- 检测结果分析（本地规则 + 可插拔 AI） ----------
# 配置优先级：环境变量 > tools/ai.env（该文件权限 600，不要提交到仓库）
def _load_env_file(path):
    if not os.path.isfile(path):
        return
    try:
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


_load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai.env'))

AI_API_KEY = os.environ.get('AI_API_KEY', '').strip()
AI_BASE_URL = os.environ.get('AI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
AI_MODEL = os.environ.get('AI_MODEL', 'gpt-4o-mini')
AI_TIMEOUT = float(os.environ.get('AI_TIMEOUT', '90'))


def _bev_iou(a, b):
    """旋转框 BEV IoU（cv2 实现，不依赖 CUDA）。a/b = [x, y, w, l, yaw]"""
    ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), float(a[4]) * 57.2957795)
    rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(b[4]) * 57.2957795)
    ret, region = cv2.rotatedRectangleIntersection(ra, rb)
    if ret == 0 or region is None:
        return 0.0
    inter = abs(cv2.contourArea(cv2.convexHull(region)))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def _match(det, gt, thr=0.3):
    used, pairs = set(), []
    for i, d in enumerate(det):
        best, bj = 0.0, -1
        for j, g in enumerate(gt):
            if j in used:
                continue
            v = _bev_iou(d, g)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= thr:
            pairs.append((i, bj, best))
            used.add(bj)
    return pairs, used


def _local_analysis(p):
    det = p.get('det') or []
    gt = p.get('gt') or []
    thr = p.get('score_thr', 0.2)
    fr = p.get('frame') or {}
    inp = p.get('input') or {}
    src = inp.get('source') or {}

    D = [[d['x'], d['y'], d['w'], d['l'], d['yaw']] for d in det]
    G = [[g['x'], g['y'], g['w'], g['l'], g['yaw']] for g in gt]
    pairs, used = _match(D, G, 0.3)
    tp, fp, fn = len(pairs), len(D) - len(pairs), len(G) - len(pairs)
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)

    L = []
    L.append(f"帧 {fr.get('frame_id', '?')}（雷达）/ {fr.get('lidar_id', '?')}（LiDAR）/ "
             f"{fr.get('cam_id', '?')}（相机）· 天气 {fr.get('climate', '?')} · score 阈值 {thr}")
    L.append(f"输入来源：雷达 {src.get('radar', '?')} / LiDAR {src.get('lidar', '?')} / "
             f"相机 {src.get('image', '?')}；雷达点 {inp.get('radar_points', '?')}，"
             f"LiDAR 点 {inp.get('lidar_points', '?')}")

    if not G and not D:
        L.append("本帧既无真值也无预测，无法评价。")
        return '\n'.join(L)

    L.append("")
    if not G:
        L.append(f"· 本帧无真值目标，模型却给出 {len(D)} 个预测，全部计为误检（precision 0%）。")
    elif not D:
        L.append(f"· 本帧 {len(G)} 个真值，阈值 {thr} 下无任何预测，全部漏检（recall 0%）。")
    else:
        L.append(f"· 命中 {tp} 个（IoU≥0.3），漏检 {fn} 个，误检 {fp} 个 → "
                 f"Recall {rec*100:.1f}%，Precision {pre*100:.1f}%。")

    # 匹配对的几何偏差
    if pairs:
        dl = np.mean([abs(D[i][3] - G[j][3]) for i, j, _ in pairs])
        dw = np.mean([abs(D[i][2] - G[j][2]) for i, j, _ in pairs])
        dc = np.mean([np.hypot(D[i][0] - G[j][0], D[i][1] - G[j][1]) for i, j, _ in pairs])
        ious = np.mean([v for _, _, v in pairs])
        L.append(f"· 命中框的平均偏差：中心 {dc:.2f} m，长度 {dl:.2f} m，宽度 {dw:.2f} m，"
                 f"平均 IoU {ious:.2f}。")
        if dl > 1.0 or dw > 0.6:
            L.append("  尺寸系统性偏大，是 IoU 0.5 下指标骤降的主因（框位置对、边界不准）。")

    # 漏检目标离自车多远
    if fn:
        far = [np.hypot(G[j][0], G[j][1]) for j in range(len(G)) if j not in used]
        L.append(f"· 漏检目标的纵向距离：{'、'.join(f'{d:.0f}m' for d in sorted(far))}"
                 f"；最近 {min(far):.0f}m，最远 {max(far):.0f}m。")

    # 误检分布
    if fp:
        fpd = [np.hypot(D[i][0], D[i][1]) for i in range(len(D)) if i not in {p[0] for p in pairs}]
        L.append(f"· 误检目标距离：{'、'.join(f'{d:.0f}m' for d in sorted(fpd))}。")

    # 雷达稀疏度提示
    rp = inp.get('radar_roi_points')
    if isinstance(rp, (int, float)) and rp:
        L.append(f"· 雷达输入：ROI 内真实回波 {int(rp)} 点，采样补足到 {inp.get('radar_points')} 点；"
                 f"论文训练用笛卡尔稀疏雷达，密度显著更高，因此本帧指标不代表论文水平。")

    L.append("")
    if rec < 0.3:
        L.append("结论：本帧召回偏低，优先怀疑雷达密度不足或目标过远。")
    elif pre < 0.5:
        L.append("结论：本帧误检偏多，可能是近处杂波或框尺寸偏大造成的重叠。")
    else:
        L.append("结论：本帧检测质量正常，可用于展示。")

    L.append("说明：以上为基于数值的本地规则分析，未调用视觉大模型；"
             "配置 AI_API_KEY 后同一按钮会自动改用多模态模型解读 BEV 图。")
    return '\n'.join(L)


def _ai_analysis(p, local_text):
    img = p.get('bev_png') or ''
    if not img.startswith('data:image/'):
        raise ValueError('缺少 BEV 图片')
    facts = local_text.split('\n说明：')[0]      # 只把客观数值喂给模型，去掉本地分析的自述
    prompt = (
        "你是一名 3D 目标检测结果分析助手。下图是 K-Radar 数据集上一帧的 BEV（鸟瞰）检测结果："
        "灰点是 4D 毫米波雷达点云，绿框是真值，红框是模型预测（框上数字为置信度），"
        "横轴是自车前方距离 0–72m，纵轴是左右 ±8m。\n\n"
        "已经算好的客观数值如下，请以它为准，不要自行编造数字：\n"
        f"{facts}\n\n"
        "请用中文输出四段，总字数控制在 300 字以内：\n"
        "① 一句话结论；② 预测与真值的差异（位置、大小、朝向）；③ 最可能的原因；"
        "④ 使用该结果时需要注意的局限（例如雷达密度、单帧样本量）。"
        "不要输出 Markdown 表格，用短句和项目符号；"
        "请直接给出最终四段分析，不要输出思考过程、草稿或自我检查。"
    )

    def call(max_tokens):
        payload = {
            'model': AI_MODEL,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': img}},
            ]}],
        }
        req = urllib.request.Request(
            AI_BASE_URL + '/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {AI_API_KEY}'})
        with urllib.request.urlopen(req, timeout=AI_TIMEOUT) as r:
            d = json.loads(r.read().decode('utf-8'))
        ch = d['choices'][0]
        return (ch['message'].get('content') or '').strip(), ch.get('finish_reason')

    text, fr = call(1200)
    if not text:                      # 推理模型偶尔只吐思考内容，放宽上限重试一次
        text, fr = call(2000)
    if not text:
        raise ValueError(f'模型未返回正文（finish_reason={fr}）')
    return text



# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    server_version = 'RAFDemo/1.0'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == '/api/meta':
                return self._json(200, {'ok': True, 'seq': '1', 'weather': STATE.get('weather', {}),
                                        'frames': frame_list(),
                                        'ckpt': os.path.basename(CKPT),
                                        'ai_enabled': bool(AI_API_KEY),
                                        'ai_model': AI_MODEL if AI_API_KEY else None,
                                        'missing': STATE.get('missing', 0)})
            if u.path == '/api/camera':
                idx = int(q.get('idx', ['0'])[0])
                if idx < 0 or idx >= len(STATE['idxs']):
                    return self._json(400, {'ok': False, 'error': '帧序号越界'})
                item = STATE['ds'][STATE['idxs'][idx]]
                b64 = _camera_b64(item['meta']['path']['front'])
                return self._json(200, {'ok': True, 'camera_jpg': b64,
                                        'cam_id': item['meta']['idx']['camf']})
            if u.path == '/favicon.ico':
                return self._send(204, b'', 'image/x-icon')
            return self._static(u.path)
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {'ok': False, 'error': str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == '/api/analyze':
            return self._analyze()
        if u.path != '/api/infer':
            return self._json(404, {'ok': False, 'error': 'not found'})
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > MAX_UPLOAD_BYTES * 3:
                return self._json(400, {'ok': False, 'error': '请求体为空或过大'})
            body = json.loads(self.rfile.read(n).decode('utf-8'))
        except Exception:
            return self._json(400, {'ok': False, 'error': '请求体不是合法 JSON'})

        try:
            idx = int(body.get('idx', 0))
            thr = float(body.get('score_thr', 0.2))
            thr = min(max(thr, 0.05), 0.9)

            radar = _decode_npy(body.get('radar_b64'), '雷达 .npy')
            lidar, lidar_fmt = _decode_lidar(body.get('lidar_b64'))
            image = _decode_image(body.get('image_b64'))
            for name, arr in (('雷达', radar), ('LiDAR', lidar)):
                if arr is not None and len(arr) > MAX_POINTS:
                    return self._json(400, {'ok': False, 'error': f'{name}点数超过 {MAX_POINTS} 上限'})

            with LOCK:
                res = infer(idx, thr, radar, lidar, image, lidar_fmt)
            return self._json(200, res)
        except ValueError as e:
            return self._json(400, {'ok': False, 'error': str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {'ok': False, 'error': f'{type(e).__name__}: {e}'})

    def _analyze(self):
        """结果分析：有 AI key 走多模态模型，否则用本地规则。"""
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > MAX_UPLOAD_BYTES * 3:
                return self._json(400, {'ok': False, 'error': '请求体为空或过大'})
            body = json.loads(self.rfile.read(n).decode('utf-8'))
        except Exception:
            return self._json(400, {'ok': False, 'error': '请求体不是合法 JSON'})
        try:
            local = _local_analysis(body)
            text, source = local, '本地规则'
            if AI_API_KEY:
                try:
                    text = _ai_analysis(body, local)
                    source = f'AI · {AI_MODEL}'
                except Exception as e:
                    traceback.print_exc()
                    text = local + f'\n\n（AI 调用失败，已回退本地分析：{type(e).__name__}: {e}）'
                    source = '本地规则（AI 失败）'
            return self._json(200, {'ok': True, 'text': text, 'source': source})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {'ok': False, 'error': f'{type(e).__name__}: {e}'})

    def _static(self, path):
        """审计报告与 Demo 同源：/ 是报告，/demo 是交互页。"""
        if path in PAGES:
            return self._send_file(PAGES[path])
        for prefix, base in MOUNTS:
            if path.startswith(prefix):
                return self._send_file(os.path.join(base, path[len(prefix):]), base)
        return self._json(404, {'ok': False, 'error': 'not found'})

    def _send_file(self, path, base=None):
        path = os.path.normpath(path)
        if base is not None:
            base = os.path.normpath(base)
            if not path.startswith(base + os.sep):
                return self._json(403, {'ok': False, 'error': 'forbidden'})
        if not os.path.isfile(path):
            return self._json(404, {'ok': False, 'error': 'not found'})
        with open(path, 'rb') as f:
            return self._send(200, f.read(), MIME.get(os.path.splitext(path)[1].lower(),
                                                      'application/octet-stream'))


def _decode_npy(b64, what='.npy'):
    if not b64:
        return None
    raw = base64.b64decode(b64)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError('文件过大')
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        raise ValueError(f'无法解析 {what} 文件，请用 numpy.save 保存数组后再上传')


def _decode_lidar(b64):
    """LiDAR 既接受 .npy，也接受 ASCII 格式的 .pcd（自动解析 x/y/z/intensity）。"""
    if not b64:
        return None, None
    raw = base64.b64decode(b64)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError('文件过大（上限 16MB）')
    if raw[:6] == b'\x93NUMPY':
        try:
            return np.load(io.BytesIO(raw), allow_pickle=False), 'npy'
        except Exception:
            raise ValueError('无法解析 .npy 文件，请用 numpy.save 保存数组后再上传')
    return _parse_pcd(raw), 'pcd'


def _parse_pcd(raw):
    text = raw.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    fields, n_points, data_type, data_start = None, None, None, None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        key = parts[0].upper()
        if key == 'FIELDS':
            fields = parts[1:]
        elif key == 'POINTS' and len(parts) > 1:
            n_points = int(float(parts[1]))
        elif key == 'DATA':
            data_type = (parts[1].lower() if len(parts) > 1 else 'ascii')
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError('不是合法的 PCD 文件（缺少 DATA 行）')
    if data_type != 'ascii':
        raise ValueError(f'只支持 ASCII 格式的 PCD（当前为 {data_type}），请先转换')
    if not fields:
        raise ValueError('PCD 缺少 FIELDS 行')

    low = [f.lower() for f in fields]
    for need in ('x', 'y', 'z'):
        if need not in low:
            raise ValueError(f'PCD 缺少 {need} 字段')
    vals = np.array(' '.join(lines[data_start:]).split(), dtype=np.float64)
    if vals.size == 0:
        raise ValueError('PCD 没有数据行')
    if vals.size % len(fields):
        raise ValueError('PCD 数据列数与 FIELDS 不一致')
    arr = vals.reshape(-1, len(fields))
    if n_points and len(arr) > n_points:
        arr = arr[:n_points]

    out = np.zeros((len(arr), 4), np.float32)
    out[:, 0] = arr[:, low.index('x')]
    out[:, 1] = arr[:, low.index('y')]
    out[:, 2] = arr[:, low.index('z')]
    for k in ('intensity', 'i'):
        if k in low:
            out[:, 3] = arr[:, low.index(k)]
            break
    return out


def _decode_image(b64):
    if not b64:
        return None
    raw = base64.b64decode(b64)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError('图片过大')
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('图片无法解码（仅支持 jpg / png）')
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    build()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'[demo] serving http://{args.host}:{args.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
