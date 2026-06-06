"""R3D-18 ⊕ pose-GRU two-stream 행동 분류 (T=16). image+pose feature-fusion.
RGB 클립 -> R3D-18(Kinetics, fine-tune) -> 512-d ┐
중앙16프레임 YOLOv8-pose 51-d/frame -> GRU -> 128-d ┴ concat -> head -> 12클래스.
baseline12 R3D-18(image-only, 0.69)과 동일 recipe/split 로 pose 순효과만 측정. 실행:
  PYTHONPATH=. python -m experiments.r3d_pose
pose 는 각 클립의 결정적 중앙 16프레임(eval 과 동일 프레임)에서 1회 추출·캐시.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import ETRIClipDataset, compute_train_stats, scan_etri_root   # noqa: E402
from metrics import class_counts, class_weights_from_counts, evaluate     # noqa: E402
from model import to_model_input                                          # noqa: E402
from splits import assert_no_leakage, group_split                         # noqa: E402
from transforms import build_eval_transform, build_train_transform        # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.class_map import remap                                      # noqa: E402

from torchvision.models.video import R3D_18_Weights, r3d_18              # noqa: E402
from ultralytics import YOLO                                             # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ETRI = os.path.join(ROOT, "etri")
OUT = os.path.join(ROOT, "runs", "r3d_pose")
CACHE = os.path.join(OUT, "pose_cache.pt")
POSE_W = os.path.join(ROOT, "yolov8s-pose.pt")
T, CLIP_LEN, SR, IMG, NUM_CLASSES = 16, 16, 2, 112, 12
N_KPT, POSE_DIM = 17, 51                 # 17 COCO kpts × [x_norm, y_norm, conf]
EPOCHS, BS, EVAL_BS, LR, WD, NW = 20, 16, 32, 1e-4, 1e-4, 8


def _scan():
    out = []
    for s in scan_etri_root(ETRI):
        core = remap(s.action_idx)
        if core is None:
            continue
        s.action_idx = core
        out.append(s)
    return out


# ----------------------------------------------------------------------- #
# Pose cache: 각 클립의 결정적 중앙 16프레임에서 51-d/frame 추출
# ----------------------------------------------------------------------- #
@torch.no_grad()
def _pose_vec(res):
    """단일 프레임 YOLO result -> 51-d (최고신뢰 person, xyn+conf). 미검출 0."""
    kp = res.keypoints
    if kp is None or kp.xyn is None or kp.xyn.shape[0] == 0:
        return np.zeros(POSE_DIM, np.float32)
    if res.boxes is not None and res.boxes.conf is not None \
            and res.boxes.conf.shape[0] == kp.xyn.shape[0]:
        j = int(res.boxes.conf.argmax())
    else:
        j = 0
    xyn = kp.xyn[j].cpu().numpy()
    conf = (kp.conf[j].cpu().numpy() if kp.conf is not None
            else np.ones(N_KPT, np.float32))
    return np.concatenate([xyn, conf[:, None]], axis=1).reshape(-1).astype(np.float32)


def _read_bgr(path, indices_1based):
    """1-based 프레임 인덱스 리스트 -> BGR 프레임 리스트(요청 순서, 폴백 0)."""
    cap = cv2.VideoCapture(path)
    ordered = sorted(set(indices_1based))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, ordered[0] - 1))
    cur, last, want = ordered[0], ordered[-1], set(ordered)
    got = {}
    while cur <= last:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if cur in want:
            got[cur] = fr
        cur += 1
    cap.release()
    fb = next(iter(got.values()), np.zeros((IMG, IMG, 3), np.uint8))
    return [got.get(i, fb) for i in indices_1based]


@torch.no_grad()
def build_pose_cache(samples, dev):
    """클립당 (T,51) pose 시퀀스 1회 추출·캐시. 프레임은 eval 결정적 중앙클립과 동일."""
    if os.path.exists(CACHE):
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    # eval 과 동일한 프레임 인덱스를 얻기 위해 phase='val' 데이터셋 로직 재사용.
    ref = ETRIClipDataset(samples, clip_length=CLIP_LEN, sampling_rate=SR,
                          img_size=IMG, transform=None, phase="val",
                          centers_per_sample=1, num_classes=NUM_CLASSES)
    pose = YOLO(POSE_W)
    cache = {}
    for k, (si, center) in enumerate(ref.index):       # val: 샘플당 1개
        s = samples[si]
        idx = ref._frame_indices(s, center)            # 16개 1-based
        frames = _read_bgr(s.rgb_path, idx)
        results = pose(frames, verbose=False, device=dev)   # 16-frame 배치
        seq = np.stack([_pose_vec(r) for r in results])     # (16,51)
        cache[s.rgb_path] = torch.from_numpy(seq).half()
        if (k + 1) % 100 == 0:
            print("  pose %d/%d" % (k + 1, len(ref.index)), flush=True)
    torch.save(cache, CACHE)
    return cache


# ----------------------------------------------------------------------- #
# Dataset wrapper: (clip, pose, label)
# ----------------------------------------------------------------------- #
class PoseWrap(Dataset):
    def __init__(self, base: ETRIClipDataset, pose_cache):
        self.base, self.pc = base, pose_cache
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        clip, label = self.base[idx]
        si, _ = self.base.index[idx]
        pose = self.pc.get(self.base.samples[si].rgb_path)
        if pose is None:
            pose = torch.zeros(T, POSE_DIM)
        return clip, pose.float(), label


# ----------------------------------------------------------------------- #
# Two-stream model
# ----------------------------------------------------------------------- #
class R3DPose(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pose_hid=128):
        super().__init__()
        bb = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.feat_dim = bb.fc.in_features              # 512
        bb.fc = nn.Identity()
        self.backbone = bb
        self.pose_gru = nn.GRU(POSE_DIM, pose_hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feat_dim + pose_hid, num_classes))
    def forward(self, x, pose):
        v = self.backbone(x)                           # (B,512)
        _, h = self.pose_gru(pose)                     # h: (1,B,128)
        return self.head(torch.cat([v, h[-1]], dim=1))


def run_epoch(model, loader, crit, opt, dev, train):
    model.train(train)
    tot, n, LG, LB = 0.0, 0, [], []
    for clip, pose, label in loader:
        clip = clip.to(dev, non_blocking=True)
        pose = pose.to(dev, non_blocking=True)
        label = label.to(dev, non_blocking=True)
        x = to_model_input(clip)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x, pose)
            loss = crit(logits, label)
        if train:
            loss.backward(); opt.step()
        bs = label.size(0)
        tot += float(loss.item()) * bs; n += bs
        LG.append(logits.detach().cpu()); LB.append(label.detach().cpu())
    return tot / max(1, n), torch.cat(LG), torch.cat(LB)


def main():
    os.makedirs(OUT, exist_ok=True)
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    sp = group_split(_scan(), val_ratio=0.15, test_ratio=0.15, seed=42)
    assert_no_leakage(sp)
    print("test participants:", sp.test_participants,
          "| train/val/test:", len(sp.train), len(sp.val), len(sp.test), flush=True)
    json.dump({"train": sp.train_participants, "val": sp.val_participants,
               "test": sp.test_participants},
              open(os.path.join(OUT, "split.json"), "w"), indent=2)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_cache = build_pose_cache(sp.train + sp.val + sp.test, dev)
    print("pose cache:", len(pose_cache), "clips", flush=True)

    # train-only mean/std (baseline 과 동일 절차)
    raw = ETRIClipDataset(sp.train, clip_length=CLIP_LEN, sampling_rate=SR,
                          img_size=IMG, transform=None, phase="train",
                          num_classes=NUM_CLASSES, seed=42)
    mean, std = compute_train_stats(raw, max_clips=256)
    print("mean/std:", mean.tolist(), std.tolist(), flush=True)

    tr_tf = build_train_transform(img_size=IMG, clip_length=CLIP_LEN, mean=mean, std=std)
    ev_tf = build_eval_transform(img_size=IMG, clip_length=CLIP_LEN, mean=mean, std=std)
    tr_ds = PoseWrap(ETRIClipDataset(sp.train, CLIP_LEN, SR, IMG, tr_tf, "train",
                                     num_classes=NUM_CLASSES, seed=42), pose_cache)
    va_ds = PoseWrap(ETRIClipDataset(sp.val, CLIP_LEN, SR, IMG, ev_tf, "val",
                                     centers_per_sample=1, num_classes=NUM_CLASSES), pose_cache)
    te_ds = PoseWrap(ETRIClipDataset(sp.test, CLIP_LEN, SR, IMG, ev_tf, "test",
                                     centers_per_sample=1, num_classes=NUM_CLASSES), pose_cache)
    dl = lambda d, b, sh: DataLoader(d, batch_size=b, shuffle=sh, num_workers=NW,
                                     pin_memory=True, persistent_workers=NW > 0,
                                     drop_last=sh)
    tr, va, te = dl(tr_ds, BS, True), dl(va_ds, EVAL_BS, False), dl(te_ds, EVAL_BS, False)

    counts = class_counts(sp.train, num_classes=NUM_CLASSES)
    weights = class_weights_from_counts(counts, scheme="inv_sqrt").to(dev)
    model = R3DPose().to(dev)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_vf, ckpt = -1.0, os.path.join(OUT, "best.pt")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        _, _, _ = run_epoch(model, tr, crit, opt, dev, True)
        vl, vlg, vlb = run_epoch(model, va, crit, None, dev, False)
        vm = evaluate(vlg, vlb, num_classes=NUM_CLASSES)
        sched.step()
        print("epoch %03d/%d | %5.1fs | val loss=%.4f acc=%.4f f1=%.4f aucpr=%.4f"
              % (ep, EPOCHS, time.time() - t0, vl, vm["accuracy"],
                 vm["f1_macro"], vm["auc_pr_macro"]), flush=True)
        if vm["f1_macro"] > best_vf:
            best_vf = vm["f1_macro"]
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "mean": mean.tolist(), "std": std.tolist(),
                        "num_classes": NUM_CLASSES, "val_metrics": vm}, ckpt)

    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    tl, tlg, tlb = run_epoch(model, te, crit, None, dev, False)
    tm = evaluate(tlg, tlb, num_classes=NUM_CLASSES)
    print("[test] loss=%.4f acc=%.4f f1=%.4f aucpr=%.4f"
          % (tl, tm["accuracy"], tm["f1_macro"], tm["auc_pr_macro"]), flush=True)
    json.dump({"loss": tl, **tm}, open(os.path.join(OUT, "test_metrics.json"), "w"), indent=2)

    res = {
        "model": "R3D-18 (Kinetics, fine-tune) ⊕ YOLOv8-pose GRU(51->128), "
                 "two-stream feature fusion, T=16",
        "selection": "best epoch by val macro-F1 (same as baseline R3D-18)",
        "split": "cross-subject group_split seed42 val/test .15 (same as baseline12)",
        "test_participants": sp.test_participants,
        "test_clips": len(sp.test),
        "best_epoch": state["epoch"],
        "test_accuracy": round(tm["accuracy"], 4),
        "test_macro_f1": round(tm["f1_macro"], 4),
        "test_aucpr_macro": round(tm["auc_pr_macro"], 4),
        "pose": "yolov8s-pose, central 16-frame clip (eval-aligned), top-conf person",
        "weights": "runs/r3d_pose/best.pt",
        "compare": {"R3D18_image_only_baseline12": {"acc": 0.6928, "f1": 0.6029,
                                                     "aucpr": 0.7170},
                    "cnn_lstm_image_pose_T5": {"acc": 0.2321, "f1": 0.1993}},
        "note": "Same RGB recipe as baseline12; pose added as 2nd stream to "
                "isolate pose effect. Pose uses eval-aligned central clip.",
    }
    json.dump(res, open(os.path.join(OUT, "r3d_pose_result.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
