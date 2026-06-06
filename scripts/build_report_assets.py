"""Build final-report evidence assets from local experiment artifacts.

Outputs:
  report/assets/*.png   Visuals for the written report.
  report/tables/*.csv   Tables that can be copied into the report.
  report/REPORT_EVIDENCE.md

The script intentionally records N/A for unsupported model families instead of
fabricating CLIP/tracking metrics that this repository never implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("ETRI_CLASS_MAP", str(REPO / "pipeline" / "etri_actions.csv"))

from dataset import ETRIClipDataset, scan_etri_root  # noqa: E402
from model import build_baseline, to_model_input  # noqa: E402
from pipeline.class_map import CORE_NAMES  # noqa: E402
from pipeline.detector import YoloDetector  # noqa: E402
from splits import assert_no_leakage, group_split  # noqa: E402
from transforms import build_eval_transform  # noqa: E402


ASSETS = REPO / "report" / "assets"
TABLES = REPO / "report" / "tables"


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _first_frame(path: str, frame_no: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    if frame_no is None:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_no = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame from: {path}")
    return frame


def _put_label(img: np.ndarray, text: str, y: int = 30) -> None:
    cv2.rectangle(img, (8, y - 24), (min(img.shape[1] - 8, 8 + 13 * len(text)), y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def _class_rows(samples_by_split: dict[str, list]) -> list[dict]:
    rows = []
    total_counts = Counter()
    split_counts = {}
    for split_name, samples in samples_by_split.items():
        cnt = Counter(s.action_idx for s in samples)
        split_counts[split_name] = cnt
        total_counts.update(cnt)

    for idx, name in enumerate(CORE_NAMES):
        train = split_counts["train"].get(idx, 0)
        val = split_counts["val"].get(idx, 0)
        test = split_counts["test"].get(idx, 0)
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "train": train,
                "val": val,
                "test": test,
                "total": train + val + test,
            }
        )
    return rows


def build_dataset_tables_and_plots(samples: list, split) -> None:
    split_samples = {"train": split.train, "val": split.val, "test": split.test}
    class_rows = _class_rows(split_samples)
    _write_csv(
        TABLES / "dataset_class_distribution.csv",
        class_rows,
        ["class_id", "class_name", "train", "val", "test", "total"],
    )

    total = sum(len(v) for v in split_samples.values())
    split_rows = []
    for name, clips in split_samples.items():
        pids = getattr(split, f"{name}_participants")
        split_rows.append(
            {
                "split": name,
                "participants": " ".join(pids),
                "n_participants": len(pids),
                "n_clips": len(clips),
                "clip_ratio": round(len(clips) / total, 4),
            }
        )
    _write_csv(
        TABLES / "split_strategy.csv",
        split_rows,
        ["split", "participants", "n_participants", "n_clips", "clip_ratio"],
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [r["class_name"] for r in class_rows]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    colors = {"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"}
    for split_name in ["train", "val", "test"]:
        vals = np.array([r[split_name] for r in class_rows])
        ax.bar(x, vals, bottom=bottom, label=split_name, color=colors[split_name])
        bottom += vals
    ax.set_title("ETRI 12-class clip distribution by split")
    ax.set_ylabel("clips")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "etri_class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([r["split"] for r in split_rows], [r["n_clips"] for r in split_rows], color=["#3b82f6", "#f59e0b", "#10b981"])
    ax.set_title("Leakage-safe participant split")
    ax.set_ylabel("clips")
    for i, r in enumerate(split_rows):
        ax.text(i, r["n_clips"] + 5, f'{r["n_clips"]}\n{r["clip_ratio"]:.1%}', ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(ASSETS / "split_clip_counts.png", dpi=180)
    plt.close(fig)


def build_performance_tables() -> None:
    base = REPO / "runs" / "baseline12"
    action = _read_json(base / "test_metrics.json")
    lstm = _read_json(base / "lstm_result.json")
    cnn_lstm = _read_json(base / "cnn_lstm_result.json")
    cnn_yolo_lstm = _read_json(base / "cnn_yolo_lstm_result.json")
    cnn_lstm_pose = _read_json(base / "cnn_lstm_pose_result.json")
    r3d_pose = _read_json(REPO / "runs" / "r3d_pose" / "r3d_pose_result.json")
    person_map = _read_json(base / "person_map.json")
    latency = _read_json(base / "detector_latency.json")
    fall = _read_json(REPO / "runs" / "urfd_fall" / "cnn_lstm_result.json")
    kfold = _read_json(REPO / "runs" / "urfd_fall" / "kfold" / "kfold_results.json")

    rows = [
        {
            "track": "object_detection",
            "model": "YOLOv8s COCO detector",
            "input": "RGB frame",
            "metric_1": "person_mAP50",
            "value_1": person_map.get("person_mAP@0.5", "N/A"),
            "metric_2": "mAP50-95",
            "value_2": "N/A - no object bbox GT / no COCO-style sweep artifact",
            "metric_3": "latency_ms_mean",
            "value_3": latency.get("latency_ms_mean", "N/A"),
            "reference_artifact": "runs/baseline12/person_map.json; runs/baseline12/detector_latency.json",
        },
        {
            "track": "action_recognition_baseline",
            "model": "skeleton-LSTM",
            "input": "Kinect 25-joint CSV",
            "metric_1": "test_accuracy",
            "value_1": lstm.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": lstm.get("test_macro_f1", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": lstm.get("test_clips", 293),
            "reference_artifact": "runs/baseline12/lstm_result.json",
        },
        {
            "track": "action_recognition_baseline",
            "model": "CNN+LSTM",
            "input": "RGB frame features",
            "metric_1": "test_accuracy",
            "value_1": cnn_lstm.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": cnn_lstm.get("test_macro_f1", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": cnn_lstm.get("test_clips", 293),
            "reference_artifact": "runs/baseline12/cnn_lstm_result.json",
        },
        {
            "track": "action_recognition_ablation",
            "model": "CNN+YOLO+LSTM",
            "input": "RGB feature + YOLO feature",
            "metric_1": "test_accuracy",
            "value_1": cnn_yolo_lstm.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": cnn_yolo_lstm.get("test_macro_f1", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": cnn_yolo_lstm.get("test_clips", 293),
            "reference_artifact": "runs/baseline12/cnn_yolo_lstm_result.json",
        },
        {
            "track": "action_recognition_improved",
            "model": "R3D-18",
            "input": "RGB clip",
            "metric_1": "test_accuracy",
            "value_1": action.get("accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": action.get("f1_macro", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": action.get("n", "N/A"),
            "reference_artifact": "runs/baseline12/test_metrics.json",
        },
        {
            "track": "action_recognition_experiment",
            "model": "CNN+LSTM+pose",
            "input": "RGB feature + YOLOv8-pose keypoints",
            "metric_1": "test_accuracy",
            "value_1": cnn_lstm_pose.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": cnn_lstm_pose.get("test_macro_f1", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": cnn_lstm_pose.get("test_clips", 293),
            "reference_artifact": "runs/baseline12/cnn_lstm_pose_result.json",
        },
        {
            "track": "action_recognition_experiment",
            "model": "R3D-18 + pose-GRU",
            "input": "RGB clip + YOLOv8-pose keypoints",
            "metric_1": "test_accuracy",
            "value_1": r3d_pose.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": r3d_pose.get("test_macro_f1", "N/A"),
            "metric_3": "n_test_clips",
            "value_3": r3d_pose.get("test_clips", 293),
            "reference_artifact": "runs/r3d_pose/r3d_pose_result.json",
        },
        {
            "track": "fall_recognition",
            "model": "URFD CNN+LSTM",
            "input": "URFD RGB temporal clip",
            "metric_1": "test_accuracy",
            "value_1": fall.get("test_accuracy", "N/A"),
            "metric_2": "macro_F1",
            "value_2": fall.get("test_macro_f1", "N/A"),
            "metric_3": "kfold_acc_mean",
            "value_3": kfold.get("summary", {}).get("CNN+LSTM", {}).get("test_acc", {}).get("mean", "N/A"),
            "reference_artifact": "runs/urfd_fall/cnn_lstm_result.json; runs/urfd_fall/kfold/kfold_results.json",
        },
    ]
    _write_csv(
        TABLES / "model_performance_summary.csv",
        rows,
        ["track", "model", "input", "metric_1", "value_1", "metric_2", "value_2", "metric_3", "value_3", "reference_artifact"],
    )

    unsupported = [
        {
            "requested_family": "CLIP",
            "requested_metrics": "Zero-shot Accuracy; Image retrieval R@1; Image retrieval R@5",
            "status": "not implemented",
            "report_handling": "Exclude from result comparison; do not fabricate metrics.",
        },
        {
            "requested_family": "tracking",
            "requested_metrics": "MOTA; IDF1; ID Switch; FPS",
            "status": "not implemented",
            "report_handling": "Exclude from result comparison; note as future work if needed.",
        },
    ]
    _write_csv(
        TABLES / "unsupported_metric_scope.csv",
        unsupported,
        ["requested_family", "requested_metrics", "status", "report_handling"],
    )


def build_sample_images(split, max_samples: int = 6) -> None:
    cfg = yaml.safe_load((REPO / "pipeline" / "config.yaml").read_text(encoding="utf-8"))
    detector = YoloDetector(
        str(REPO / cfg["paths"].get("yolo_weights", "yolov8s.pt")),
        person_conf=cfg["detector"].get("person_conf", 0.4),
        object_conf=cfg["detector"].get("object_conf", 0.3),
        object_classes=set(cfg["detector"].get("object_classes", ["bed", "chair", "tv"])),
    )

    selected = []
    seen = set()
    for sample in split.val + split.test:
        if sample.action_idx in seen:
            continue
        seen.add(sample.action_idx)
        selected.append(sample)
        if len(selected) >= max_samples:
            break

    rows = []
    for i, sample in enumerate(selected, start=1):
        frame = _first_frame(sample.rgb_path)
        dets = detector.predict(frame)
        overlay = frame.copy()
        for cls, (x1, y1, x2, y2), conf in dets:
            color = (0, 220, 0) if cls == "person" else (220, 160, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(overlay, f"{cls} {conf:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        label = CORE_NAMES[sample.action_idx]
        _put_label(overlay, f"GT action: {label} | YOLO boxes: prediction overlay", 34)
        out = ASSETS / f"sample_{i:02d}_{label}.png"
        cv2.imwrite(str(out), overlay)
        rows.append(
            {
                "image": str(out.relative_to(REPO)),
                "source_clip": _rel(sample.rgb_path),
                "participant": sample.participant,
                "session": sample.session,
                "gt_action_label": label,
                "bbox_label_status": "YOLO prediction overlay; ETRI object bbox GT not available",
                "mask_status": "not used / no mask GT",
            }
        )
    _write_csv(
        TABLES / "sample_image_manifest.csv",
        rows,
        ["image", "source_clip", "participant", "session", "gt_action_label", "bbox_label_status", "mask_status"],
    )


def collect_failure_cases(split, limit: int = 3) -> list[dict]:
    ckpt = REPO / "runs" / "baseline12" / "best.pt"
    if not ckpt.exists():
        return []
    state = torch.load(str(ckpt), map_location="cpu")
    cfg = state["config"]
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    num_classes = int(state["num_classes"])
    tf = build_eval_transform(
        img_size=int(cfg["img_size"]),
        clip_length=int(cfg["clip_length"]),
        mean=mean,
        std=std,
    )
    ds = ETRIClipDataset(
        split.test,
        clip_length=int(cfg["clip_length"]),
        sampling_rate=int(cfg["sampling_rate"]),
        img_size=int(cfg["img_size"]),
        transform=tf,
        phase="test",
        centers_per_sample=int(cfg.get("centers_per_sample", 1)),
        num_classes=num_classes,
    )
    dl = DataLoader(ds, batch_size=int(cfg.get("eval_batch_size", 16)), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    failures = []
    offset = 0
    with torch.no_grad():
        for clips, labels in dl:
            logits = model(to_model_input(clips.to(device)))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            for j in range(labels.size(0)):
                if int(preds[j]) == int(labels[j]):
                    continue
                item_idx = offset + j
                sample_idx, center = ds.index[item_idx]
                sample = ds.samples[sample_idx]
                frame = _first_frame(sample.rgb_path, center - 1)
                gt = CORE_NAMES[int(labels[j])]
                pred = CORE_NAMES[int(preds[j])]
                p = float(probs[j, int(preds[j])])
                _put_label(frame, f"FAIL GT={gt} | pred={pred} ({p:.2f})", 34)
                out = ASSETS / f"failure_{len(failures)+1:02d}_gt-{gt}_pred-{pred}.png"
                cv2.imwrite(str(out), frame)
                failures.append(
                    {
                        "image": str(out.relative_to(REPO)),
                        "source_clip": _rel(sample.rgb_path),
                        "participant": sample.participant,
                        "gt_action": gt,
                        "pred_action": pred,
                        "pred_confidence": round(p, 4),
                        "analysis_note": _failure_note(gt, pred),
                    }
                )
                if len(failures) >= limit:
                    return failures
            offset += labels.size(0)
    return failures


def _failure_note(gt: str, pred: str) -> str:
    if gt in {"mobility", "posture_transition"} or pred in {"mobility", "posture_transition"}:
        return "Movement/posture classes share short temporal patterns; one center clip can miss the full transition."
    if gt in {"sedentary_screen", "phone", "other_social"} or pred in {"sedentary_screen", "phone", "other_social"}:
        return "Fine hand/object cues are small in 1080p home video; RGB clip-level classifier can confuse seated screen/phone/social actions."
    if gt in {"cooking_kitchen", "housework", "hygiene_grooming"} or pred in {"cooking_kitchen", "housework", "hygiene_grooming"}:
        return "Indoor daily actions overlap in body pose and scene context; object GT is unavailable, so context is weak."
    return "Clip-level label may not be visually distinctive in the sampled center window."


def build_failure_tables(split) -> None:
    cm_path = REPO / "runs" / "baseline12" / "best.test.confusion.npy"
    rows = []
    if cm_path.exists():
        cm = np.load(cm_path)
        pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    pairs.append((int(cm[i, j]), i, j))
        for count, gt, pred in sorted(pairs, reverse=True)[:8]:
            rows.append(
                {
                    "gt_action": CORE_NAMES[gt],
                    "pred_action": CORE_NAMES[pred],
                    "count": count,
                    "analysis_note": _failure_note(CORE_NAMES[gt], CORE_NAMES[pred]),
                }
            )
    _write_csv(
        TABLES / "confusion_top_errors.csv",
        rows,
        ["gt_action", "pred_action", "count", "analysis_note"],
    )

    failure_rows = collect_failure_cases(split)
    _write_csv(
        TABLES / "failure_case_manifest.csv",
        failure_rows,
        ["image", "source_clip", "participant", "gt_action", "pred_action", "pred_confidence", "analysis_note"],
    )


def build_report_markdown(samples: list, split) -> None:
    action = _read_json(REPO / "runs" / "baseline12" / "test_metrics.json")
    person = _read_json(REPO / "runs" / "baseline12" / "person_map.json")
    fall = _read_json(REPO / "runs" / "urfd_fall" / "cnn_lstm_result.json")
    split_total = len(split.train) + len(split.val) + len(split.test)
    md = f"""# Final Report Evidence Pack

이 문서는 보고서 작성자가 그대로 인용하거나 표/그림을 복사할 수 있도록 로컬 산출물을 정리한 것입니다.

## 1. 데이터셋 설계

- ETRI EPreTX / ETRI-Activity3D: RGB 행동 인식 학습 및 평가. 로컬 사용량은 `{len(samples)}` clips, 12개 생활 행동 클래스로 remap.
- URFD: 낙상/ADL 이진 분류 학습 및 평가. 로컬 사용량은 `datasets/fall/urfd` 기준 fall 30, ADL 40.
- COCO/YOLOv8: 사람 및 주요 객체 검출은 사전학습 detector를 사용. ETRI에는 객체 bbox GT가 없어 객체 mAP는 산정하지 않고, JointCSV 기반 person pseudo-GT만 평가.

필수 그림/표:

- 클래스별 분포 그래프: `report/assets/etri_class_distribution.png`
- split별 clip 수 그래프: `report/assets/split_clip_counts.png`
- 클래스별 수량 표: `report/tables/dataset_class_distribution.csv`
- 분할 전략 표: `report/tables/split_strategy.csv`
- 샘플 프레임/YOLO bbox 캡처: `report/assets/sample_*.png`
- 샘플 manifest: `report/tables/sample_image_manifest.csv`

## 2. 데이터 분할과 leakage 방지

- split 함수: `src/splits.py::group_split`
- 전략: participant 단위 group split. 같은 participant가 train/valid/test에 동시에 들어가지 않도록 `assert_no_leakage`로 검증.
- 설정: val_ratio 0.15, test_ratio 0.15, seed 42.
- 실제 clip 수: train `{len(split.train)}`, valid `{len(split.val)}`, test `{len(split.test)}`, total `{split_total}`.
- 실제 participant:
  - train: `{", ".join(split.train_participants)}`
  - val: `{", ".join(split.val_participants)}`
  - test: `{", ".join(split.test_participants)}`

## 3. 정량 성능 비교

- 성능 요약 표: `report/tables/model_performance_summary.csv`
- 행동 인식 기준 모델 R3D-18: accuracy `{action.get("accuracy", "N/A")}`, macro-F1 `{action.get("f1_macro", "N/A")}`.
- 사람 검출: person mAP@0.5 `{person.get("person_mAP@0.5", "N/A")}`, latency mean `{_read_json(REPO / "runs" / "baseline12" / "detector_latency.json").get("latency_ms_mean", "N/A")}` ms.
- URFD 낙상 CNN+LSTM: accuracy `{fall.get("test_accuracy", "N/A")}`, macro-F1 `{fall.get("test_macro_f1", "N/A")}`.
- CLIP/Tracking: 구현 범위가 아니므로 metric을 만들지 않음. 제외 근거 표: `report/tables/unsupported_metric_scope.csv`.

## 4. Failure Case

- confusion 상위 오분류 표: `report/tables/confusion_top_errors.csv`
- 실제 오분류 캡처: `report/assets/failure_*.png`
- failure manifest: `report/tables/failure_case_manifest.csv`

보고서에는 성공 사례와 함께 failure 이미지를 최소 2~3개 포함하고, `analysis_note`의 내용을 기술적 원인으로 풀어 쓰면 됩니다.

## 5. 구현/서비스 관점 노력

- 설정 분리: 학습 설정 `src/config.yaml`, 런타임 설정 `pipeline/config.yaml` 분리.
- 하드코딩 제거: detector threshold, 모델 경로, fall threshold를 `pipeline/config.yaml`에서 읽음.
- 모델 로딩 캐시: Streamlit `@st.cache_resource`로 detector/recognizer 재사용.
- 입력 안정성: 업로드 분석은 mp4 처리 경로 중심으로 구성하고, writer 실패 시 PyAV/libx264 -> cv2 fallback을 둠.
- 검증 방식: unit/smoke marker 분리, `conda run -n actiondetect PYTHONPATH=. python -m pytest -q`로 전체 회귀 확인.

## 6. 참고 자료

- ETRI EPreTX 공식 데이터 페이지: https://epretx.etri.re.kr/dataDetail?id=12
- UR Fall Detection Dataset 공식 페이지: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Streamlit run 공식 문서: https://docs.streamlit.io/develop/api-reference/cli/run
- pytest exit code 공식 문서: https://docs.pytest.org/en/7.2.x/reference/exit-codes.html
"""
    (REPO / "report").mkdir(parents=True, exist_ok=True)
    (REPO / "report" / "REPORT_EVIDENCE.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-failure-inference", action="store_true")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    samples = scan_etri_root(str(REPO / "etri"))
    split = group_split(samples, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert_no_leakage(split)

    build_dataset_tables_and_plots(samples, split)
    build_performance_tables()
    build_sample_images(split)
    if args.skip_failure_inference:
        _write_csv(TABLES / "failure_case_manifest.csv", [], ["image", "source_clip", "participant", "gt_action", "pred_action", "pred_confidence", "analysis_note"])
    else:
        build_failure_tables(split)
    build_report_markdown(samples, split)

    print(f"[report] wrote assets under {ASSETS.relative_to(REPO)}")
    print(f"[report] wrote tables under {TABLES.relative_to(REPO)}")
    print("[report] wrote report/REPORT_EVIDENCE.md")


if __name__ == "__main__":
    main()
