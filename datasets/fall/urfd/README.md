# URFD — UR Fall Detection Dataset (cam0 RGB subset)

URFD를 ETRI(`detect/etri/`) 보완용 **낙상 라벨 데이터**로 받아둔 사본입니다.
ETRI-Activity3D에는 fall 클래스가 없어서 별도로 확보했습니다.

## Source / License
- 페이지: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- 라이선스: **CC BY-NC-SA 4.0** (비상업·학술용)
- 필수 인용:
  > Bogdan Kwolek, Michal Kepski. *Human fall detection on embedded platform
  > using depth maps and wireless accelerometer.* Computer Methods and Programs
  > in Biomedicine, Vol. 117, Issue 3, Dec 2014, pp. 489–501.

## What is downloaded
```
urfd/
├── fall/   fall-01-cam0.mp4 .. fall-30-cam0.mp4   (30 clips, 23 MB)
├── adl/    adl-01-cam0.mp4  .. adl-40-cam0.mp4    (40 clips, 71 MB)
└── meta/
    ├── urfall-cam0-falls.csv   per-frame depth-feature + label (fall set)
    └── urfall-cam0-adls.csv    per-frame depth-feature + label (ADL set)
```
- 모든 mp4 = `cam0` (정면) RGB. cam1(측면)·Depth zip·Accelerometer는 제외.
- 디스크: 약 94 MB.

## Per-frame label convention (meta CSV 3rd column)
URFD 공식 정의:
- `-1` = 정상(서있음/일상)
- ` 0` = 전이(낙상 중)
- ` 1` = 누워 있음(낙상 후)

CSV 컬럼: `seq_id, frame_idx, label, h_ratio, w_h_ratio, depth_max, ...` (총 11열).

## Clip-level binary mapping (학습용 권장)
- `fall/fall-##-cam0.mp4` → `label = 1 (FALL)`
- `adl/adl-##-cam0.mp4`  → `label = 0 (ADL)`
- 시간정밀 평가가 필요하면 `meta/urfall-cam0-falls.csv`의 frame-level 라벨 사용.

## ETRI 코드와 함께 쓰려면
`src/dataset.py:scan_etri_root`는 파일명 `A###_*.mp4`의 `A###`로 클래스를
파싱합니다. URFD는 라벨 체계가 달라 그대로는 합쳐지지 않으며, 통합 학습에
쓰려면 별도 어댑터(`scan_urfd_root`)나 파일명 리네이밍(`A056_*` 등) 추가가
필요합니다. **이 디렉터리에는 원본 파일명을 보존해 두었습니다.**

## ⚠ Domain shift caveat
URFD는 staged 실험실 환경(사무실/병실, 5명 연기자가 의도적 낙상)으로 구성되어
학습 분포와 실거주(ETRI/홈) 환경 분포가 크게 다릅니다.

- 같은 split 내 R3D-18 / CNN+LSTM 정확도 1.000은 **도메인 내부 성능**일 뿐
- 실거주 영상에 zero-shot 적용 시 정확도 급락 가능성 매우 높음
- 일반화 검증은 별도 in-the-wild 셋(OmniFall 등) 필요
- 운영 환경에서는 임계치 조정(`P(fall) ≥ 0.7~0.9`)과 별도 in-the-wild 검증 권장
  (`pipeline/config.yaml` `fall.urfd_prob_thr`; `paths.urfd_fall_ckpt` 미로드 시 낙상 검출 비활성)

## 재현 (다운로드 명령)
```bash
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"
seq -f "fall-%02g-cam0.mp4" 1 30 \
  | xargs -n1 -P8 -I{} curl -sSL --retry 3 -o "fall/{}" "$BASE/{}"
seq -f "adl-%02g-cam0.mp4"  1 40 \
  | xargs -n1 -P8 -I{} curl -sSL --retry 3 -o "adl/{}"  "$BASE/{}"
curl -sSL -o meta/urfall-cam0-falls.csv "$BASE/urfall-cam0-falls.csv"
curl -sSL -o meta/urfall-cam0-adls.csv  "$BASE/urfall-cam0-adls.csv"
```
