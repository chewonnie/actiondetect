# 코드 작성 계획 — 고령자 일상행동 모니터링 시스템

> 상태: **PENDING APPROVAL** (합의 계획 / Planner→Architect→Critic 2회 반복 후 APPROVE)
> 근거 문서: `고급 컴퓨터 비전 프로젝트 제안서_조수연_신채원.docx`
> 작성 원칙: 한 파일 = 한 책임, 주니어가 그대로 따라 할 수 있는 검증 단계 포함.

---

## 0. 한 줄 요약

제안서의 5단계 아키텍처(입력→추론→후처리→집계→대시보드)를 **그대로** 구현하되,
**행동 라벨의 출처만** 바꾼다.
- **YOLOv8 (COCO 사전학습)**: 매 프레임 실시간 박스 오버레이 + 사람/사물 검출.
- **기존 `src/` R3D-18 분류기**: 55→12 클래스로 재학습하여 **로그/집계에 쓰는 행동 라벨**을 생산.
- 사람↔사물 규칙(`context_rules`)은 **대시보드 보조 태그(선택)** 로만 강등. 라벨 출처가 아니다.
- YOLO 파인튜닝 + CVAT 객체 라벨링은 **기본 경로에서 제외**, 플래그로 잠근 스트레치(P8).

이 결정의 이유는 문서 맨 끝 **ADR** 참조.

---

## 1. 왜 이렇게 하나 (핵심 의사결정)

| 항목 | 이유 |
|---|---|
| R3D-18을 버리지 않는다 | ETRI 데이터는 파일명 `A###`에 **행동 라벨이 이미 들어있다**(`src/dataset.py:44-50`). `src/`는 누수 안전 분할·클래스 가중치·F1 체크포인트가 이미 동작하는 자산이다. 가장 어려운 목표(행동 정확도 ≥40%)를 검증된 분류기에 맡긴다. |
| 규칙 기반은 라벨 출처가 아니다 | COCO에는 "손/입/바닥" 클래스가 없어 `손-입 거리`, `바닥 근접` 규칙은 추가 모델 없이 성립 불가. 규칙은 "폰 근처/컵 근처" 같은 **보조 태그**로만. |
| 객체 라벨링은 스트레치 | ETRI는 객체 bbox 라벨을 제공하지 않는다. 약 40~50명 × 1fps = 수만 프레임 수작업 검수는 학생 팀 예산을 초과. 기본 경로는 COCO 사전학습으로 충분. |

---

## 2. 디렉토리 / 모듈 구조

기존 `src/` 는 **거의 그대로 재사용**한다(딱 한 줄 플래그 게이트 편집만).

```
detect/
├── src/                         # 기존 R3D-18 학습 코드 (재사용)
│   ├── dataset.py               # ★ 유일한 src/ 수정: scan_etri_root 안 1곳(플래그 게이트)
│   ├── config.yaml              # class_map 키 추가, num_classes 12로(플래그 켰을 때만)
│   └── (그 외 train/eval/splits/metrics/transforms/model = 무수정)
├── pipeline/
│   ├── etri_actions.csv         # ★ P0 산출물: ETRI 55클래스 라벨명 (외부 출처에서 확보)
│   ├── class_map.py             # 55(A###) → 12 핵심클래스 remap 함수
│   ├── config.yaml              # 런타임 전용 설정 (학습 설정과 분리)
│   ├── frames.py                # 30fps→1fps 프레임 소스 (파일 / webrtc 2가지)
│   ├── detector.py              # YOLOv8 래퍼 (.pt/ONNX), 프레임당 (cls,bbox,conf)
│   ├── action_model.py          # R3D-18 ckpt 로드 + 네이티브fps 16프레임 버퍼 → 12클래스
│   ├── smoother.py              # deque 슬라이딩 윈도우 다수결 (R3D-18 출력 안정화)
│   ├── context_rules.py         # (선택) 사람↔사물 IoU 보조 태그. 플래그 OFF 테스트 가능
│   ├── activity_logger.py       # logs/YYYY-MM-DD.csv append
│   ├── aggregate.py             # pandas 30분/1시간/1일 리샘플 집계
│   └── alerts.py                # 7일 기준선, -30% 2일 연속 → 알림
├── app/
│   └── dashboard.py             # Streamlit: 실시간/요약카드/타임라인/추이/알림
├── (스트레치, 플래그로 잠금)
│   ├── pipeline/make_yolo_labels.py   # COCO 자동라벨 → CVAT export
│   └── pipeline/train_yolo.py         # YOLOv8 파인튜닝
├── requirements.txt             # ultralytics/onnxruntime/streamlit/streamlit-webrtc/plotly 추가
└── PLAN.md
```

**설정 파일 2개는 네임스페이스가 완전히 분리된다.**
`src/config.yaml` = 학습 전용(기존). `pipeline/config.yaml` = 런타임 전용(임계치/경로/윈도우/플래그). 키를 공유하지 않는다.

---

## 3. 모듈별 상세 (각 모듈 = 한 파일, 검증 포함)

> 표기: **무엇** / **입출력** / **주의** / **검증(주니어가 직접 돌려보는 체크)**

### 3.0 `pipeline/etri_actions.csv` (P0 산출물 — **완료됨**)
- **무엇**: ETRI 55개 행동의 `action_idx(0-based),file_token(A###),name_en,name_ko`.
- **상태**: ✅ **확보 완료.** `pipeline/etri_actions.csv` 생성됨. 이전의 HARD BLOCKER **해소**.
- **출처**: 라벨 목록은 **오직** `YOWOv3.zip:YOWOv3/config/etri_idx2name_{ko,en}.yaml`에서만 추출(원 데이터셋 ETRI EPreTX, `epretx.etri.re.kr/dataDetail?id=12`). **YOWOv3 코드/모델 전체는 사용자 결정에 따라 범위 외 — 라벨만 가져옴.**
- **검증**: 55행(idx 0..54), `A001→0` 매핑이 `src/dataset.py:parse_action_index`와 일치, `A000`은 행동 아님(제외), 중복 없음.

### 3.1 `pipeline/class_map.py`
- **무엇**: `remap(action_idx) -> Optional[int]`. `etri_actions.csv`를 읽어 55→12 매핑. drop이면 `None`.
- **12 핵심 클래스 후보(인간 확인 필요, REQUIRES HUMAN CONFIRMATION)**: 식사 / 음용 / 약 복용 / 휴대폰 사용 / 독서 / TV 시청 / 보행 / 좌식 / 와상·수면 / 기립·착석(자세 전환) / 낙상 / 기타. → **반드시 공식 라벨명과 대조해 팀이 확정**.
- **주의**: 출력은 **빈틈 없는 0..11 연속 정수**여야 한다(이유: `src/metrics.py:54`가 `np.zeros(num_classes)` 밀집 빈을 쓰므로, 비연속이면 클래스 가중치가 0분모/퇴화).
- **검증**:
  1. `A001..A055` 각각이 12개 중 정확히 하나로 매핑되거나 명시적 drop.
  2. `set(모든 remap 출력) == set(range(12))` — **빈 클래스 없음**.
  3. 단, 위 (2)는 P1에서 **실제 추출된 데이터** 기준 클래스별 표본 수 점검과 별개. (3.x P1 참조)

### 3.2 `src/dataset.py` — 유일한 `src/` 수정 (딱 한 곳, **환경변수** 게이트)
- **정련(구현 시 확정)**: PLAN 초안은 `src/config.yaml`의 `class_map` 키로 게이트하려 했으나, config 키를 쓰면 `train.py:155`·`eval.py:33`의 `scan_etri_root` 호출부 2곳을 고쳐 키를 전달해야 한다 → "src/는 dataset.py 1곳만 수정" 제약 위반. 그래서 **환경변수 `ETRI_CLASS_MAP` 게이트**로 확정. config 키보다 surgical 제약을 더 잘 만족.
- **무엇**: `scan_etri_root`의 `parse_action_index` + None 가드(`dataset.py:121-123`) **바로 다음**에 remap 블록 + 모듈 상단에 lazy 로더 `_load_class_map()` 1개:
  ```python
  _cm = _load_class_map()      # ETRI_CLASS_MAP env var → {action_idx:core_idx}
  if _cm:
      if action_idx not in _cm:
          continue
      action_idx = _cm[action_idx]
  ```
- **주의**: 환경변수 `ETRI_CLASS_MAP=<etri_actions.csv 경로>` 미설정이면 **기존 55클래스 베이스라인을 비트 단위로 재현**(remap 미적용). 12클래스 학습 시에만 env 설정 + `num_classes:12` config. `train.py`/`eval.py`/`splits.py`/`metrics.py` **무수정**.
- **왜 이 한 곳이면 충분한가**: `dataset.py:333`은 `int(sample.action_idx)` 단순 패스스루, `train.py:180-181`·`eval.py`·`splits.py`·`metrics.py`는 모두 remap된 0..11만 본다. `A000`은 `parse_action_index`의 `n>=1` 가드로 이미 제외.
- **검증(무수정 증명)**: env 미설정 시 모든 `sample.action_idx == parse_action_index(파일명)` (remap 미적용 = 베이스라인 동일), env 설정 시 모든 라벨 ∈ {0..11}이고 클립 수 동일(확정 매핑은 drop 없음). → 실측: P01-P05 490클립, 미설정 55 distinct / 설정 dense {0..11}, 클립 수 보존 확인됨.

### 3.3 12클래스 재학습 (새 코드 없음 — 기존 `src/train.py`/`eval.py` 사용)
- **무엇**: 환경변수 `ETRI_CLASS_MAP=pipeline/etri_actions.csv` + `num_classes: 12` config(`configs/p1_12class.yaml`) → `python src/train.py --config configs/p1_12class.yaml` (train.py가 best.pt 저장 후 자동 final test) → cross-subject test 정확도/macro-F1 기록.
- **주의**: `src/eval.py`는 `num_classes`를 **체크포인트에서** 읽는다(`eval.py:28-29`). P1 베이스라인 증명 단계에서 `--config`로 불일치한 `num_classes`를 덮어쓰지 말 것(체크포인트 값이 조용히 쓰임).
- **검증**: P1 게이트(아래) 참조.

### 3.4 `pipeline/frames.py`
- **무엇**: 30fps→1fps 프레임 소스. 백엔드 2개: (a) 업로드 파일 경로, (b) `streamlit-webrtc` 실시간.
- **주의**: 이 1fps 스트림은 **YOLO 오버레이/로그 전용**. R3D-18 입력과 무관(3.6 참조).
- **검증**: 30초 클립 → 약 30프레임, 각 640×640.

### 3.5 `pipeline/detector.py`
- **무엇**: Ultralytics YOLOv8 래퍼. `.pt` 또는 ONNX 로드. 프레임당 `list[(cls,bbox,conf)]` — 사람 + COCO 객체(cup,bottle,book,cell phone,tv,bed,chair…).
- **검증**: COCO 사전학습 스모크 — 샘플 프레임에서 사람 1명 검출.

### 3.6 `pipeline/action_model.py` — 시간 계약 핵심
- **무엇**: 학습된 R3D-18 ckpt 로드. **별도의 네이티브-fps 버퍼**를 두고 `clip_length(16) × sampling_rate(2) = 32` 네이티브 프레임 범위(≈1.07s @30fps)를 학습과 동일하게 샘플 → 12클래스 logits. N프레임마다 추론.
- **주의**: 1fps YOLO 스트림과 **분리**. R3D-18 버퍼는 학습 분포와 동일해야 P1의 `eval.py` 숫자가 배포 성능을 대표한다.
- **dual-rate join 계약**: 1fps 로그 각 행은 **가장 최근 완료된** R3D-18 라벨을 사용(hold-last, 지연 ≤ 1윈도우). 프레임마다 재추론 금지. `aggregate.py`/`alerts.py`는 라벨이 프레임 타임스탬프보다 ≤1윈도우 지연될 수 있음을 허용.
- **검증**: 버퍼 시간 범위 == `src/config.yaml` `clip_length×sampling_rate` 단언. 알려진 test 클립에서 기대 coarse 클래스 출력.

### 3.7 `pipeline/smoother.py`
- **무엇**: deque 슬라이딩 윈도우 다수결로 R3D-18 단발 플립 제거.
- **검증**: 합성 노이즈 시퀀스 입력 → 안정화 출력.

### 3.8 `pipeline/urfd_fall_model.py` — URFD R3D-18 단독 낙상 (옵션)
- **무엇**: URFD(Kwolek 2014, CC BY-NC-SA 4.0)로 학습한 2-class R3D-18로 `P(fall)` 산출. dashboard 1b 패널은 `fall.urfd_prob_thr` 임계값을 넘는 URFD 이벤트만 낙상 신호로 사용한다.
- **CNN+LSTM 변형**: `experiments/urfd_fall/train_urfd_cnnlstm.py` — MobileNetV3-small(동결) 576-d → 2층 LSTM(hidden=128) → 2 class. 가중치는 `runs/urfd_fall/cnn_lstm.pt`. R3D-18과 동일 stratified split.
- **⚠ 도메인 시프트**: URFD = staged 실험실(사무실/병실) 320×240 **좌=Depth | 우=RGB** 결합 영상. 실거주(ETRI/홈) 분포와 다르므로 zero-shot 일반화 보장 없음. 학습 셋 acc=1.000은 **도메인 내부 성능**일 뿐, 일반화 검증은 별도 in-the-wild 셋(OmniFall 등) 필요. 운영 시 임계치 조정과 별도 검증이 필요.
- **검증**: `tests/test_urfd_fall.py` — 임계치/쿨다운/엣지 5 케이스(ActionModel 의존성 없이 unit).

### 3.9 `pipeline/context_rules.py` (선택)
- **무엇**: 사람↔사물 IoU/거리 → "폰 근처/컵 근처" 보조 태그(대시보드 표시용).
- **검증**: 플래그 OFF 시 파이프라인 결과 불변.

### 3.10 `pipeline/activity_logger.py`
- **무엇**: `logs/YYYY-MM-DD.csv`에 append. 스키마: `timestamp,class,confidence,bbox,subject_id`.
- **주의**: 단일 가정 배포라 RGB만으로 `subject_id` 재식별 불가 → `pipeline/config.yaml`의 **상수**(예: `P_home`).
- **검증**: 3 이벤트 → 헤더 + 3행, 같은 날 재오픈 시 truncate 아닌 append.

### 3.11 `pipeline/aggregate.py`
- **무엇**: pandas로 로그 로드 → 30분/1시간/1일 리샘플. 클래스별 누적시간/횟수/주요 발생 시간대.
- **검증**: 합성 1일 로그 → 알려진 합계 일치.

### 3.12 `pipeline/alerts.py`
- **무엇**: 7일 롤링 기준선, 클래스별 -30% 초과 + **2일 연속** 시 알림.
- **주의**: 콜드 스타트 — 이력 7일 미만이면 "기준선 부족, 알림 없음" 상태.
- **검증**: 8일 합성 시계열 → 지속 감소일에만 발화, 단일일 dip엔 미발화. 워밍업(1~7일) 동작도 검증.

### 3.13 `app/dashboard.py`
- **무엇**: Streamlit — 실시간 webrtc + bbox 오버레이 / 일별 요약 카드(식사·보행·좌식·수면) / Plotly 24h 타임라인 / 7일 추이 + 임계선 / 알림 패널.
- **검증**: 기록된 로그 디렉토리로 실행 시 5개 패널 모두 렌더.

### 3.14 (스트레치, 플래그로 잠금) `make_yolo_labels.py` + `train_yolo.py`
- **무엇**: COCO 자동라벨 → CVAT export → YOLOv8s 파인튜닝 → 객체 mAP@0.5.
- **주의**: **기본 경로 아님.** 객체 mAP@0.5≥0.5 목표는 이 경로로만 도달. 기본 경로는 COCO 사전학습 객체 mAP를 **정직하게 보고**하고 ≥0.5는 스트레치로 명시.

---

## 4. 단계별 빌드 순서 (각 단계 검증 실패 시 STOP)

| 단계 | 내용 | 통과 기준(검증) |
|---|---|---|
| **P-1** 환경/데이터 | README conda env 생성; `ultralytics/onnxruntime/streamlit/streamlit-webrtc/plotly` pip 추가 + `requirements.txt` 갱신; RGB/JointCSV zip을 `./etri/RGB/P##/<session>/A###_*.mp4`로 추출; 기존 `src/train.py` 1 epoch, `participants:['P01','P02','P03']`(splits.py가 train/val/test 각 ≥1 → 최소 3명) 스모크 | `eval.py`가 55클래스 숫자 출력(새 코드 전에 베이스라인 경로 증명) |
| **P0** 라벨/매핑 | ✅ `etri_actions.csv` **확보 완료**(블로커 해소); `class_map.py` + 단위테스트; `pipeline/config.yaml`; **무수정 증명**(ETRI_CLASS_MAP 미설정 = 베이스라인 동일) | 3.1·3.2 검증 통과 |
| **P1** 12클래스 게이트 | `class_map` 설정 + `num_classes:12` 재학습; **추출된 실데이터** 기준 클래스별 표본 수 보고(빈약 클래스 = ≥40% 위험 입력) | **PASS = test 정확도 ≥0.40 AND macro-F1 기록.** <0.40 → `class_weight:effective`로 1회 재시도. 그래도 <0.40 → **STOP + confusion matrix 첨부 에스컬레이션** (자동 모델 교체·암묵 진행 금지) |
| **P2** 검출기 | `detector.py` COCO 스모크; test split에서 사람 mAP@0.5 측정 | 사람 mAP@0.5 ≥ 0.7 보고 |
| **P3** 인식 코어 | `frames.py` + `action_model.py`(시간 계약) + `smoother.py`; **E2E fps 벤치마크** | 버퍼 범위 단언 통과; fps는 **목표 명시 또는 "report-only"로 기록**. R3D-18이 캐던스 못 맞추면 STOP+에스컬레이션(임의 모델 교체 금지) |
| **P4** 낙상 | URFD 모델 어댑터 | 임계치/쿨다운 단위테스트 통과 |
| **P5** 로그 | `activity_logger.py` | append/스키마 검증 |
| **P6** 집계/알림 | `aggregate.py` + `alerts.py` | 합성 로그/시계열 검증 |
| **P7** 대시보드 | `app/dashboard.py` | 5개 패널 렌더 |
| **P8** (스트레치) | `make_yolo_labels.py` + `train_yolo.py` | 플래그 ON 시에만; 객체 mAP@0.5 보고(≥0.5는 스트레치) |

---

## 5. 리스크와 완화 (정직하게 표면화)

| 리스크 | 탐지 시점 | 완화 |
|---|---|---|
| R3D-18이 12클래스·~40명에서 <40% | **P1(최조기)** | `class_weight:effective` 1회 재시도 → 안 되면 STOP+에스컬레이션. 자동 모델 교체·오버샘플링은 범위 외(샘플러 훅 없음) |
| 두 모델이 단일 GPU 실시간 초과 | P3 벤치마크 | STOP+에스컬레이션(임의 경량 모델 대체 금지 — 사용자 결정 사항) |
| 객체 mAP@0.5≥0.5 미달 | P2/P8 | 기본 경로는 COCO 숫자 정직 보고, ≥0.5는 스트레치로 명시(숨기지 않음) |
| ~~ETRI 55 라벨명 부재~~ | ~~P0~~ | ✅ **해소** — `YOWOv3.zip`의 `etri_idx2name_{ko,en}.yaml`에서 라벨만 추출해 `pipeline/etri_actions.csv` 확보. YOWOv3 코드/모델은 범위 외 |
| 로컬 ~40-50명만 보유 | P1 표본점검 | split 비율 적응(val≈0.1/test≈0.1), 사용 피험자 수 문서화 |
| 빈약 핵심 클래스 | P1 (실데이터 기준) | 표본 수를 ≥40% 게이트 판단의 명시 입력으로 사용 |
| 프라이버시(고령자 가정 영상) | 설계 | 기본 정책: 메타데이터만 저장, 원본 영상 비보관, `pipeline/config.yaml`+README 명시 |
| URFD 도메인 시프트(staged 실험실 → 실거주) | 배포 시 | 1차: `fall.urfd_prob_thr` 조정; 2차: OmniFall/in-the-wild 셋으로 추가 검증; ckpt 미로드 시 낙상 검출 비활성 |

---

## 6. ADR (Architecture Decision Record)

- **Decision**: 제안서의 5단계 아키텍처를 유지하되 stage-3 행동 라벨 출처를 "규칙 기반"에서 "기존 R3D-18(55→12 재학습)"으로 교체하는 **하이브리드**. YOLOv8은 실시간 오버레이/사람·객체 검출을 담당하고, 낙상은 URFD 학습 모델 신호만 사용한다. 규칙은 보조 태그로 강등. 객체 라벨링·YOLO 파인튜닝은 플래그 잠금 스트레치.
- **Decision Drivers**: (1) ETRI는 행동 라벨이 파일명에 내장 — 지도 신호가 공짜. (2) `src/`는 누수 안전·검증된 자산. (3) 가장 어려운 목표(≥40%)는 가장 강한 증거를 가진 경로에 둔다. (4) 객체 bbox 라벨 부재 = 수만 프레임 수작업 불가. (5) CLAUDE.md §2 단순성·§3 외과적 변경·§4 검증 가능 목표.
- **Alternatives considered**:
  - A. 규칙 기반(COCO 위) — COCO에 손/입/바닥 없음, ≥40% 검증 P9까지 불가, 동작 자산 폐기. **기각**.
  - C. 순수 R3D-18(YOLO 제외) — 가장 단순하나 제안서의 실시간 오버레이와 "객체 검출" 트랙 부합도 최저. **기각**.
  - B. 하이브리드 — **채택**.
- **Why chosen**: 세 수치 목표 모두에 가장 강한 증거를 배치, 제안서 아키텍처 형태 보존, src/ 변경은 단 1곳 플래그 게이트(베이스라인 비트 단위 재현), 가장 어려운 메트릭을 P1에서 조기 검증.
- **Consequences**: 런타임 2모델(지연/메모리 — P3에서 벤치). 데이터 결(클립 라벨) vs 트랙 결(프레임 실시간 검출)의 본질적 긴장은 dual-rate 분리 + hold-last join으로 흡수하되 여전히 측정 대상. ~~ETRI 라벨 목록은 외부 의존(P0 하드 블로커)~~ → **해소**: `YOWOv3.zip`에서 라벨만 추출 확보. **YOWOv3 코드/모델은 사용자 결정에 따라 범위 외**(아키텍처 변경 없음, 라벨만 차용).
- **Follow-ups**: P0 ETRI 라벨 목록 확보 및 12분류 인간 확정; P1·P3 벤치마크 수치 기록; 스트레치 P8 진행 여부는 P1~P7 결과 보고 결정.

---

## 7. 합의 이력

- v1 (Planner): YOLO+규칙 라벨. → Architect: 동작 R3D-18 폐기·≥40% 미검증(HIGH) 지적, 하이브리드 권고.
- v2 (하이브리드): → Critic **ITERATE** — CRITICAL 1(가짜 "boundary, src/ UNCHANGED") + MAJOR 4(env/data, 시간 불일치, ETRI 라벨 부재, 모호한 fallback).
- v3: 5개 수정 + Architect 확인(밀집 codomain 단언, hold-last join 계약 추가) → Critic **APPROVE** (예약 2건 반영, 블로커 없음).

---

## 8. 파일별 검증 방식 (주니어가 그대로 실행)

### 8.0 공통 규약
- 새 코드의 테스트는 `tests/test_<모듈>.py`에 두고 **pytest**로 실행한다.
- 검증은 3종류로 분류한다:
  - **U (순수 단위)**: 데이터·GPU·네트워크 불필요. 합성 입력만 사용. 언제든 즉시 실행. `pytest -m unit`
  - **S (스모크)**: 의존성/데이터/가중치 필요. 해당 Phase에서만 실행. `pytest -m smoke`
  - **G (메트릭 게이트)**: 전체 실행 결과 수치로 통과/실패 판정 (P1/P2/P3).
- 각 파일은 "통과 기준 = 단언문(assert)"이 명확해야 한다. 애매하면 그 파일은 미완성으로 간주.

### 8.1 순수 단위(U) — 데이터/GPU 없이 합성 입력으로 검증

| 파일 | 테스트 픽스처(합성) | 단언(통과 기준) | 실행 |
|---|---|---|---|
| `pipeline/class_map.py` | `etri_actions.csv` 로드 | (1) `remap`이 0..54 입력 각각에 12 중 하나 또는 `None` 반환; (2) `set(None 아닌 출력)==set(range(12))` (빈틈 없는 밀집); (3) `remap(-1) is None` (A000 방어) | `pytest tests/test_class_map.py` |
| `pipeline/smoother.py` | 라벨 시퀀스 `[3,3,7,3,3,3]` | 윈도우 다수결 출력이 단발 `7`을 제거하고 `3` 유지; 윈도우 미충족 구간은 정의된 초기값 반환 | `pytest tests/test_smoother.py` |
| `pipeline/activity_logger.py` | tmp dir + 3개 이벤트 | 헤더 1줄 + 3행; 같은 날 재호출 시 **append**(truncate 아님); 스키마 `timestamp,class,confidence,bbox,subject_id` 정확 | `pytest tests/test_logger.py` |
| `pipeline/aggregate.py` | 합성 1일 CSV(알려진 횟수/시간) | 30분/1시간/1일 리샘플 합계가 **손으로 계산한 값과 정확히 일치**; 클래스별 누적시간·횟수·피크 시간대 일치 | `pytest tests/test_aggregate.py` |
| `pipeline/alerts.py` | 합성 8일 시계열(7일 정상→2일 -35%) | (1) 2일 연속 감소일에만 발화; (2) 단일일 dip 미발화; (3) 이력<7일이면 "기준선 부족, 알림 없음" 상태 반환 | `pytest tests/test_alerts.py` |
| `pipeline/context_rules.py` | 사람·컵 박스 IoU 케이스 + 플래그 OFF | 플래그 ON: 기대 태그(`near cup`); 플래그 OFF: 빈 태그 + **파이프라인 결과 불변** | `pytest tests/test_context_rules.py` |

> U 테스트는 전부 합성 입력이라 P-1(환경) 전에도 돌릴 수 있고 1초 내 끝난다. 이게 주니어의 1차 안전망.

### 8.2 스모크(S) — 의존성/데이터 필요, 해당 Phase에서

| 파일 | 선행조건 | 단언(통과 기준) | 실행 |
|---|---|---|---|
| `src/dataset.py` (env-게이트 편집) | `./etri` 추출됨 (P-1) | env `ETRI_CLASS_MAP` 미설정 시 `action_idx==parse_action_index(파일명)`(베이스라인 비트 보존), 설정 시 모든 라벨 ∈{0..11} & 클립 수 동일 | `pytest -m smoke tests/test_dataset_remap.py` (P0) |
| `pipeline/frames.py` | 샘플 mp4 1개 | 30초 클립 → **약 30프레임**(1fps), 각 프레임 `640×640×3`; 파일·webrtc 백엔드 동일 인터페이스 | `pytest -m smoke tests/test_frames.py` (P3) |
| `pipeline/detector.py` | `ultralytics` + YOLOv8 가중치 | COCO 사전학습으로 사람 포함 샘플 프레임에서 **`person` 1개 이상 검출**, 반환 형식 `list[(cls,bbox,conf)]` | `pytest -m smoke tests/test_detector.py` (P2) |
| `pipeline/action_model.py` | P1 학습된 R3D-18 ckpt | (1) 버퍼 시간범위 `== clip_length×sampling_rate`(=32) **단언**; (2) 알려진 test 클립 → 기대 coarse 클래스; (3) hold-last: 새 추론 전까지 직전 라벨 유지 | `pytest -m smoke tests/test_action_model.py` (P3) |
| `app/dashboard.py` | 기록된 로그 디렉토리 | `streamlit run` 후 5개 패널(실시간/요약카드/타임라인/추이/알림) 모두 렌더, 예외 없음 (수동 체크리스트 + 스모크 기동) | `streamlit run app/dashboard.py` (P7) |
| (스트레치) `make_yolo_labels.py` / `train_yolo.py` | 플래그 ON | 자동라벨 산출물이 CVAT 포맷 유효; 파인튜닝 후 객체 mAP@0.5 수치 보고 | P8에서만 |

### 8.3 메트릭 게이트(G) — 새 코드 아님, 기존 `src/` 스크립트 사용

| 게이트 | 명령 | 통과 기준 |
|---|---|---|
| **P-1 베이스라인 증명** | `python src/train.py --config configs/smoke55.yaml` (`participants:['P01','P02','P03']`, `epochs:1`, ETRI_CLASS_MAP 미설정) → `python src/eval.py --checkpoint runs/smoke55/best.pt --split test` | `eval.py`가 55클래스 정확도 숫자 출력(경로 증명). 주의: `eval.py`는 `num_classes`를 ckpt에서 읽음 — 불일치 `--config` 덮어쓰기 금지. ✅ 실측 완료(acc 0.0098, n=102, 55클래스) |
| **P1 행동 정확도(핵심)** | `class_map` 설정+`num_classes:12` → `train.py` → `eval.py` | **PASS = test 정확도 ≥0.40 AND macro-F1 기록.** <0.40 → `class_weight:effective` 1회 재시도 → 그래도 <0.40 → **STOP + confusion matrix 첨부 에스컬레이션** (자동 진행 금지). + 실추출 데이터 기준 클래스별 표본 수 보고 |
| **P2 사람 검출** | test split에서 `detector.py` mAP 측정 | 사람 mAP@0.5 ≥ 0.7 보고 |
| **P3 실시간성** | 양 모델 동시 E2E 벤치 | fps 수치 기록. 목표 미달 시 STOP + 에스컬레이션(임의 모델 교체 금지). 목표 미정이면 "report-only"로 명시 |

### 8.4 검증의 큰 그림
- **순서**: U(8.1) 먼저 작성·통과 → 그 위에 S(8.2) → 마지막 G(8.3). 한 단계라도 실패하면 다음으로 못 넘어간다(각 Phase STOP 규칙과 동일).
- **주니어 체크포인트**: "이 파일이 끝났다"의 정의 = 해당 행의 `pytest ...` 가 초록불. 초록불 아니면 미완성.
- **회귀 보호**: `src/dataset.py` 1줄 편집의 무수정 증명(8.2 첫 행)이 기존 55클래스 베이스라인을 깨지 않았음을 보장한다.

---

> 다음 단계: 본 계획 승인 시 실행은 `/oh-my-claudecode:team`(병렬, 권장) 또는 `/oh-my-claudecode:ralph`(순차)로. **현재는 PENDING APPROVAL 상태이며 어떤 코드도 작성·실행하지 않았다.**
