# Final Report Evidence Pack

이 문서는 보고서 작성자가 그대로 인용하거나 표/그림을 복사할 수 있도록 로컬 산출물을 정리한 것입니다.

## 1. 데이터셋 설계

- ETRI EPreTX / ETRI-Activity3D: RGB 행동 인식 학습 및 평가. 로컬 사용량은 `2028` clips, 12개 생활 행동 클래스로 remap.
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
- 실제 clip 수: train `1443`, valid `292`, test `293`, total `2028`.
- 실제 participant:
  - train: `P01, P04, P06, P07, P08, P10, P11, P12, P13, P15, P16, P17, P19, P20`
  - val: `P03, P05, P18`
  - test: `P02, P09, P14`

## 3. 정량 성능 비교

- 성능 요약 표: `report/tables/model_performance_summary.csv`
- 행동 인식 기준 모델 R3D-18: accuracy `0.6928327645051194`, macro-F1 `0.6028563815012266`.
- 사람 검출: person mAP@0.5 `0.799`, latency mean `10.319368040654808` ms.
- URFD 낙상 CNN+LSTM: accuracy `0.9`, macro-F1 `0.899`.
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
