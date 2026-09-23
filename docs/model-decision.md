# S0 모델·런타임 결정

작성일: 2026-09-22, S5 최종 측정 반영: 2026-09-23. 이 문서는 실제 ARM64
실행과 고정 holdout을 바탕으로 한 v0.1 기본 프로필 결정이다. 모델 카드의
benchmark나 파일 크기만으로 내린 결정이 아니다. deep 품질 목표는 통과했지만
ARM 지연 목표는 미달했으므로 “제품 목표 전부 달성”으로 해석하지 않는다.

## 선택

기본 제품 프로필은 **`compact-multilingual`**로 한다.

- 모델: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- revision: `1427fd652930e4ba29e8149678df786c240d8825`
- 제품 ONNX: `onnx/model_qint8_arm64.onnx`, 118,620,017 bytes,
  SHA-256 `1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4`
- tokenizer: `tokenizer.json`, 17,082,660 bytes,
  SHA-256 `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626`
- tokenizer/graph 규약: XLM-R tokenizer, `<pad>` ID 1, 최대 pair 256,
  최대 query 64 (special token 포함), `input_ids`와 `attention_mask`,
  `logits` 하나. score는 보정되지 않은 `relevance_logit`이다.
- 초기 실행 설정: batch 1, intra-op 4, graph optimization Level 3.
  ARM 양자화 graph는 같은 행도 서로 다른 batch 조성에서 raw logit이
  달라지는 것을 관찰했으므로, 결과 재현성을 위해 기본 batch를 1로
  고정한다. `--threads`는 나중에 명시적으로 재정의할 수 있다.

`tiny-en`은 영어 기준선/비교용으로만 유지한다. 한국어 query를 64-token
계약에서 안정적으로 실행할 수 없었고, 영어 전용 모델을 기본값으로 바꿔
한국어 지원을 완료한 것으로 처리하지 않는다. GTE는 개발 증거 풀에서 더
좋은 한국어 결과를 냈지만 이 4-core ARM에서 처리 시간이 훨씬 길어 제품
기본값으로 채택하지 않았다.

## 검증된 실행 조합

제품 bundle의 런타임 기준은 공식 ONNX Runtime Linux ARM64 1.20.0이다.

| 항목 | 값 |
| --- | --- |
| runtime archive | `onnxruntime-linux-aarch64-1.20.0.tgz`, 5,368,581 bytes, SHA-256 `b4d7c6e2c45f8edabe5d28e9bc59ec8d5a4a4af36660cda16e94b2ad85f2a52a` |
| runtime library | `lib/libonnxruntime.so.1.20.0`, SHA-256 `fd7ec997121748668da52a0ec76d1aa6a9096dbd598cad743d56a4fa37ee9f97` |
| build wrapper | `ort = ort-sys = 2.0.0-rc.9`, dynamic loading only |
| executable lookup | absolute `SUPERGREP_ORT_LIB`, 아니면 executable 옆 `runtime/libonnxruntime.so.1.20.0`; current directory 탐색 없음 |

Debian 시스템 `libonnxruntime.so.1.21.0`은 모델을 실행할 수는 있었지만
대량의 중복 ONNX schema 경고를 출력했다. 배포 런타임으로 채택하지 않았다.

토크나이저 JSON의 producer-side 512 padding/truncation은 로드 시 해제한다.
그렇지 않으면 GTE tokenizer가 짧은 한국어 query도 512 tokens로 보이게
되어 v0.1의 64-token 입력 검사를 잘못 통과/거절할 수 있었다. 제품은
자체의 64/256 계약으로 길이를 검사하고, 넘는 입력은 절단하지 않는다.

## 실제 S0 증거

모든 수치는 이 저장소의 실제 ARM64 (Cortex-A76 4 CPU, RAM 7.9 GiB)에서
로컬 파일과 위 runtime으로 얻었다. 측정 중 다른 사용 부하가 있었으며,
특정 순간의 memory availability/swap 상태를 하드웨어 일반 성능으로
해석하지 않는다.

고정 development evidence pool은 20 intent의 영어/한국어 paired query
40개와, 그 label에서 만든 22개 source span이다. 이는 **제품 chunker나
holdout 평가가 아닌 S0 재순위 검사**다. raw JSON은 아래 artifact에 남긴다.

| 모델 / 설정 | all Hit@5 / MRR@10 | EN Hit@5 / MRR@10 | KO→EN Hit@5 / MRR@10 | load / scoring |
| --- | --- | --- | --- | --- |
| tiny-en, L1/t2/b1, English only | 1.000 / 1.000 | 1.000 / 1.000 | 해당 없음 | 324ms / 8,615ms |
| compact, L1/t2/b1 | 0.900 / 0.830 | 1.000 / 0.950 | 0.800 / 0.710 | 3,473ms / 37,379ms |
| compact, L3/t4/b1 | 0.925 / 0.798 | (raw artifact 참조) | 0.850 / 0.647 | 7,423ms / 22,927ms |
| GTE INT8, L1/t1/b1 | 0.925 / 0.875 | 1.000 / 0.975 | 0.850 / 0.775 | 6,987ms / 188,279ms |

- compact L1: [s0-compact-b1.json](../artifacts/benchmarks/s0-compact-b1.json)
- compact L3: [s0-compact-b1-l3-t4.json](../artifacts/benchmarks/s0-compact-b1-l3-t4.json)
- GTE: [s0-gte-b1.json](../artifacts/benchmarks/s0-gte-b1.json)
- tiny English baseline: [s0-tiny-en-b1.json](../artifacts/benchmarks/s0-tiny-en-b1.json)

성능 목표인 “128 passages × 최대 256 tokens, model-load 제외 median 10초”는
S0에서 통과하지 못했다. 최대 길이에 가깝게 만든 동일 passage 128개에서
compact의 관찰된 최선(L3/t4/b1)은 246 tokens, 추론 15,613ms였고,
GTE는 246 tokens, 61,515ms였다. GTE를 더 느린 제품 기본값으로 바꿔 이
실패를 해결하지 않는다.

- compact 128-passages: [s0-compact-long-128-l3-t4-b1.stdout](../artifacts/benchmarks/s0-compact-long-128-l3-t4-b1.stdout)
- GTE 128-passages: [s0-gte-long-128-l3-t4-b1.stdout](../artifacts/benchmarks/s0-gte-long-128-l3-t4-b1.stdout)

GTE의 실제 local artifact는 변환본
`onnx-community/gte-multilingual-reranker-base` @
`ee64367e35a2db0da46bb6497e13a18f8bd585cb`의 `onnx/model_int8.onnx`,
340,858,200 bytes, SHA-256
`ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35`이다.
matching tokenizer는 17,082,999 bytes, SHA-256
`3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98`이다.
이는 비교용이고 registry의 제품 다운로드 profile에는 넣지 않는다.

## FP32 대조

선택한 compact revision의 원본 `onnx/model.onnx`도 실제로 내려받아
대조했다: 470,883,696 bytes, SHA-256
`3e9a03ed1e966f7c5288dd4230e3d6a9bf5e3a170a06f1f4241c5bca12c6487c`.
한국어 retry query와 관련 passage 1개/무관 passage 2개를 같은 tokenizer,
L1/t1/b1로 점수화했을 때 FP32와 ARM INT8 모두 관련 passage를 1위로
놓았다.

| artifact | relevant / distractor 1 / distractor 2 raw logit | load | inference (3 single calls) | peak RSS |
| --- | --- | --- | --- | --- |
| FP32 | 3.0824065 / -7.8468723 / -7.0102391 | 7,378ms | 190ms | 1,212,608 KiB |
| ARM INT8 | 3.3387182 / -8.0815077 / -6.3085341 | 4,244ms | 74ms | 527,232 KiB |

원시 값은 같을 필요가 없으며, 이 소규모 순위 대조는 holdout 품질 대체가
아니다. raw evidence는 [FP32](../artifacts/benchmarks/s0-compact-fp32-rank.stdout)와
[INT8](../artifacts/benchmarks/s0-compact-int8-rank.stdout)에 있다.

## S5 품질 결과와 지연 제한

Holdout은 프로필·기본 batch 설정을 고정한 뒤 한 번 평가했다. 결과를 보고
기본값을 조정하지 않았다. 보유한 corpus는 40개의 제품 청크(모두 K=128
이하)이므로 이 품질 run의 fast/deep 결과가 같고, 큰 corpus에서의 fast
후보 회수율을 증명하지 않는다.

| Holdout slice | Deep Hit@5 | Deep MRR@10 | Fast evidence recall | Lexical Hit@5 |
| --- | ---: | ---: | ---: | ---: |
| English | 0.85 | 0.758 | 0.90 | 0.80 |
| Korean | 0.85 | 0.733 | 0.90 | 0.05 |
| Korean query → English code (10 queries) | 0.70 | 0.650 | 0.80 | 0.10 |
| Low lexical overlap (10 queries) | 0.70 | 0.650 | 0.80 | 0.40 |

계획에 적힌 EN/KO deep Hit@5 ≥0.80, EN fast evidence recall ≥0.90,
EN fast Hit@5 ≥ matched lexical 및 deep 대비 손실 10%p 이내, low-overlap
재정렬 개선 기준은 이 고정 set에서 통과했다. 다만 public snapshot code
slice는 deep Hit@5 0.70, fast evidence recall 0.80에 그쳤다. 두 공개 code
label은 34/54줄 excerpt 전체를 gold로 가리킨다. 한 chunk가 label의 절반
이상을 덮어야 적중인 고정 규칙과 256-token 입력 한계 때문에 recall이
보수적일 수 있다. 평가 뒤 label은 바꾸지 않았다.

이 표의 영어·한국어 전체 행은 각각 20개 질의이고, 코드 행만 10개다.
기존 전체 한국어 수치 0.85를 한국어 코드 검색 수치로 표기한 오류를
정정했다. 모든 40개 청크가 K=128 안에 들어갔으므로 이 표의 fast
evidence recall은 실제 후보 축소 효과가 아니라 gold의 청크 표현 가능성도
포함한다. 평가기 재집계에서 표현 가능한 gold만을 분모로 두면 이 고정
corpus의 conditional candidate recall은 1.00이다.

### 더 큰 corpus의 후보 축소 감도 검사

원래 28개 fixture를 그대로 두고 무관한 1줄 텍스트 파일 240개를 더한 별도
corpus에서, 같은 frozen model과 holdout 질의 40개를 다시 평가했다. 전체
280개 청크 중 fast는 128개를 선택한다. 이는 이미 검토한 질의·label을
재사용한 **감도 검사**이며 새로운 blind holdout 성적은 아니다. 원본
fixture와 label은 수정하지 않았다.

| 질의 언어 | Deep Hit@5 | Fast Hit@5 | Fast gold recall | 표현 가능 gold 중 fast recall |
| --- | ---: | ---: | ---: | ---: |
| English (20) | 0.85 | 0.85 | 0.90 | 1.00 |
| Korean (20) | 0.85 | 0.50 | 0.50 | 0.56 |
| 전체 (40) | 0.85 | 0.675 | 0.70 | 0.78 |

마지막 열은 해당 질의에서 모델 입력 청크 하나로 표현할 수 있는 gold가
있는 경우만의 macro 평균이다(전체 36개, 한국어 18개). 한국어 8개는
표현 가능한 정답이 fast 후보에서 빠졌다. 따라서 원래 작은 corpus의
fast/deep 일치는 대형 corpus의 한국어 검색 품질로 일반화할 수 없다.
이 감도 검사에서 영어 후보 회수율 목표는 0.90으로 간신히 충족하지만,
한국어 fast/deep Hit@5 차이는 35%p다. 기존 15초/10초 지연 목표도
계속 별도 미충족이다. 원시 기록은
[scale run](../artifacts/evaluation/compact-multilingual-scale-v2-holdout.jsonl)과
[집계](../artifacts/evaluation/compact-multilingual-scale-v2-holdout-summary.json)에 있다.

개발 set에서 수행한 두 비교는 holdout과 분리했다. 모든 비교는 같은 제품
청크 40개를 사용했고 후보 회수율은 1.0이어서 관찰된 차이는 후보 누락이
아닌 모델 순위 차이다.

| Development configuration | Korean Hit@5 | Korean MRR@10 | Low-overlap Hit@5 | Decision |
| --- | ---: | ---: | ---: | --- |
| compact, batch 1 (selected) | 0.70 | 0.609 | 0.90 | Frozen default |
| GTE multilingual, batch 1 | 0.85 | 0.694 | 1.00 | Quality better, 128×246-token inference 61.5 s; not viable as default |
| compact, batch 4 | 0.75 | 0.625 | 0.80 | Small ranking change, does not meet 0.80 Korean target |

GTE’s independent 20-query Korean evaluation also took roughly 2.5 minutes
end-to-end. Batch-4 values are from a separate development-only run; default
batch remains 1 because quantized score values can vary with batch composition.
Raw records: [compact development](../artifacts/evaluation/compact-multilingual-development-summary.json),
[GTE Korean development](../artifacts/evaluation/gte-development-ko-summary.json),
[compact batch 4](../artifacts/evaluation/compact-multilingual-development-ko-b4-summary.json),
and [frozen holdout](../artifacts/evaluation/compact-multilingual-holdout-summary.json).

The 21 fresh-process fast runs over the deterministic 10 MiB / 1,000-file
corpus had 25.69 s median / 26.42 s p95 process wall time, 1.64 s median model
load, 10.24 s median inference, and 733,456 KiB p95 sampled peak RSS. The RSS
target (1.5 GiB) passed. The 15 s fast CLI goal and 10 s 128-passage model
inference goal did not: the isolated long-passage S0 run was 15.613 s, and the
product fast inference median was 10.236 s.

The phase data identifies two comparable costs rather than a single startup
problem: `chunking_ms` median was 10.709 s for all 29,000 chunks, including
exact tokenizer pair-length fitting before candidate selection; `search_ms`
median was 12.674 s, of which the scorer reports 10.236 s as ONNX inference for
the 128 selected chunks. Thus reading/discovery (33 ms median) and model load
(1.645 s) are not the dominant fast-path costs. The `search_ms` parent timer
already contains inference and must not be added to it. Evidence:
[`cli-fast-10MiB-1000files.json`](../artifacts/benchmarks/cli-fast-10MiB-1000files.json).

다른 development 질의 21개를 각각 fresh process로 실행한 재측정에서도
20개 steady-state run의 wall median/p95는 26.615/31.630초, inference
median은 10.501초, sampled RSS p95는 733,648 KiB였다. 두 지연 목표는
여전히 미충족이다. 첫 실행은 wall 30.181초였다. 이전과 같은 반복
합성 corpus이므로 실제 저장소 전체의 분포를 대표하지 않는다. 질의 ID,
binary SHA-256, corpus digest 및 원시 시간은
[`cli-fast-varied-10MiB-1000files.json`](../artifacts/benchmarks/cli-fast-varied-10MiB-1000files.json)에 있다.
선택한 첫 21개 development 기록은 코드 질의 20개와 설정 질의 1개다.
영어·한국어 및 질의 길이는 달라지지만 category별 균형 표본은 아니다.

Deep stress runs use the same deterministic 10 MiB / 1,000-file corpus with
`--max-chunks` set to each row's exact chunk count. The app therefore exits 3
and reports partial scan coverage by design, while every retained chunk was
scored (`scoring_complete=true`). The two entries are individual fresh CLI
processes, not a statistically stable percentile:

| Constructed/evaluated chunks | Batch inference (s) | Process wall time (s) | Sampled peak RSS (KiB) |
| ---: | ---: | ---: | ---: |
| 128 | 9.729 / 10.031 | 12.102 / 12.419 | 727,248 / 726,928 |
| 512 | 40.391 / 48.170 | 43.088 / 51.018 | 727,744 / 727,456 |
| 2,048 | 321.882 / 171.699 | 326.242 / 175.918 | 727,264 / 728,064 |

The 2,048-chunk timing varies widely between these two observations. These
measurements are throughput evidence, not a latency guarantee. Raw JSON is in
[`artifacts/benchmarks/`](../artifacts/benchmarks/) and the exact commands and
status are in [`docs/progress.md`](progress.md).

The later implementation pass reused the model-tokenized query and calculated
independent exact pair lengths concurrently. The pinned tokenizer's prepared
pair counts matched direct encoding on the fixed evaluation queries and files;
the frozen holdout JSONL was byte-for-byte identical after the change. On the
same 21-query synthetic workload, median chunking fell from 11.405 s to
5.019 s and process wall time from 26.615 s to 21.580 s. Inference median was
11.665 s in this later run, and its wall p95 rose to 36.346 s. The 15 s/10 s
goals still miss, and inference timing varied across runs. Evidence:
[`parallel fitting benchmark`](../artifacts/benchmarks/cli-fast-parallel-fitting-varied-10MiB-1000files.json).

The original small-corpus quality targets pass, but the larger-corpus Korean
candidate sensitivity check remains poor. The speed change does not alter
candidate selection or scoring, so it does not address that recall problem.
Changing default K, chunk length, or batch score composition would alter
recall/ranking behavior; the measured batch-4 development result reached
Korean Hit@5 0.75, below the 0.80 target.
On the optimized binary, a three-query development timing check with
`--batch-size 4` also increased inference time to 20.056/22.013/13.052 s,
versus 10.471/12.123/10.107 s for batch 1 on the corresponding workload.
This short comparison is not a full percentile estimate, but it gives no
speed reason to change the frozen batch-1 default.
