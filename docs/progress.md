# supergrep v0.1 진행 기록

이 파일은 실행 시점의 증거를 기록한다. 계획 문서의 예시나 fake scorer
테스트만으로 실제 모델 검증을 완료로 표시하지 않는다.

## S0 — 모델·런타임 실험 (선택 완료, 제품 성능 목표는 미달)

- 2026-09-22: 빈 저장소와 적용 가능한 `AGENTS.md` 부재를 확인했다.
  실제 실행 환경은 Linux `aarch64`, Cortex-A76 4 CPU, RAM 7.9 GiB다.
  초기에는 available RAM 약 1.1 GiB/swap 2 GiB 포화였고, 이후 측정 시에는
  약 4.5–4.7 GiB available이 관찰됐다. 다른 작업 부하를 멈추거나 swap을
  변경하지 않았으므로 이 수치는 일반적인 하드웨어 성능 주장이 아니다.
- 2026-09-22: `ort = ort-sys = 2.0.0-rc.9`를 정확히 lock하고
  dynamic-loading feature만 사용했다. Debian system ORT 1.21.0은 대량의
  duplicate schema 경고 때문에 배포 대상에서 제외했다. 공식 Linux ARM64
  ORT 1.20.0 archive/library hash와 runtime resolver는
  [model-decision.md](model-decision.md)에 기록했다.
- 2026-09-22: immutable revision의 tiny, compact, GTE INT8 artifact와
  필요한 tokenizer를 temporary file → SHA-256 → rename 방식으로 로컬 개발
  캐시에 준비했다. 모델 다운로드는 이 단계의 명시적 실험만 사용했고,
  search 경로에는 network fallback을 넣지 않았다.
- 실제 모델 실행 명령의 형태:

  ```bash
  cargo run --locked -j 1 --bin s0_probe -- \
    --model <pinned-local-onnx> --tokenizer <pinned-local-tokenizer> \
    --runtime <official-ort-1.20.0-library> --threads <n> \
    --max-pair-tokens 256 --max-query-tokens 64 --pad-id <id> \
    --optimization <level> --score-batch-size <n> --query <query> <passage...>
  cargo run --locked -j 1 --example s0_compare -- \
    --model <pinned-local-onnx> --tokenizer <pinned-local-tokenizer> \
    --runtime <official-ort-1.20.0-library> --split development
  ```

- 실제 local ONNX graph/score를 확인했다. compact는
  `input_ids`/`attention_mask` → single `logits`, GTE/tiny는
  `input_ids`/`attention_mask`/`token_type_ids` → single `logits`이며,
  finite raw logit을 반환했다. GTE tokenizer JSON에 들어 있던 512 fixed
  padding/truncation을 해제하는 버그도 수정했다. 그렇지 않으면 짧은 한국어
  query가 512 tokens로 잘못 측정됐다.
- development evidence-pool 결과와 raw output:
  `artifacts/benchmarks/s0-tiny-en-b1.json`,
  `s0-compact-b1.json`, `s0-compact-b1-l3-t4.json`,
  `s0-gte-b1.json`. compact L1/t2/b1의 Korean→English Hit@5/MRR@10은
  0.800/0.710, GTE L1/t1/b1은 0.850/0.775이지만 GTE 전체 scoring은
  188,279ms였다. 이 pool은 후보 recall/holdout 평가가 아니다.
- 128개 × 246-token의 model-load 제외 실제 추론은 compact L3/t4/b1
  15,613ms ([raw](../artifacts/benchmarks/s0-compact-long-128-l3-t4-b1.stdout)),
  GTE L3/t4/b1 61,515ms
  ([raw](../artifacts/benchmarks/s0-gte-long-128-l3-t4-b1.stdout))였다.
  따라서 10초 목표는 **미달**이다. 이 사실을 축소하거나 GTE로 숨기지 않고,
  S5에서 전체 CLI의 두 번 이내 개발-set 개선으로 원인을 분리한다.
- compact original FP32 (470,883,696 bytes,
  `3e9a03ed…c6487c`)와 ARM INT8은 같은 한국어 retry query의 관련 passage를
  같은 3-way 순위 비교에서 모두 1위로 놓았다. raw logit/시간/RSS는
  [FP32](../artifacts/benchmarks/s0-compact-fp32-rank.stdout),
  [INT8](../artifacts/benchmarks/s0-compact-int8-rank.stdout)에 남겼다.
- 기본 제품 profile은 `compact-multilingual`, L3/t4/b1로 선택했다.
  이유, revision/hash, batch-dependent INT8 logit 관찰, GTE 비교와 미완료
  조건은 [model-decision.md](model-decision.md)에 모두 기록했다.

## S1 — 탐색·source snapshot·청크 (구현 및 테스트 완료)

- 2026-09-22: `src/discovery.rs`, `src/source.rs`, `src/chunk.rs`를
  추가했다. deterministic walk, `.git` 영구 제외, ignore/hidden 분리,
  explicit file/symlink 정책, 2MiB/file·64MiB total·50,000 chunk 기본
  한도와 reasoned diagnostics를 구현했다.
- source는 bytes/text/line starts/metadata snapshot을 보존하고 CRLF,
  Unicode, 마지막 newline 없음, long line의 byte/line 좌표를 테스트한다.
  chunker는 line/paragraph 우선, UTF-8-safe split/overlap/cap을 제공한다.
- 검증: `cargo clippy --locked --lib --test discovery --test chunking -- -D warnings`,
  `cargo test --locked` (당시 34 passed), independent review
  `.omo/evidence/s1-foundation-code-review.md` (27 focused tests).
- 미완료 handoff: 제품 search가 real query/passage token 길이를 써서
  `chunk_source_with_token_count`로 모델 입력 전 재분할해야 한다.

## S2 — 모델 수명·runtime resolver (구현 완료, CLI wiring 진행 중)

- 2026-09-22: `models/registry.toml`에 immutable compact/tiny revision,
  model hash/size 및 compact tokenizer hash/size를 등록했다.
  `src/model/{registry,download}.rs`는 HTTPS explicit download만 허용하며
  staging file hash/size/fsync, revision directory atomic rename, file lock,
  local-only verify를 구현한다. registry/verify/search는 socket을 열지 않는다.
- `src/model/runtime.rs`는 absolute `SUPERGREP_ORT_LIB` 또는 supplied
  executable 옆 bundle만 해석하고 cwd를 탐색하지 않는다. `tests/runtime.rs`
  6개 통과, lifecycle `tests/inference.rs` 7개 통과를 보고받았다.
- tiny tokenizer의 independent SHA/size는 아직 registry에 없어 downloader가
  tiny profile의 companion tokenizer를 자동 준비하지 않는다. 이를 성공으로
  표시하지 않으며, 제품 CLI에서는 명확히 안내해야 한다.

## S3 — lexical/fast/deep engine (진행 중)

- 2026-09-22: `src/lexical.rs`와 focused test를 추가했다. Unicode NFKC,
  Korean/identifier split, current-query BM25, path/body score 분리와
  deterministic tie를 구현했다. `cargo test --locked --test lexical -j 1`
  은 5 passed였다.
- fast/deep candidate engine과 model-exact re-splitting을 이어서 통합하고,
  fake scorer 테스트와 실제 compact integration을 모두 수행해야 한다.

## 평가 corpus

- 2026-09-22: 직접 작성한 UTF-8 corpus/paired English+Korean development
  및 holdout query를 유지하면서, holdout intent 10개를 public immutable
  snapshots로 교체했다: `serde-rs/json` @ `afdf6fc…` 5개 (MIT OR Apache-2.0),
  `BurntSushi/ripgrep` @ `3fce3b5…` 5개 (Unlicense OR MIT). source URL,
  commit, license, original line/path, local SHA-256은 `eval/manifest.json`에
  있다.
- 검증: `python3 -B scripts/evaluate.py --check` → 28 fixtures, 90 query
  records, valid. 아직 final product evaluator JSONL/holdout metric은 없다.

## S1–S6 최종 구현 체크포인트 (2026-09-23)

위 S1–S3 항목은 해당 날짜의 작업 중간 체크포인트다. 아래의 완료 상태가
이후 변경을 반영하며 이전의 미완료 문장을 대체한다.

### S1 — 탐색·snapshot·위치 보존 청크: 완료

- ignore/hidden/glob/direct-root 정책, `.git` 영구 제외, symlink 거부,
  UTF-8/NUL 필터, 파일/총량/chunk cap, 결정적 순회 및 부분 상태/진단을
  구현했다. 실제 원문 snapshot 기준으로 0-based half-open byte 위치와
  1-based inclusive line을 제공한다.
- CRLF·한글·emoji·긴 줄·마지막 newline·비 UTF-8 경로의 좌표 및 safe JSON/
  human 출력은 단위/통합 테스트로 확인했다. 매 CLI 실행은 새로 순회하고
  영구 검색 인덱스를 사용하지 않는다.

### S2 — 모델 준비·검증·추론: 완료

- `models/registry.toml`은 `tiny-en`과 `compact-multilingual`의 immutable
  revision, 모델과 tokenizer SHA-256/byte 수, graph/tokenizer 규약, input
  이름, raw relevance logit, ONNX Runtime wrapper/runtime 버전을 고정한다.
  이전 중간 기록의 tiny tokenizer 누락은 정식 registry artifact와 verify
  테스트를 추가해 해결했다.
- 검색은 cache verify 뒤 실제 ONNX CPU session을 로드하며 네트워크/API 호출
  경로가 없다. 다운로드만 사용자가 명시적으로 요청할 수 있고, staging/hash
  검증을 거친다. 누락·손상·불일치 모델은 오류로 반환하고 lexical fallback을
  하지 않는다.
- 실제 release binary/model/runtime 검색 smoke에서 product model ID/revision,
  `scoring_complete=true`, 모델 load·tokenization·inference timing을 확인했다.

### S3 — lexical / fast / deep: 완료

- `--lexical`은 positive BM25만 출력하며 모델/ORT/cache를 준비하지 않는다.
  fast는 최대 K=128의 결정적 lexical/path/neighbor/diverse candidates,
  deep는 생성된 모든 model-fitting chunk를 평가한다. fast coverage, lexical
  evidence, scoring completeness, 처리 제한과 진단을 JSON/human에 표시한다.
- 테스트: N≤K fast/deep 동일 score 순위, N>K fast cap/deep 전체 scoring,
  zero lexical evidence, Korean query broad sampling, batch↔chunk 정렬,
  nonfinite/mismatch 오류, overlap duplicate suppression.

### S4 — CLI·오프라인·종료 동작: 완료

- `model list/download/verify`, `doctor`, search help 및 README 명령을 구현했다.
  search/model list/model verify/doctor/lexical은 cache 외 네트워크에 접근하지
  않고 model download만 네트워크를 사용하는 명령이다. CLI JSON은 stdout의
  단일 object, 진단/오류는 stderr다.
- `tests/cli.rs`의 closed-stdout-pipe 및 SIGINT 시나리오에서 panic이 없음을
  확인했다. 같은 test suite에서 실행 간 edit/rename/delete 반영과 total-byte
  cap의 JSON partial/exit 3도 확인한다. unreadable file은 `ReadFailure`와
  partial 상태를 낸다.
- 실제 bundle/model의 offline 통합 검증은 bundle smoke에서 OS syscall
  tracer로 수행했다. fresh temporary directory/CWD에서 `SUPERGREP_ORT_LIB`를
  unset한 packaged binary가 adjacent runtime으로 deep 검색·model load를
  완료했고, 실제 fixture와 반환 byte/line 범위도 일치했다.
  `strace -f -e trace=network`에서 socket/connect/sendto/recvfrom 호출이
  없었다. 승인된 ptrace 실행 evidence는
  `artifacts/validation/bundle-offline-span-final.log`이다. 일반 sandbox의
  첫 시도는 ptrace 제한으로 실패했고, 통과로 오인하지 않고 별도 기록했다.

### S5 — 평가·성능: 품질 목표 통과, latency 미달

- S0의 probe가 아닌 product `examples/evaluate.rs`로 development/holdout/
  irrelevant split을 평가했다. `python3 -B scripts/evaluate.py --check`는
  28 fixture, 90 query record와 serde/ripgrep 두 public immutable snapshot
  프로젝트를 검증한다. 결과 JSONL은 span-based gold, product candidate IDs,
  fast/deep/lexical 결과를 보존한다.
- Holdout은 compact batch-1을 고정한 뒤 평가했고 그 결과로 조정하지 않았다.
  EN deep Hit@5 0.85 / fast evidence recall 0.90 / lexical Hit@5 0.80;
  KO deep Hit@5 0.85 / lexical Hit@5 0.05; Korean→English code Hit@5 0.70
  (코드 10개 중 7개). 기존 0.85는 코드만이 아니라 한국어 전체 질의의 값이었다.
  어휘 겹침이 낮은 10 query의 deep Hit@5 0.70, lexical 0.40이다.
  이 작은 corpus는 40 chunk라 K=128 이하이며 fast candidate 정책의 대형
  corpus recall 증거는 아니다. Public code snapshot만 보면 deep Hit@5 0.70,
  evidence recall 0.80으로 더 낮고, 34/54-line gold excerpt 2개는 선언된
  half-span evidence 규칙상 chunk limit 때문에 보수적으로 집계될 수 있다.
  label은 결과 확인 후 수정하지 않았다.
- Development 비교 1: 같은 40 chunk Korean set에서 GTE multilingual
  batch-1은 Hit@5 0.85/MRR@10 0.694로 compact batch-1 0.70/0.609보다
  품질이 높았으나 128×246-token S0 inference가 61.515초였다. 제품 기본값으로
  전환하지 않았다.
- Development 비교 2: compact batch-4는 Korean Hit@5 0.75/MRR@10 0.625로
  compact batch-1을 소폭 개선했지만 0.80 목표에는 못 미쳤고, batch 조성에
  따른 quantized raw logit 변동 위험이 있어 기본 batch-1을 유지했다. 이는
  두 개발-set 실험 이후의 선택 결정이며 추가 속도 조정은 수행하지 않았다.
- 고정 10 MiB/1,000 file fast benchmark, 21 fresh process: wall time median
  25,686.7ms / p95 26,418.7ms; model load median 1,644.5ms; batch inference
  median 10,235.5ms; RSS sampled p95 733,456 KiB. 1.5 GiB memory 목표는 통과.
  15초 CLI와 128×≤256-token 10초 inference 목표는 통과하지 못했다. 동일한
  긴 246-token input의 S0 128-passage inference는 15,613ms였다.
- Deep stress command:

  ```bash
  SUPERGREP_MODEL_CACHE=/tmp/supergrep-model-cache.bLSH09 \
  SUPERGREP_ORT_LIB=/home/eunhhu/work/supergrep/.supergrep-dev-runtime/onnxruntime-linux-aarch64-1.20.0/lib/libonnxruntime.so.1.20.0 \
    bash scripts/benchmark-deep.sh target/release/supergrep 2 artifacts/benchmarks
  ```

  `--max-chunks`를 명시해 128/512/2,048 retained chunks를 전부 score했다.
  따라서 각 CLI 결과는 `scoring_complete=true`, `partial=true`,
  `scan_complete=false`인 의도적 bounded-stress 결과다. 첫/두 번째 fresh
  process의 batch inference / process wall time은 각각 128: 9.729/12.102초,
  10.031/12.419초; 512: 40.391/43.088초, 48.170/51.018초; 2,048:
  321.882/326.242초, 171.699/175.918초다. RSS는 모두 약 727–728 MiB였다.
  2,048은 두 관측의 편차가 커 성능 보장 또는 안정된 percentile로 취급하지
  않는다. raw 결과는 `artifacts/benchmarks/cli-deep-*-chunks.json`이다.

### S6 — 문서·release·로컬 bundle: 완료

- `cargo build --release --locked -j 1 --bin supergrep --example evaluate`로
  ARM64 release를 빌드했다. `artifacts/packages-final/`의 bundle은 binary,
  ONNX Runtime 1.20.0 ARM64 library/provider library, registry, README 및
  Apache/MIT/ThirdParty notices를 포함하며 model은 포함하지 않는다.
- fresh temporary directory에서 다른 CWD로 이동하고
  `SUPERGREP_ORT_LIB`를 unset한 채 bundle help, `doctor`, 실제 model의
  `--deep` 검색을 실행했다. adjacent runtime, 명시한 prepared model cache,
  exact byte/line coordinates, score coverage 확인 및 no-network syscall
  trace가 모두 통과했다.
- 최종 tarball SHA-256:
  `e5c3ada1aa02b869a71320f152f6f6057680364aacec2645170f591b48369d4c`
- 새 bundle smoke log: `artifacts/validation/bundle-offline-span-final.log`.

### 최종 검증 명령 및 남은 사용자 결정

- `cargo fmt --all --check`, `cargo clippy --locked --all-targets -j 1 -- -D warnings`,
  `cargo test --locked -j 1`: 현재 통합 회귀 보강 후 78 passed, 1 ignored
  (opt-in offline test), 0 failed. pipe-close, SIGINT, edit/rename/delete,
  read failure, partial exit-code tests가 포함된다. 최종 재실행 로그:
  `artifacts/validation/span-checks.log`.
- Offline integration, 허용된 `ptrace` 환경:

  ```bash
  SUPERGREP_TEST_MODEL_CACHE=/tmp/supergrep-model-cache.bLSH09 \
  SUPERGREP_TEST_ORT_LIB=/home/eunhhu/work/supergrep/.supergrep-dev-runtime/onnxruntime-linux-aarch64-1.20.0/lib/libonnxruntime.so.1.20.0 \
    cargo test --locked -j 1 --test offline -- --ignored --nocapture
  ```

  1 passed, 실제 model search의 exact byte/line evidence와 network syscall 없음.
  evidence: `artifacts/validation/offline-span-final.log`. 일반 sandbox의
  ptrace 거부 evidence는 별도 `offline-test.log`로 남아 있으며 성공 처리에
  섞지 않았다.
- 평가 manifest 정의 검사는 28 fixture/90 record로 유효하다.
  evidence: `artifacts/validation/evaluation-definition-final.json`.
- 이 단계 당시에는 개발-set model 비교와 batch 설정 비교가 끝났고
  두 ARM latency 목표가 미달하여 지연 수용·추가 실험·목표 수정 중
  선택이 남았다고 기록했다. 아래 리뷰 후 검증에서 한국어 fast 후보
  회수율 문제도 확인했으므로 현재 남은 문제를 지연 하나로 보지 않는다.

## 재개된 goal의 지연 원인 진단 (2026-09-23, 설정 변경 없음)

- 현재 fast benchmark의 20 steady-state process 중앙값을 분해하면 exact
  tokenizer pair-length fitting을 포함한 all-chunk construction/fitting은
  10.709초(29,000 chunks), selected 128-chunk search call은 12.674초이며
  이 안의 ONNX inference는 10.236초다. 모델 load 중앙값은 1.645초,
  discovery는 33ms다. `search_ms`는 inference를 포함하는 parent timer라
  둘을 합산하면 안 된다. 증거: `artifacts/benchmarks/cli-fast-10MiB-1000files.json`.
- 따라서 지연은 모델 초기화만의 문제가 아니다. 추가 실험이 승인되면 첫
  저위험 진단은 exact pair-length tokenizer 호출을 batch/vectorize해도 모든
  fitted byte span과 후보 ID가 완전히 동일한지 검증하는 것이다. 기본 K,
  chunk 크기 또는 scorer batch 조정은 recall/ranking tradeoff를 바꾼다.
  이미 실행한 compact batch-4 개발 비교의 Korean Hit@5는 0.75로 목표 0.80
  미달이어서 이를 기본값으로 채택하지 않았다.
- 이 진단은 실험·제품 설정 변경이 아니다. 사용자에게 선택을 요청한 범위
  (현재 한계 수용 / 추가 개발-set 실험 승인 / latency 목표 수정) 외의 실행은
  하지 않았다. Holdout 결과는 계속 동결한다.

## 리뷰 후 후보·평가 검증 (2026-09-23)

- `docs/review-2026-09-23.md`의 F1–F5를 점검했다. fast가 파일 경로 앞부분만
  표본으로 고르는 F2를 수정하여, 경로 범위 전체의 양끝과 중간을 단계적으로
  선택한다. 145개 파일 중 마지막 `z-target.rs`만 정답이고 lexical 근거가
  없는 회귀 사례에서 128개 후보 안에 포함된다.
- 기존 40청크 holdout의 fast 후보 recall 0.90은 축소 품질이 아니었다.
  표현 가능한 gold만 분모로 하면 1.00이지만 모든 청크가 K=128 안에
  들어갔기 때문이다. 평가기에 원래 gold의 청크 표현 가능성, fast 후보
  회수율과 조건부 회수율을 구분해 기록하도록 했다.
- 원본 28개 파일에 결정적 무관 텍스트 240개를 더한 별도 corpus는
  manifest/hash 검증을 통과했다. 실제 frozen model과 기존 holdout 40개로
  평가한 280청크 감도 검사는 영어 deep/fast Hit@5가 모두 0.85였지만,
  한국어는 deep 0.85, fast 0.50이었다. 표현 가능한 한국어 gold의 fast
  후보 회수율은 macro 0.56(18개 질의 중 8개에서 정답 후보 누락)이었다.
  이 결과는 후보 축소가 한국어 검색 품질을 크게 떨어뜨릴 수 있음을
  보여준다. 이미 본 label을 재사용했으므로 새로운 blind holdout 통과
  근거로 쓰지 않는다. raw/summary는
  `artifacts/evaluation/compact-multilingual-scale-v2-holdout*`이며,
  `scripts/generate_scale_corpus.py`로 corpus를 재생성할 수 있다.
- Korean→English **code** deep Hit@5는 원래 holdout에서 0.70(7/10)이다.
  한국어 전체 0.85와 분리하여 README·모델 결정 문서의 수치를 고쳤다.
- benchmark harness는 서로 다른 development 질의, binary/model/corpus
  식별자, 자식 출력용 임시 파일과 timeout을 사용한다. 8 MiB stdout을
  쓰는 가짜 CLI 회귀 테스트가 통과했다. 동일 질의만 21회 반복한 기존
  25.687초 값은 기존 workload의 결과로만 유지한다.
- 21개 서로 다른 development 질의, 각 fresh process로 고정 10 MiB/1,000
  파일을 재측정했다. 20개 steady-state run의 wall median/p95는
  26.615/31.630초, batch inference median은 10.501초, sampled RSS p95는
  733,648 KiB다. 첫 run wall은 30.181초였다. 15초 전체 실행 및 10초
  추론 목표는 미달한다. 합성 corpus와 당시 ARM 시스템에 대한 측정이며
  실제 저장소의 성능 보장은 아니다. raw:
  `artifacts/benchmarks/cli-fast-varied-10MiB-1000files.json`.
- 수정한 release binary를 이전 145파일 재현 corpus에서 실제 모델로
  실행했다. lexical 근거가 없고 `chunks_total=145`, `chunks_evaluated=128`인
  상태에서 마지막 경로 `z-target.rs`가 후보에 포함되어 최종 1위였다.
  증거: `/tmp/supergrep-review-selection-acyss315/fast-fixed.json`.
- 새 ARM64 bundle은 `artifacts/packages-reviewed/`에 만들었다.
  bundle 내부 binary와 측정 binary SHA-256은 모두
  `ce326eac169f1a67c7ff8d2e680978b685d0169b0a601917916ca771e2602b56`,
  tarball SHA-256은
  `0c8e0ebd8d128c9691a3b8b1069c681bd6eb93ac23c688684d9d8a039d25c82d`다.
  fresh directory에서 bundled runtime, 실제 모델, exact byte/line span,
  network syscall 미발생을 `scripts/test-bundle.sh`로 재검증했다.
- `cargo fmt --all --check`, `cargo clippy --locked --all-targets -j 1 -- -D warnings`,
  `cargo test --locked -j 1` 통과: Rust 79 passed / 1 opt-in ignored.
  Python benchmark 회귀 1개도 통과했고, 원본 및 268파일 확장 fixture의
  manifest 검사가 각각 통과했다.

## 정확한 토큰 fitting 병렬화 (2026-09-23)

- 모델 청크 fitting은 기존에 최대 29,000개 청크마다 동일 질의를 다시
  검증·토큰화했다. `PreparedPairQuery`가 질의를 한 번 준비하고, 각
  passage의 정확한 pair 길이를 pinned tokenizer의 post-processor로
  계산하도록 변경했다. 원래 모델의 90개 평가 질의와 전체 fixture 파일 및
  BERT WordPiece fixture에서 기존 직접 pair 길이와 일치했다.
- 독립 초기 청크의 pair 길이를 Rayon으로 병렬 계산한다. 결과는 원래
  source/byte 순서로 소비하고, 길이 초과 청크는 이전과 같은 분할 경로를
  사용한다. source ID lookup도 반복 선형 탐색에서 hash lookup으로
  바꿨다. UTF-8·CRLF 분할과 청크 상한에 대한 직렬/병렬 동치 테스트가
  통과했다.
- 고정 holdout 40질의를 수정한 release evaluator로 재실행했다.
  `artifacts/evaluation/compact-multilingual-holdout-parallel.jsonl`과 기존
  `compact-multilingual-holdout.jsonl`의 SHA-256은 모두
  `5f773503dc107a9eb512d927efedecbee02624396e6c154ea82008f776514b59`다.
  따라서 이 변경으로 청크 범위·후보·점수·순위가 바뀌지 않았다.
- 같은 10 MiB/1,000파일, 같은 21개 development 질의의 새 측정은
  chunking median 11.405→5.019초, CLI wall median 26.615→21.580초였다.
  inference median은 10.501→11.665초로 이 실행에서 오히려 높았고,
  wall p95도 31.630→36.346초로 악화했다. RSS p95는 736,336 KiB로
  여전히 1.5 GiB 이하이다. OS 부하를 통제하지 않은 다른 시점의 실행이라
  p95 변화의 원인을 구현에 단정하지 않는다. 원시 실행 21회:
  `artifacts/benchmarks/cli-fast-parallel-fitting-varied-10MiB-1000files.json`.
  15초 전체 실행과 10초 추론 목표는 여전히 미달이다.
- 같은 release binary와 고정 corpus의 첫 세 development 질의에서
  `--batch-size 4`를 별도로 측정했다. inference는 20.056/22.013/13.052초로,
  batch 1의 10.471/12.123/10.107초보다 모두 느렸다. 기존 개발 질의
  품질 비교도 batch 4를 기본값으로 채택할 근거가 없으므로 batch 1을
  유지한다. `artifacts/benchmarks/cli-fast-parallel-batch4-3runs.json`.
- `cargo fmt --all --check`, `cargo clippy --locked --all-targets -j 1 -- -D warnings`,
  `cargo test --locked -j 1`가 통과했다. pinned tokenizer의 별도 opt-in
  동치 테스트도 실제 준비된 모델 cache에서 통과했다. 새 ARM64 bundle은
  `artifacts/packages-parallel/supergrep-v0.1.0-linux-aarch64.tar.gz`이고,
  binary SHA-256은
  `99a87d47e3911c76ffea5f16a643685f18dcde72fb4d09fd8b94e8c8b2323535`,
  tarball SHA-256은
  `c97d34ac90e3e1251cc7089938408444b8ac772f9633b0017f37c34d9bdc0941`다.
  별도 디렉터리에서 실제 모델 검색, byte/line 및 network syscall 부재
  검증이 통과했다. 새 변경의 명령·exit·hash 증거는
  `artifacts/validation/parallel-fitting-2026-09-23.md`와
  `artifacts/validation/parallel-bundle-smoke.log`에 있다.
- 사용자 범위 결정: 측정된 지연과 대형 corpus의 한국어 fast 후보 회수율
  한계를 명시하고 **로컬 v0.1 구현을 완료**한다. 15초/10초 목표를 달성했다고
  주장하거나 수치를 조정하지 않는다. 공개 배포·원격 push는 이 범위 밖이다.
  계획의 완료 항목별 근거는 `docs/v0.1-audit.md`에 정리했다.
