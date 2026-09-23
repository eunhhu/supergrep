# supergrep v0.1 구현 계획

작성일: 2026-09-22. 상태: 구현 전 계획.

목표는 **프로젝트 폴더의 코드·문서·설정을 자연어로 찾는 로컬 텍스트 검색 CLI**다. 모델을 준비한 뒤에는 API 키, 네트워크, 사전 인덱싱 없이 실행한다. 검색 결과는 실제 파일 경로와 원본 위치로 반환한다.

이 문서는 기능 목록뿐 아니라 구현 순서, 모델 선택 실험, 실패 시 처리, 완료 증거를 정의한다. 후속 실행 요청은 `docs/implementation-goal.md`에 있다. 현재 단계에서는 모델 다운로드나 제품 구현을 수행하지 않았다.

## 1. 출발점과 결정 사항

| 항목 | 결정 / 확인 상태 |
| --- | --- |
| 작업 디렉터리 | `/home/eunhhu/work/supergrep` |
| 저장소 | 초기 커밋과 구현 파일이 없는 빈 Git 저장소 |
| 주 사용 환경 | **사용자 확인: Linux ARM64, CPU 4코어, RAM 8GB** |
| 관찰한 환경 | Debian 13, Cortex-A76, Rust/Cargo 1.98.1. 계획 작성 시점의 환경이며 실행 시 재확인 |
| 제품 언어 | Rust. 최종 사용자에게 Python 설치를 요구하지 않음 |
| 검색 파일 | 확장자 제한 없는 UTF-8 텍스트. 코드, Markdown, 문서, 설정, 제한 크기의 로그·CSV·JSONL |
| 질의 언어 | 계획 기본 가정: 한국어와 영어. 한국어→영문 코드 검색을 독립적인 품질 항목으로 검증 |
| 추론 | 프로세스 내부의 로컬 ONNX cross-encoder, CPU 우선 |
| 인덱스 | 사전 인덱싱 명령과 벡터 DB 없음. v0.1은 매 실행마다 읽으며, 디스크에는 모델만 캐시 |
| 배포 목표 | 우선 Linux ARM64에서 실행 가능한 로컬 배포 묶음. 공개 배포는 별도 단계 |

참고 대화의 마지막 제안인 텍스트 전체 지원, 빠른 검색과 `--deep`, 후보 누락 측정을 반영했다. 앞선 대화의 속도 수치, 코드 검색 성능, “바이너리 하나로 끝난다”는 설명은 검증된 요구 사항으로 취급하지 않는다.

### v0.1에 포함할 것

- 검색 루트 하나를 받아 코드·문서·설정을 함께 검색한다. 루트 생략 시 현재 디렉터리다.
- ignore 규칙, 숨김 파일 제외, 바이너리·잘못된 UTF-8 제외, 읽기 및 처리량 제한을 제공한다.
- 원본 위치를 보존하는 청크, 빠른 후보 검색, 로컬 모델 재정렬, 전체 청크 평가를 구현한다.
- 사람이 읽는 출력과 버전이 있는 JSON 출력, 모델 준비·검증 명령, 오프라인 동작을 제공한다.
- 실제 모델을 실행한 통합 검증과 ARM 환경의 속도·메모리·검색 품질 결과를 남긴다.

### 다음 버전으로 남길 것

Tree-sitter/심볼 검색, PDF·DOCX·이미지·압축파일, UTF-16 변환, stdin/로그 follow, GUI·MCP 서버·상주 데몬, GPU 최적화, 임베딩 검색, 영구 검색 인덱스, 모델 학습·증류, 자동 번역 LLM은 제외한다. 구조는 확장 가능하게 하되 플러그인 프레임워크까지 만들지 않는다.

## 2. 가장 먼저 검증할 위험

1. **후보 회수율:** 단어가 겹치지 않는 정답은 lexical 검색으로 찾기 어렵다. 다국어 reranker만 붙여도 이 문제는 남는다. `--deep`은 같은 대상 청크 전체를 평가해 후보 누락과 모델 순위 오류를 분리하는 기준선이다. 이 구조의 근거는 [Sentence Transformers의 retrieve/rerank 설명](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)에 있다.
2. **ARM에서의 사용성:** 후보 수뿐 아니라 입력 길이, 토큰화, 모델 초기화, 메모리, 런타임 배포가 비용이다. 후보 128개에 몇 ms라는 약속은 측정 전에는 하지 않는다.
3. **한국어→영문 코드:** 다국어 문서 검색 지원과 이 기능의 품질은 별개다. 작은 모델이 기준에 미달하면 더 큰 모델과 속도 차이를 측정한다.
4. **ONNX 호환성:** 양자화 그래프, tokenizer, 입력 tensor, 런타임 ABI가 함께 맞아야 한다. CLI를 완성한 뒤 마지막에 발견하지 않도록 S0에서 확인한다.

## 3. 모델 선택: 작은 실험 후 기본값 고정

아래는 비교 후보이며 성능 순위가 아니다. 용량은 확인한 ONNX 파일 하나의 표시 크기이고 tokenizer·런타임·실행 메모리를 포함하지 않는다.

| 프로필 이름 | 모델 / 초기 아티팩트 | 역할과 한계 |
| --- | --- | --- |
| `tiny-en` | `cross-encoder/ms-marco-MiniLM-L6-v2`, `onnx/model_qint8_arm64.onnx`, 약 23.2MB | 영어 속도 기준선. 영어 모델이므로 한국어 지원 기본값으로 채택하지 않음 |
| `compact-multilingual` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `onnx/model_qint8_arm64.onnx`, 약 119MB | **먼저 실험할 다국어 기본값 후보**. 한국어 코드 검색은 미검증 |
| `quality-multilingual` | `Alibaba-NLP/gte-multilingual-reranker-base`; ONNX 변환본 `onnx-community/gte-multilingual-reranker-base`의 `onnx/model_int8.onnx`, 약 341MB | 경량 후보의 한국어 품질이 부족할 때 비교. ARM 실행·원본 대비 출력 검증 후 사용 |

영어 모델은 MS MARCO passage ranking으로 학습되었고 Apache-2.0으로 표시되어 있다. 공식 모델 저장소에 ARM 양자화 파일이 있다. [모델 카드](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [파일 목록](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main/onnx).

경량 다국어 모델도 Apache-2.0이며 번역된 mMARCO로 학습되었다. 이 설명만으로 한국어→코드 품질을 보증할 수 없다. [모델 카드](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1), [ARM 파일 목록](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/main/onnx).

GTE 원본은 306M 파라미터, 다국어, Apache-2.0으로 표시되어 있다. ONNX Community 변환본은 원본과 구분하여 출처·revision·변환 차이를 기록한다. 긴 context 지원을 이유로 v0.1의 입력을 8K로 늘리지 않는다. [원본 카드](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base), [변환본 파일 목록](https://huggingface.co/onnx-community/gte-multilingual-reranker-base/tree/main/onnx).

### S0 실험 절차

1. 작은 Rust 실행 예제로 ARM64 ONNX Runtime 로딩과 실수 score 출력을 확인한다. 초기 성공에는 `tiny-en`을 쓰고, 다국어 후보로 이어간다.
2. 의미가 같은 영어·한국어 질의와 정답·그럴듯한 오답을 갖춘 개발용 12개 intent를 만든다. 파일 종류를 섞고 동의어·부정·다른 구현을 포함한다.
3. 모델마다 같은 query/passage 쌍을 단건과 batch로 실행한다. tokenizer의 special tokens, padding, query/passage 순서, ONNX 입출력 이름·형태·dtype을 기록한다.
4. FP32 원본/검증된 reference와 양자화 모델을 비교한다. FP32 구현 간 오차, INT8 순위 변화는 다른 검증 항목이다. reference 도구가 필요하면 격리된 개발용 Python 환경만 허용한다.
5. 입력 길이 256, batch 1/4/8, 추론 thread 1/2/4를 작은 표본으로 비교한 뒤 상위 설정 두 개만 상세 측정한다. 측정마다 모델 하나만 로드한다.
6. 경량 다국어가 개발 집합에서 품질·속도 목표를 만족하면 기본 프로필로 선정한다. 미달하면 GTE를 한 번 비교한다. 이때는 개발 집합만 사용하고, 최종 holdout 통과는 S5에서 확인한다. 영어만 되는 모델로 바꿔 한국어 지원이 완료되었다고 처리하지 않는다.
7. 결과와 선정 이유를 `docs/model-decision.md`에 작성하고, 선정한 모델과 비교용 영어 모델만 정식 다운로드 대상으로 제공한다. 실험 후보를 전부 제품 옵션으로 유지할 필요는 없다.

`models/registry.toml`에는 실제로 확인한 immutable revision, 파일별 SHA-256/크기, 출처, 라이선스, tokenizer 및 입력 규약, score 변환, 모델 최대 길이, 런타임 호환 버전을 기록한다. `main`이나 가짜 해시를 완료된 manifest에 남기지 않는다. INT8 파일명만 보고 해당 CPU에서 더 빠르다고 단정하지 않는다.

## 4. 엔진과 배포 구조

기본 흐름은 `탐색 → 제한 내 원문 읽기 → 위치 보존 청크 → 후보 선택 → batch score → 중복 억제 → 출력`이다.

| 구성 요소 | 선택 / 책임 |
| --- | --- |
| CLI | `clap`, 옵션 검증과 명령 분기 |
| 파일 탐색 | `ignore` crate. 외부 `rg` 실행 파일은 필수 의존성으로 두지 않음 |
| 원문/청크 | Rust 표준 I/O, 원문 snapshot, line index, UTF-8 byte offset |
| 문자열 후보 검색 | 식별자 분할과 BM25 점수. 현재 질의에 필요한 통계만 메모리에 계산 |
| 모델 토큰화 | Hugging Face `tokenizers`; 초기 검증 후보는 offset API가 있는 0.22.2 |
| 추론 | `ort` + ONNX Runtime CPU, 검증된 버전 조합을 고정 |
| JSON/오류/다운로드 | `serde`, `serde_json`, 명시적 오류 타입, TLS 지원 HTTP client와 SHA-256 |
| 테스트 | Rust unit/integration, CLI 시나리오, 작은 자체 평가 corpus, 실제 모델 통합 테스트 |

`ignore`가 제공하는 필터를 사용하되 플래그 조합은 직접 테스트한다. [WalkBuilder 문서](https://docs.rs/ignore/latest/ignore/struct.WalkBuilder.html). tokenizer offset API는 해당 고정 버전에서 확인하고 Rust byte offset 의미를 테스트로 고정한다. [Encoding 0.22.2](https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.Encoding.html).

처음에는 Cargo package 하나에 library와 binary를 둔다. `main.rs`에 검색 로직을 몰지 않고 다음 모듈로 나눈다. 대규모 workspace는 필요 없다.

```text
src/
  main.rs, lib.rs, cli.rs, error.rs
  discovery.rs, source.rs, chunk.rs, lexical.rs, search.rs, output.rs
  model/{mod.rs, registry.rs, download.rs, tokenizer.rs, onnx.rs}
models/registry.toml
tests/{fixtures/, discovery.rs, chunking.rs, cli.rs, offline.rs, inference.rs}
eval/{corpus/, queries.jsonl, manifest.json, README.md}
examples/evaluate.rs
scripts/{evaluate.py, benchmark.sh, package.sh}
docs/{implementation-plan.md, implementation-goal.md, model-decision.md, progress.md}
artifacts/{evaluation/, benchmarks/, validation/}
```

위 경로 중 계획 문서 두 개를 제외한 나머지는 **앞으로 만들 산출물**이다. 실제 모델과 대형 측정 중간 파일은 Git에 넣지 않는다. 재현에 필요한 manifest와 작은 요약은 저장한다.

런타임은 초기 구현에서 `ort`의 동적 로딩 경로를 사용한다. `supergrep` 실행 파일 옆의 검증된 runtime 라이브러리를 찾고, 개발자용 명시 경로 설정을 지원한다. 임의의 현재 작업 디렉터리에서 라이브러리를 탐색하지 않는다. 배포물은 **실행 파일 + 플랫폼별 runtime + 고지 파일**이며 모델은 별도다. `--help`, `--lexical`, 모델 목록은 runtime 없이도 동작해야 한다. 정적 링크는 검증 후 별도 최적화다. [ort linking 문서](https://github.com/pykeio/ort/blob/main/docs/content/setup/linking.mdx).

## 5. 데이터와 검색 동작 계약

### 5.1 탐색과 자원 제한

- 기본적으로 `.gitignore`, `.ignore`, Git exclude/global 규칙과 숨김 제외를 적용한다. Git 밖에서도 일반 텍스트 검색이 가능하다. `.gitignore`는 Git 경계에 관한 `ignore`의 기본 동작을 유지하고 문서화한다.
- `--hidden`은 숨김 제외만, `--no-ignore`는 ignore 파일 규칙만 해제한다. 두 동작을 혼동하지 않는다. `.git` 내부는 v0.1에서 항상 제외한다.
- 심볼릭 링크를 따라가지 않고 일반 파일만 읽는다. 명시한 루트가 파일이면 숨김/ignore 필터보다 사용자의 지정을 우선하되 타입·인코딩·크기 제한은 적용한다. 명시적인 symlink 루트는 지원하지 않는다는 오류를 낸다.
- 파일당 기본 2MiB, 총 읽기 budget 64MiB, 생성 청크 50,000개로 제한한다. 옵션으로 조정할 수 있으나 전부 메모리에 담을 수 있다는 약속은 하지 않는다.
- 예측 가능한 경로 순서로 읽는다. 파일 metadata만 신뢰하지 않고 실제 read 길이도 제한한다. NUL이 있거나 유효하지 않은 UTF-8이면 파일을 제외한다. 잘못된 바이트를 replacement character로 바꿔 검색하지 않는다.
- 원문은 파일별로 한 번 저장하고 청크는 범위를 참조한다. 전체 corpus의 token ID/offset을 영구 보관하지 않고 파일 단위로 처리한다. 검색 중 원문 snapshot을 유지하며 결과 출력 시 다시 읽은 다른 내용으로 대체하지 않는다.
- 너무 큰 파일, 읽기 실패, 탐색 실패, 처리 한도는 원인별로 보고한다. ignore/숨김/바이너리 등 정책상 제외와 처리 실패를 구분한다. 탐색하지 않은 하위 파일 수를 정확히 센 것처럼 보고하지 않는다.
- 출력 파일을 검색 대상에 다시 넣는 상황과, 실행 중 파일 변경을 다룬다. 변경을 감지한 파일은 결과를 버리고 진단에 기록한다. 마지막 read 뒤의 변경까지 원자적으로 방지한다고 주장하지 않는다.

### 5.2 청크와 위치

`Source`는 경로, 원본 bytes/text, line starts, 읽은 시점의 metadata를 보관한다. `Chunk`는 source ID, `start_byte`/`end_byte`, `start_line`/`end_line`, `kind`, 선택적 context를 갖는다. byte 범위는 0-based half-open, 줄은 1-based inclusive다.

청크는 줄·문단 경계를 우선하되 길이를 보장해야 한다. 초기 목표는 본문 약 160 model tokens, overlap 최대 32 tokens, 최대 40줄이다. 문단이 길면 token offset으로 나누고 원문 byte 범위로 돌려놓는다. 공백뿐인 내용은 제외하고, 긴 한 줄도 여러 청크로 처리한다. UTF-8 경계, CRLF, 마지막 개행 유무를 보존한다.

최종 `(query, passage)` 입력은 기본 최대 256 tokens다. 질의 예산은 최대 64 tokens이며 초과 질의는 잘라 쓰지 않고 명확한 입력 오류를 낸다. tokenizer별 special tokens를 포함한 실제 pair 길이를 검사한다. 청크가 여전히 넘으면 원본 범위를 보존하며 다시 분할한다. 이 분할을 후보 선택 **이전**에 완료하여 `--candidates`가 실제 모델 입력 개수를 뜻하게 한다. 모델에 들어가지 않은 뒷부분을 평가했다고 표시하지 않는다.

모델 입력의 초기 passage는 본문만 사용한다. 경로는 lexical 점수와 출력에 사용한다. 경로/context를 모델에 붙이는 변경은 평가로 이점이 입증될 때만 추가하고 같은 입력 규약을 reference에도 적용한다.

`--lexical`은 모델 없이 동작하도록 같은 Source/Chunk 계약을 사용하되 줄·byte 기반 chunker를 쓴다. 양수 lexical 점수의 청크만 반환한다. 따라서 모델 품질 비교용 lexical 기준선은 별도로 **동일한 모델 청크 집합**에서 점수를 계산한다. 서로 다른 청크를 비교해 개선을 과장하지 않는다.

### 5.3 기본 검색과 `--deep`

- 기본 후보 budget `K=128`, 결과 `top-k=10`. 전체 청크 수 `N≤K`이면 모두 평가한다.
- `N>K`이면 lexical 점수·경로 신호·이웃 청크·분산 표본을 합쳐 최대 K개를 평가한다. 전체 파일을 읽으므로 fast도 전체 읽기 비용은 발생한다.
- `--deep`은 대상 범위 내 생성된 **모든 청크**를 batch로 평가한다. lexical 일치 여부로 제외하지 않으며 후보 수를 조용히 제한하지 않는다. 사용자 처리 한도/실패가 있으면 불완전 상태를 표시한다.
- `--deep`에서도 모델이 정답을 상위에 놓는다는 보장은 없다. 실제 정답 label과 평가한다.
- 모델이 없거나 추론이 실패하면 명시적인 오류를 낸다. 몰래 lexical 결과로 바꾸지 않는다. lexical만 원하는 경우 `--lexical`을 쓴다.

초기 후보 선택은 다음의 작고 결정적인 구현으로 시작한다. 비율은 개발 집합에서만 조정한다.

1. 질의·경로·본문에 Unicode 정규화/소문자화, 단어 분할, `snake_case`/`camelCase` 분할을 적용하되 원문은 바꾸지 않는다. 원래 식별자도 보존한다. 부정어는 제거하지 않는다.
2. 본문 BM25를 계산한다. 초기 `k1=1.2`, `b=0.75`. 경로 단어 일치는 별도 점수로 둔다. 전체 TF index 대신 현재 질의 단어의 tf/df와 chunk length만 유지한다.
3. K의 5/8은 본문 점수, 1/8은 경로 점수, 1/8은 상위 hit의 인접 청크, 1/8은 경로·파일 내 위치가 분산된 결정적 표본으로 채운다. 중복 제거 후 남는 자리는 양수 lexical 후보, 다음으로 분산 표본에서 채운다.
4. K가 작은 경우의 반올림과 tie-break를 명시하고, 최종 개수가 K를 넘지 않게 한다. 원시 경로/byte offset 순서를 마지막 tie-break로 쓴다. 무작위 선택에 의존하지 않는다.

분산 표본은 recall 보장이 아니다. 특히 한국어 질의와 영문 파일에서 lexical hit가 없고 `N>K`이면 `lexical_evidence=false`, `scoring_complete=false`를 내보내고 좁은 경로의 `--deep`을 안내한다. 낮은 recall을 숨기기 위해 평가 질의별 번역 사전이나 정답 경로를 코드에 넣지 않는다. v0.1의 확실한 다국어 평가 경로는 작은 범위의 전체 평가이며, 큰 범위 fast의 한계는 측정 결과로 명시한다.

### 5.4 추론과 결과 순위

- `Scorer::score_batch(query, passages)`는 입력 순서에 대응하는 유한한 score를 반환한다. production 구현은 실제 ONNX 모델이고 fake scorer는 테스트 전용이다.
- session은 한 번 로드한다. 초기 batch=4, intra-op threads=2, inter-op=1로 시작하되 S0 결과로 확정한다. tokenizer와 inference의 중첩 병렬화로 4코어를 과도하게 점유하지 않는다.
- sequence 길이별 batch 정렬을 쓰면 원래 chunk ID로 정확하게 복원한다. 모델별 `input_ids`, `attention_mask`, 선택적 `token_type_ids`와 output shape를 검증한다.
- **score는 관련성 확률이 아니다.** 원래 relevance logit을 사용하고 `score_kind`와 모델 ID/revision을 함께 기록한다. query나 모델이 다른 점수를 직접 비교하지 않는다. 확률처럼 보이는 `0.97`/`97%` 출력과 보정되지 않은 threshold는 v0.1에서 제공하지 않는다.
- 최종 순서는 score 내림차순, 경로, byte offset이다. 같은 파일의 중복 청크는 byte 범위 overlap/min(length)이 0.6 이상인 경우 낮은 점수 쪽을 억제한다. 같은 줄의 서로 다른 긴 청크를 줄 번호만으로 합치지 않는다.
- 기본 출력은 상위 후보를 보여준다. 관련 없는 질의에도 후보가 나올 수 있으며 “관련 결과 없음” 판정은 별도 calibration 없이 구현하지 않는다.

## 6. CLI와 출력 계약

```bash
supergrep model list
supergrep model download compact-multilingual
supergrep model verify compact-multilingual
supergrep doctor

supergrep "요청 실패 시 재시도 간격을 정하는 부분" .
supergrep "여러 번 로그인에 실패하면 접근을 제한하는 부분" ./src/auth --deep
supergrep "where are requests retried?" . --model tiny-en --top-k 5 --json
supergrep "retry delay" . --lexical
supergrep "cache invalidation" . --glob '*.rs' --candidates 256
```

위 명령은 구현할 인터페이스 예시다. 기본 모델 확정 후 README의 다운로드 예시를 실제 선택과 맞춘다.

초기 옵션은 `--deep`, `--lexical`, `--model <profile-or-directory>`, `--top-k`, `--candidates`, `--json`, `--hidden`, `--no-ignore`, 반복 가능한 `--glob`, `--max-file-size`, `--max-total-bytes`, `--max-chunks`, `--threads`다. `--deep`/`--lexical`은 상호 배타적이다. 의미 없는 옵션 조합은 parser에서 거절한다. 검색어가 subcommand/옵션과 충돌하면 `--`를 사용하는 escape도 테스트한다.

검색 명령은 네트워크 요청을 전혀 하지 않는다. 모델이 없으면 다운로드 명령을 안내한다. 다운로드 명령만 네트워크를 사용하며 사용자 코드나 질의를 보내지 않는다. 모델 cache는 플랫폼 cache 디렉터리 아래 `supergrep/models/<id>/<revision>/`에 둔다. 경로 override를 제공하고 다운로드는 임시 파일→hash 검증→원자적 완료로 처리한다. 실패·중단 파일을 완성된 모델로 인식하지 않으며 동시 설치도 처리한다. 로컬 모델 디렉터리도 지원하는 manifest/ONNX 계약을 통과해야 한다.

JSON은 stdout에 JSON object 하나를 출력한다. 진행률·경고는 stderr에 쓴다. 루트 object는 `schema_version`, `query`, `root`, `mode`, `model`, `results`, `stats`, `diagnostics`를 포함한다. `results`에는 path, byte/line 범위, kind, score, score_kind, snippet이 들어간다. 경로는 검색 루트 기준 상대 경로로 반환하고 root로 해석할 수 있게 한다. 일반 출력에서는 control character를 escape하고 JSON에서는 JSON 규약에 맞게 encode한다. Linux의 비 UTF-8 경로는 `path=null`, `path_display`와 별도 `path_bytes_base64`로 보존하여 손실된 문자열을 식별자로 쓰지 않는다.

`stats`는 최소한 읽은 파일/bytes, 생성 청크, 선택 후보, 실제 평가 청크, 중복 억제 개수, 제외·실패 원인별 개수, 모델 로드/탐색·읽기/청크·토큰화/후보 선택/추론/전체 시간을 담는다. 다음 상태를 분리한다.

| 필드 | 의미 |
| --- | --- |
| `scan_complete` | 적용한 파일 정책 범위의 순회를 처리 한도나 오류 없이 마쳤는가 |
| `scoring_complete` | 생성한 청크 전체를 모델로 평가했는가. 보통 fast에서는 false |
| `partial` | read/순회/자원 제한 등의 문제로 정상 수행 범위를 끝내지 못했는가 |
| `lexical_evidence` | 질의와 대상의 lexical 일치 근거가 있었는가 |

lexical 모드에서는 `model=null`, `scoring_complete=null`, `score_kind=bm25`를 사용한다. 모델 score와 lexical score를 같은 종류로 해석하지 않는다.

정상적인 fast 후보 축소 자체는 `partial`로 분류하지 않는다. oversized file/읽기 실패/총량 제한은 불완전 수행으로 보고한다. 모든 제외 사유가 사라졌다는 의미로 `scan_complete`를 쓰지 않는다.

종료 코드는 0=정상 결과 있음, 1=정상 실행했으나 출력 대상 없음, 2=입력/모델/치명적 실행 오류, 3=한도 또는 일부 읽기·추론 실패로 결과가 불완전함, 130=사용자 중단으로 정한다. 부분 결과가 있으면 3을 0/1보다 우선한다. 정상 fast의 `scoring_complete=false`는 오류가 아니다. Ctrl-C는 신속히 중단하고 완전 검색으로 표시하지 않는다.

## 7. 검증 데이터와 성공 기준

### 7.1 품질 데이터

모델 카드의 일반 문서 benchmark를 코드 검색 품질로 대신하지 않는다. 자체 검증은 고정한 작은 corpus와 공개 코드 snapshot을 함께 사용한다.

- **기능 fixtures:** 직접 작성한 코드·설정·문서. 파일명 단서가 없는 정답, 동일 단어의 오답, 중복, 부정문, 긴 줄을 포함한다.
- **품질 corpus:** 초기 40개 intent, 각 영어·한국어 질의 한 쌍으로 80개 질의. 코드 동작 20, 설정 8, 문서 8, 로그/데이터 4개 intent. 이 중 최소 12개는 영어에도 정답과 직접적인 단어 겹침이 적도록 한다.
- 개발 20 intent와 holdout 20 intent로 분리한다. 동일 intent의 양 언어, 번역 변형, 같은 정답 위치는 같은 split에 둔다. S0의 12 intent는 개발 split 일부이며 holdout을 쓰지 않는다.
- holdout의 최소 절반은 서로 다른 공개 프로젝트 두 개 이상에서 선택한 실제 코드·문서의 작은 snapshot으로 구성한다. source URL/commit/license와 원래 위치를 기록한다. 현재 대화에 없는 개인 저장소를 임의로 평가 corpus에 추가하지 않는다.
- 정답은 경로뿐 아니라 byte/line evidence span과 관련도 0/1/2로 기록한다. 함수 설명·설정·테스트가 동시에 관련 있으면 복수 정답을 허용한다. label의 근거를 남기고 결과를 본 뒤 유리하게 label을 수정하지 않는다.
- 별도 무관 질의 최소 10개로 오답 score 분포를 기록한다. v0.1의 threshold/무응답 기능 성공률로 계산하지 않는다.

질의별로 아래를 비교한다. 서로 다른 모델을 비교할 때 gold는 원본 span으로 유지하고 각 모델의 청크로 매핑한다. 개발용 `examples/evaluate.rs`가 library 엔진에서 동일 청크 집합의 lexical 점수, fast 후보 ID, fast/deep 결과를 JSONL로 내보내고 `scripts/evaluate.py`는 이를 집계한다. 별도 검색 알고리즘을 평가 스크립트에 다시 구현하지 않는다. CLI 전체 시간은 공개 CLI를 실행하는 benchmark에서 따로 측정한다.

| 평가 | 확인할 질문 |
| --- | --- |
| 동일 청크 집합의 lexical 기준선 | 모델 재정렬 없이도 찾을 수 있었는가 |
| fast 후보 집합의 gold Recall@K / Hit@K | 정답이 모델 입력에 들어왔는가 |
| fast + 모델의 Hit@5, MRR@10, nDCG@10 | 최종 결과가 개선되었는가 |
| deep + 같은 모델 | 후보 손실을 제거하면 정답을 찾는가 |
| fast/deep top-10 overlap | 같은 모델이 본 범위 차이로 얼마나 달라졌는가 |

Recall은 중복 청크 수 대신 중복 제거한 gold evidence 단위로 계산한다. 줄만 스치면 정답으로 세지 않도록 evidence byte 범위의 50% 이상을 담는 청크를 적중으로 정의한다. gold span은 한 검색 청크에 들어갈 수 있는 최소 근거로 작성한다. Hit@5는 질의별 정답이 상위 5개에 하나라도 있는 비율이며 deep 결과를 gold로 취급하지 않는다.

### 7.2 품질·성능 목표와 실패 처리

다음 수치는 **제품 목표이며 측정 결과가 아니다.** 기능 완료와 품질 적합을 구분해 보고한다.

| 항목 | 초기 통과 목표 |
| --- | --- |
| 위치·순회·JSON·오프라인 계약 | 해당 deterministic 테스트 모두 통과 |
| deep 영어 / 한국어 Hit@5 | holdout에서 각각 0.80 이상 |
| fast 영어 후보 Recall@128 | holdout에서 0.90 이상 |
| fast 영어 Hit@5 | 동일 청크 lexical보다 낮지 않고 deep 대비 손실 10%p 이내 |
| fast 한국어 | 영어 fast와 같은 수치를 지향하되 v0.1의 필수 통과 조건은 측정·한계 표시. 기준 미달이면 큰 범위 fast의 한국어 품질 보장을 하지 않음 |
| 재정렬의 효용 | 어휘 겹침이 적은 holdout 하위집합에서 deep가 lexical보다 Hit@5가 높음 |
| 기본 다국어 모델, 128 passages × 최대 256 tokens | 모델 로드 제외 추론 median 10초 이내 목표 |
| 기본 fast 전체 실행 | 고정 약 10MiB/1,000파일 corpus, 준비된 모델로 median 15초 이내 목표 |
| peak RSS | 위 corpus에서 전체 프로세스 1.5GiB 이하 목표 |

측정 전 CPU 부하·메모리 여유·저장 매체를 기록한다. 다른 프로세스를 종료하거나 swap 설정을 바꾸지 않는다. 현재 사용 중인 ARM 환경이므로 혼잡한 측정을 하드웨어 일반 성능으로 해석하지 않는다.

프로세스 시작부터 결과까지의 시간을 주 지표로 한다. 각 질의를 새 CLI 프로세스로 실행해 모델 로딩 비용을 포함한다. 모델 파일 준비 후 첫 실행 1회와 이후 서로 다른 질의 최소 20회 결과를 분리해 median/p95를 기록한다. kernel cache를 강제로 지우지 않으며 첫 실행을 엄밀한 cold-disk 측정이라고 부르지 않는다. 모델 로드 제외 batch benchmark는 별도로 기록한다.

`--deep`은 128/512/2,048 청크에서 총시간과 처리율을 측정한다. 자동 대규모 전체 검색을 기본값으로 켜지 않는다. 추가 stress 측정은 한도 진단과 bounded memory 검증용이며 품질 데이터에 섞지 않는다.

목표에 미달하면 개발 split에서 최대 두 차례 원인을 분리해 수정한다. 후보 누락, 모델 순위, 토큰화 손실, 초기화 비용을 각각 확인한다. K·길이를 줄여 빨라진 경우 같은 설정으로 품질도 재측정한다. 이후에도 미달하면 수치를 낮춰 성공으로 만들지 말고 실행 가능한 구현, 실패 근거, 선택 가능한 대안까지 남긴다. 한국어 deep 품질이나 ARM 사용성을 포기하는 범위 변경은 사용자의 결정이 필요하다. 검증되지 않은 상태를 goal 완료로 표시하지 않는다.

### 7.3 필수 실행 시나리오

1. Git 내부/외부의 텍스트 폴더에서 ignore, negation, hidden, glob, 직접 파일 지정의 의미가 유지된다.
2. NUL/잘못된 UTF-8/큰 파일/읽기 실패/빈 폴더/없는 루트/심볼릭 링크가 정해진 진단과 종료 코드를 낸다.
3. 한글·emoji·CRLF·끝 개행 없음·초장문 한 줄을 잘라도 결과 byte 범위가 실제 검색한 내용과 일치한다.
4. 파일 편집 후 재실행하면 바뀐 결과가 반영되고, 삭제·이름 변경 뒤 이전 결과가 남지 않는다.
5. K 미만 corpus의 fast와 deep는 같은 모델 score 순위를 낸다. K 초과에서는 deep가 모든 청크를 평가한다.
6. lexical overlap이 없는 한국어 질의에서도 작은 corpus의 deep가 실제 정답을 찾는지 평가하고, 큰 범위 fast의 불완전 coverage를 출력한다.
7. 진짜 모델을 batch 1과 4로 실행했을 때 score가 허용 오차 내이고 chunk/score 대응이 같다. NaN·shape 불일치는 오류다.
8. 준비된 bundle/모델에서 네트워크를 차단한 채 검색한다. 검색 경로의 socket/connect 시도가 없음을 OS 관찰 또는 동등한 독립 수단으로 확인한다. proxy 설정만으로 오프라인 증거를 대신하지 않는다.
9. 모델 파일 누락/손상/중단 다운로드/잘못된 runtime 버전에서 안내가 정확하고 silent fallback이 없다.
10. stdout JSON이 파싱 가능하며 경고는 stderr에만 나온다. 경로의 공백·개행·control character·비 UTF-8 bytes가 식별자를 손상시키지 않는다.
11. 자원 한도·Ctrl-C·출력 pipe 종료가 traceback/panic이나 “완료” 오표시 없이 처리된다.
12. 패키지를 새 임시 디렉터리에 풀고 프로젝트 작업 디렉터리 밖에서 검색한다. 실행 파일이 개발 머신의 우연한 library 경로에 의존하지 않는다.

## 8. 구현 순서와 각 단계의 완료 증거

| 단계 | 작업 범위 | 다음 단계로 넘어갈 증거 |
| --- | --- | --- |
| S0: 모델·런타임 실험 | Cargo 최소 골격, ARM runtime, 영어 및 다국어 실제 추론, 작은 개발 corpus, 프로필 선택 | 실제 score/배치/속도/RSS 기록, `docs/model-decision.md`, 확인된 revision/hash |
| S1: 파일 검색 기반 | discovery/source/chunk, 자원 제한, model 없는 lexical 경로, 위치/ignore 테스트 | 코드·문서·설정 fixtures에서 원본 위치/제외 사유/종료 코드 검증 |
| S2: 모델 수명 관리와 deep | registry/download/verify, tokenizer 입력 일치, batch scorer, deep 전체 평가 | 실제 모델의 일관된 배치 score, 전체 청크 수 대조, 손상/누락 모델 오류 |
| S3: fast 후보 검색 | BM25·경로·이웃·표본 후보, 결정적 순위, 중복 억제, coverage 통계 | 동일 corpus의 lexical/fast/deep 비교, K 제한과 zero-overlap 사례 검증 |
| S4: 사용자 CLI 완성 | 정식 CLI/JSON/doctor, stderr 진행 정보, 종료 코드, 오프라인 계약 | 7.3 시나리오의 관련 CLI 통합 증거, 도움말/사용 예시 |
| S5: 품질·성능 평가 | holdout 평가, ARM 전체 시간/RSS, 실패 원인 수정, 결과 보고 | 고정 manifest와 실제 raw results, benchmark 요약, 한계와 미충족 항목 |
| S6: 로컬 배포·문서 | release build, runtime 포함 묶음, README, 검증 스크립트, CI 설정 | fresh-directory smoke, 외부 경로 의존 없음, 완료 체크리스트와 남은 위험 |

의존 관계는 S0→S1→S2→S3→S4→S5→S6다. 초기부터 실제 모델을 연결하되 단계별 테스트는 좁게 실행한다. 로컬 실행을 하나의 주 작업으로 진행하고, 모델 여러 개의 추론이나 대규모 빌드를 이 4코어 환경에서 동시에 돌리지 않는다.

S0에서 runtime 로딩만 실패하면 ABI/라이브러리 경로를 먼저 확인하고 검증된 호환 버전 한 조합을 추가로 시험한다. 양자화 kernel 문제는 FP32 작은 모델로 분리 진단한다. 제품 runtime을 Python 서버로 대체하거나 외부 API를 붙여 요구 사항을 바꾸지 않는다. 막힌 원인과 독립적으로 가능한 작업을 구분하여 진행한다.

### 후속 구현 시 사용할 검증 명령의 형태

아래 명령은 관련 파일과 script를 구현한 뒤 실행한다. `cargo test`가 ignored 실제 모델 테스트까지 수행한다고 착각하지 않도록 별도 단계를 둔다.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo test --locked --test inference -- --ignored
cargo test --locked --test offline -- --ignored
cargo build --release --locked
cargo run --release --locked --example evaluate -- --split holdout --output artifacts/evaluation/holdout.jsonl
python3 scripts/evaluate.py --input artifacts/evaluation/holdout.jsonl
bash scripts/benchmark.sh target/release/supergrep
bash scripts/package.sh
```

script 옵션은 구현 시 실제 인터페이스와 동기화한다. 개발용 Python은 평가 JSON 집계 등에만 사용한다. 일반 unit/CLI 테스트는 네트워크·모델 다운로드 없이 실행되어야 한다. 실제 모델/오프라인 검증은 필요한 artifact를 먼저 준비하고 다운로드 실패를 skip 성공으로 바꾸지 않는다. 전체 필수 검증을 한 번 통과한 뒤에는 새 변경이나 미해결 실패가 없으면 반복 실행하지 않는다.

## 9. 완료 산출물과 인계 기준

- [ ] Rust release CLI가 ARM64에서 실제 로컬 모델로 텍스트를 검색한다.
- [ ] 원본 byte/line 위치, ignore와 파일 처리 한도, 오류·partial 동작이 검증되었다.
- [ ] fast와 deep가 구현되었고 candidate recall과 모델 품질을 구분한 평가가 있다.
- [ ] 모델 다운로드/검증과 준비 후 오프라인 검색을 실제로 확인했다.
- [ ] 기본 모델 revision/hash, tokenizer 규약, runtime 버전, 라이선스 출처가 고정되었다.
- [ ] 한국어/영어의 품질 표와 ARM의 실제 시간·메모리 표가 있고 미충족 목표를 숨기지 않는다.
- [ ] 로컬 배포 묶음을 다른 디렉터리에서 실행했으며 최종 사용자에게 Python이 필요 없다.
- [ ] README에 준비 방법, 검색 예시, 검색 범위와 한계, score 의미, 필요한 runtime/model, 재현 방법이 있다.
- [ ] 필수 테스트·평가 명령과 exit status가 `artifacts/validation/` 및 `docs/progress.md`에 기록되었다.

`docs/progress.md`는 각 단계의 완료/미완료, 실행 명령, 결과 artifact, 결정 변경, 다음 행동을 적는 체크포인트다. 생성된 파일이나 fake scorer의 성공만으로 실제 모델 기능이 완료되었다고 표시하지 않는다. 검증을 실행하지 못한 항목은 이유와 필요한 조건을 적는다.

작업명과 로컬 binary 이름은 `supergrep`으로 유지한다. 이미 [Etsy의 동명 프로젝트](https://github.com/etsy/supergrep)가 있으므로 공개 패키지명은 배포 직전에 별도 확인한다. v0.1 goal은 로컬 구현·검증·패키징까지이며 registry publish, 원격 push, 공개 release와 모델 재배포는 이 계획에 포함하지 않는다.
