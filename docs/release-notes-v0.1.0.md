# supergrep v0.1.0

Linux ARM64 CLI bundle with a local ONNX cross-encoder, fast/deep search, and
exact file byte and line ranges. The bundle includes ONNX Runtime 1.20.0 but
does not include model weights; download the pinned model explicitly with
`supergrep model download compact-multilingual` after unpacking.

On the measured four-core ARM host and synthetic 10 MiB/1,000-file workload,
the 21-query median wall time was 21.58 seconds, inference median was 11.67
seconds, and sampled RSS p95 was 736,336 KiB. The original 15-second wall and
10-second inference goals were not met. On a separate 280-chunk sensitivity
corpus, Korean fast Hit@5 was 0.50 versus deep Hit@5 of 0.85. See the README
for the evaluation setup and limitations.
