#!/usr/bin/env bash
set -euo pipefail

sg_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sg_name="supergrep-v0.1.0-linux-aarch64"
sg_release_binary=${SUPERGREP_BINARY:-"$sg_root/target/release/supergrep"}
sg_ort_library=${SUPERGREP_ORT_LIB:-"$sg_root/.supergrep-dev-runtime/onnxruntime-linux-aarch64-1.20.0/lib/libonnxruntime.so.1.20.0"}
sg_ort_provider=${sg_ort_library%/libonnxruntime.so.1.20.0}/libonnxruntime_providers_shared.so
sg_output_parent=${1:-"$sg_root/artifacts/packages"}
if [[ "$sg_output_parent" != /* ]]; then
  sg_output_parent="$sg_root/$sg_output_parent"
fi
sg_bundle="$sg_output_parent/$sg_name"
sg_archive="$sg_output_parent/$sg_name.tar.gz"

if [[ ! -x "$sg_release_binary" ]]; then
  echo "missing executable release binary: $sg_release_binary (run cargo build --release --locked first)" >&2
  exit 1
fi
if [[ ! -f "$sg_ort_library" || ! -f "$sg_ort_provider" ]]; then
  echo "ONNX Runtime 1.20.0 bundle libraries are missing; set SUPERGREP_ORT_LIB to the official ARM64 library" >&2
  exit 1
fi
if [[ -e "$sg_bundle" || -e "$sg_archive" ]]; then
  echo "refusing to overwrite existing package output: $sg_bundle or $sg_archive" >&2
  exit 1
fi

mkdir -p "$sg_output_parent"
sg_output_parent=$(cd -- "$sg_output_parent" && pwd)
sg_bundle="$sg_output_parent/$sg_name"
sg_archive="$sg_output_parent/$sg_name.tar.gz"
if [[ -e "$sg_bundle" || -e "$sg_archive" ]]; then
  echo "refusing to overwrite existing package output: $sg_bundle or $sg_archive" >&2
  exit 1
fi
mkdir -p "$sg_bundle/bin/runtime" "$sg_bundle/licenses"
cp "$sg_release_binary" "$sg_bundle/bin/supergrep"
cp "$sg_ort_library" "$sg_bundle/bin/runtime/libonnxruntime.so.1.20.0"
cp "$sg_ort_provider" "$sg_bundle/bin/runtime/libonnxruntime_providers_shared.so"
cp "$sg_root/README.md" "$sg_bundle/README.md"
cp "$sg_root/LICENSE" "$sg_bundle/licenses/supergrep-Apache-2.0.txt"
cp "$sg_root/models/registry.toml" "$sg_bundle/models-registry.toml"

sg_ort_dir=$(dirname -- "$sg_ort_library")
sg_ort_root=$(cd -- "$sg_ort_dir/.." && pwd)
cp "$sg_ort_root/LICENSE" "$sg_bundle/licenses/onnxruntime-MIT.txt"
cp "$sg_ort_root/ThirdPartyNotices.txt" "$sg_bundle/licenses/onnxruntime-third-party-notices.txt"
(
  echo "supergrep v0.1.0 Linux ARM64 local bundle"
  echo "Required runtime: ONNX Runtime 1.20.0 (dynamic loading)"
  echo "Model files are intentionally not included; use model download explicitly."
  echo ""
  echo "Bundle contents and SHA-256:"
  cd "$sg_bundle"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "$sg_bundle/SHA256SUMS"

tar -czf "$sg_archive" -C "$sg_output_parent" "$sg_name"
echo "created bundle directory: $sg_bundle"
echo "created archive: $sg_archive"
