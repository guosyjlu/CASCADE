#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${1:-./rag_data}"
mkdir -p "$SAVE_PATH"

CURL_OPTS=( -L --fail --retry 5 --retry-delay 2 -C - )
if [[ -n "${HF_TOKEN:-}" ]]; then
  CURL_OPTS+=( -H "Authorization: Bearer $HF_TOKEN" )
fi

download() {
  local url="$1"
  local out="$2"
  echo ">>> downloading: $url"
  curl "${CURL_OPTS[@]}" -o "$out" "$url"
  if [[ ! -s "$out" ]]; then
    echo "ERROR: $out is empty. Check network/token." >&2
    exit 1
  fi
}

echo "[1/4] Download Wikipedia 2018 corpus (wiki-18.jsonl.gz)..."
download "https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz" \
         "$SAVE_PATH/wiki-18.jsonl.gz"

echo "[2/4] Download E5 slides part_aa / part_ab (very big)..."
download "https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa" \
         "$SAVE_PATH/part_aa"
download "https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab" \
         "$SAVE_PATH/part_ab"

echo "[3/4] Merge E5 slides e5_Flat.index ..."
cat "$SAVE_PATH/part_aa" "$SAVE_PATH/part_ab" > "$SAVE_PATH/e5_Flat.index"

echo "[4/4] Unzip corpus as wiki-18.jsonl ..."
gzip -d "$SAVE_PATH/wiki-18.jsonl.gz"

echo "✅ Complete"
echo "  - Corpus:  $SAVE_PATH/wiki-18.jsonl"
echo "  - Index:   $SAVE_PATH/e5_Flat.index"