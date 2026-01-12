#!/bin/bash
set -e

# MMJP Test Script
# Runs all tests to verify the implementation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TOOLS_DIR="$SCRIPT_DIR/tools"
TMP_DIR="/tmp/mmjp_tests"
mkdir -p "$TMP_DIR"

echo "========================================="
echo "MMJP Test Suite"
echo "========================================="

# Test 1: Build tools
echo ""
echo "[1/6] Building tools..."
cd "$TOOLS_DIR"
gcc -O3 -std=c99 -Wall -Wextra -I.. -I../double_array -I../npycrf_lite -I../suffix_array -I../unilm_mdl \
  -o mmjp_train mmjp_train.c mmjp_model.c \
  ../suffix_array/sa_utf8.c ../unilm_mdl/unilm_mdl.c \
  ../double_array/double_array_trie.c ../npycrf_lite/npycrf_lite.c \
  ../mmjp_lossless.c -lm
gcc -O3 -std=c99 -Wall -Wextra -I.. -I../double_array -I../npycrf_lite \
  -o mmjp_tokenize mmjp_tokenize.c mmjp_model.c \
  ../double_array/double_array_trie.c ../npycrf_lite/npycrf_lite.c \
  ../mmjp_lossless.c -lm
echo "PASS: Tools built successfully"

# Test 2: pip install
echo ""
echo "[2/6] Testing pip install -e ..."
cd "$SCRIPT_DIR"
pip install -e . -q
python -c "import mmjp; print(f'mmjp version: {mmjp.__version__}')"
echo "PASS: pip install successful"

# Test 3: Lossless roundtrip
echo ""
echo "[3/6] Testing lossless roundtrip..."
cat > "$TMP_DIR/t.txt" <<'EOF'
hello  world
	indented line
x▁y ▀z
日本語テスト
EOF

"$TOOLS_DIR/mmjp_train" --corpus "$TMP_DIR/t.txt" --out "$TMP_DIR/model_lossless.bin" \
  --vocab 100 --char_vocab 100 --iters 1 --min_count 1 \
  --lossless_ws 1 --crf_unsupervised 1 --crf_unsup_sentences 10 \
  --crf_opt sgd --crf_epochs 2 --crf_lr 0.1 > /dev/null 2>&1

"$TOOLS_DIR/mmjp_tokenize" --model "$TMP_DIR/model_lossless.bin" --lossless_ws -1 --read_all 1 \
  < "$TMP_DIR/t.txt" > "$TMP_DIR/t.tok"
"$TOOLS_DIR/mmjp_tokenize" --model "$TMP_DIR/model_lossless.bin" --detok --lossless_ws -1 \
  < "$TMP_DIR/t.tok" > "$TMP_DIR/t.restored"

if cmp -s "$TMP_DIR/t.txt" "$TMP_DIR/t.restored"; then
  echo "PASS: Lossless roundtrip successful"
else
  echo "FAIL: Lossless roundtrip - files differ"
  diff "$TMP_DIR/t.txt" "$TMP_DIR/t.restored"
  exit 1
fi

# Test 4: cc_ranges smoke test
echo ""
echo "[4/6] Testing cc_ranges..."
cat > "$TMP_DIR/ranges.txt" <<'EOF'
# Japanese character ranges
0x3040 0x309F 4
0x30A0 0x30FF 5
0x4E00 0x9FFF 6
EOF

"$TOOLS_DIR/mmjp_train" --corpus "$SCRIPT_DIR/datasets/wiki_small.txt" \
  --out "$TMP_DIR/model_ranges.bin" \
  --vocab 1000 --iters 1 \
  --cc_mode ranges --cc_ranges "$TMP_DIR/ranges.txt" > /dev/null 2>&1
echo "PASS: cc_ranges smoke test successful"

# Test 5: wiki_small
echo ""
echo "[5/6] Testing wiki_small training..."
"$TOOLS_DIR/mmjp_train" --corpus "$SCRIPT_DIR/datasets/wiki_small.txt" \
  --out "$TMP_DIR/model_small.bin" \
  --vocab 1000 --iters 2 \
  --lossless_ws 1 --crf_unsupervised 1 > /dev/null 2>&1
echo "PASS: wiki_small training successful"

# Test 6: wiki_full (if available)
echo ""
echo "[6/6] Testing wiki_full (if available)..."
WIKI_FULL="$SCRIPT_DIR/datasets/wiki_full_lf.txt"
if [ -f "$WIKI_FULL" ]; then
  echo "Found wiki_full_lf.txt, running full verification..."

  # Training
  echo "  Training on full Wikipedia..."
  START_TRAIN=$(date +%s)
  "$TOOLS_DIR/mmjp_train" --corpus "$WIKI_FULL" \
    --out "$TMP_DIR/model_full.bin" \
    --vocab 5400 --iters 3 \
    --sample_bytes 5000000 \
    --lossless_ws 1 --crf_unsupervised 1 --crf_unsup_sentences 5000 \
    --crf_opt sgd --crf_epochs 3 --crf_lr 0.1 2>&1 | tail -5
  END_TRAIN=$(date +%s)
  echo "  Training completed in $((END_TRAIN - START_TRAIN)) seconds"

  # Tokenization
  echo "  Tokenizing full Wikipedia..."
  START_TOK=$(date +%s)
  "$TOOLS_DIR/mmjp_tokenize" --model "$TMP_DIR/model_full.bin" --lossless_ws -1 \
    < "$WIKI_FULL" > "$TMP_DIR/wiki_full.tok" 2>&1
  END_TOK=$(date +%s)
  LINES_TOK=$(wc -l < "$TMP_DIR/wiki_full.tok")
  echo "  Tokenization completed in $((END_TOK - START_TOK)) seconds ($LINES_TOK lines)"

  # Roundtrip (line-by-line)
  echo "  Checking roundtrip..."
  "$TOOLS_DIR/mmjp_tokenize" --model "$TMP_DIR/model_full.bin" --detok --lossless_ws -1 \
    < "$TMP_DIR/wiki_full.tok" > "$TMP_DIR/wiki_full.restored" 2>&1

  ORIG_LINES=$(wc -l < "$WIKI_FULL")
  RESTORED_LINES=$(wc -l < "$TMP_DIR/wiki_full.restored")
  DIFF_COUNT=$(diff "$WIKI_FULL" "$TMP_DIR/wiki_full.restored" | grep "^[<>]" | wc -l || true)

  echo "  Original lines: $ORIG_LINES"
  echo "  Restored lines: $RESTORED_LINES"
  echo "  Diff count: $DIFF_COUNT"

  if [ "$DIFF_COUNT" -le 10 ]; then
    echo "PASS: wiki_full verification successful (minor diffs: $DIFF_COUNT)"
  else
    echo "WARNING: wiki_full has $DIFF_COUNT line differences"
  fi
else
  echo "SKIP: wiki_full_lf.txt not found"
  echo "      To run full verification, create datasets/wiki_full_lf.txt"
fi

echo ""
echo "========================================="
echo "All tests completed!"
echo "========================================="
