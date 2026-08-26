#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=== f5-voice-loop 環境準備 ==="

COSY_VENV="${COSYVOICE_VENV:-$HOME/CosyVoice/.venv}"

if [[ ! -f "$COSY_VENV/bin/python" ]]; then
  echo "錯誤：找不到 $COSY_VENV/bin/python，請確認 CosyVoice 環境已建立。"
  exit 1
fi

echo "確認 F5-TTS 相依套件..."
uv pip install --python "$COSY_VENV/bin/python" ema-pytorch cached_path pydub vocos rjieba pypinyin jieba torchdiffeq wandb accelerate datasets click --no-deps
uv pip install --python "$COSY_VENV/bin/python" --no-deps f5-tts

echo "執行 selfcheck..."
"$COSY_VENV/bin/python" voice_loop.py --selfcheck

echo "=== 設定完成！可直接執行 $COSY_VENV/bin/python voice_loop.py ==="
