#!/usr/bin/env bash
# Exports Llama 3.2 1B-Instruct with 8da4w (8-bit dynamic activation, 4-bit weight)
# groupwise quantization to a Meta ExecuTorch .pte binary targeting Arm XNNPACK.
set -e

python3 -m executorch.examples.models.llama.export_llama \
  --checkpoint "Llama-3.2-1B-Instruct/consolidated.00.pth" \
  --params "Llama-3.2-1B-Instruct/params.json" \
  -kv \
  --use_sdpa_with_kv_cache \
  -X \
  -qmode 8da4w \
  --group_size 128 \
  -d fp32 \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name="../models/llama3_2_1b_q4.pte"

echo "Exported ../models/llama3_2_1b_q4.pte (~782 MB) for Raspberry Pi Arm CPU."
