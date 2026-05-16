#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cgsd_env.sh"

"$SCRIPT_DIR/cgsd_round0_select.sh"

ROUND=1 "$SCRIPT_DIR/cgsd_train_round.sh"
ROUND=1 "$SCRIPT_DIR/cgsd_eval_round.sh"

ROUND=1 BUDGET=150 "$SCRIPT_DIR/cgsd_select_round.sh"
ROUND=2 "$SCRIPT_DIR/cgsd_train_round.sh"
ROUND=2 "$SCRIPT_DIR/cgsd_eval_round.sh"

ROUND=2 BUDGET=100 "$SCRIPT_DIR/cgsd_select_round.sh"
ROUND=3 "$SCRIPT_DIR/cgsd_train_round.sh"
ROUND=3 "$SCRIPT_DIR/cgsd_eval_round.sh"

ROUND=3 "$SCRIPT_DIR/cgsd_finalize.sh"

printf 'experiment 1 default 3-round run complete: %s\n' "$OUT"
