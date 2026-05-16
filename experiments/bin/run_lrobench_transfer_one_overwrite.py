#!/usr/bin/env python
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query', required=True)
    args=ap.parse_args()
    q=args.query
    base=Path('experiments/08_lrobench/step8_cross_query_transfer')
    summary=json.loads((base/'per_query_inputs/lrobench_split_summary.json').read_text())
    items={item['group']: item for item in summary['items']}
    item=items[q]
    input_dir=Path(item['input_dir'])
    run_root=Path('experiments/runs/lrobench_cross_query')
    src=run_root/'per_query_independent'/q
    out=run_root/'cross_query_transfer'/q
    init_adapter=run_root/'average_source_adapter/adapter'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src/'cgsd_split_ids.json', out/'cgsd_split_ids.json')
    shutil.copy2(src/'cgsd_train_rows.jsonl', out/'cgsd_train_rows.jsonl')
    (out/'round_0').mkdir(exist_ok=True)
    for name in ['round_summary.json','selection_summary.json']:
        shutil.copy2(src/'round_0'/name, out/'round_0'/name)
    model='model/qwen3-0.6b'
    run([sys.executable,'scripts/cgsd_train_round.py','--output_dir',str(out),'--round_index','1','--model_path',model,'--data_path',str(input_dir/'data.jsonl'),'--split_ids_path',str(out/'cgsd_split_ids.json'),'--train_rows_path',str(out/'cgsd_train_rows.jsonl'),'--init_adapter_path',str(init_adapter),'--lora_r','1','--lora_target_modules','qv','--lora_layer_scope','all','--epochs','3','--lr','2e-4','--batch_size','4','--gradient_accumulation_steps','4','--eval_batch_size','8','--max_length','512','--cache_policy','overwrite'])
    run([sys.executable,'scripts/cgsd_predict.py','--output_dir',str(out),'--round_index','1','--model_path',model,'--data_path',str(input_dir/'data.jsonl'),'--split_ids_path',str(out/'cgsd_split_ids.json'),'--checkpoint_dir',str(out/'round_1/model'),'--cache_policy','overwrite'])
    run([sys.executable,'scripts/cgsd_calibrate.py','--output_dir',str(out),'--round_index','1','--temperature','15','--alpha','0.07','--previous_round_summary_path',str(out/'round_0/round_summary.json'),'--previous_selection_summary_path',str(out/'round_0/selection_summary.json'),'--train_rows_path',str(out/'cgsd_train_rows.jsonl'),'--cache_policy','overwrite'])
    run([sys.executable,'scripts/cgsd_finalize.py','--output_dir',str(out),'--round_index','1','--cache_policy','overwrite'])
    print(f'done {q}', flush=True)

if __name__ == '__main__':
    main()
