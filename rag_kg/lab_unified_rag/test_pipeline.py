# test_pipeline.py
from lab_core import build_reasoning_context, retrieve_evidence
import json

# Load 1 case thật
with open('data/cases/all_results.jsonl', encoding='utf-8') as f:
    case = json.loads(f.readline())

ctx = build_reasoning_context(case, 0)
print('Case:', ctx['case_id'])
print('Abnormal tests:', ctx['abnormal_tests'])
print('Conditions:', ctx['conditions'])
print()

evidence = retrieve_evidence(ctx)
print(f'Evidence retrieved: {len(evidence)}')
print()
for i, ev in enumerate(evidence[:5], 1):
    print(f'{i}. [{ev["type"]}] {ev["source"]} p.{ev["page"]} | score={ev.get("final_score", ev.get("score", 0)):.3f}')
    print(f'   {ev["text"][:150]}')
    print()