# QA evaluation runner

Folder này chứa phần đánh giá riêng:

- `test_50_cbc.py`: bộ 50 câu CBC.
- `test_50_biochem.py`: bộ 50 câu sinh hóa.
- `context/question_context_100.jsonl`: context KG/book evidence lấy từ `rag_kg/lab_unified_rag/data/outputs/question_context_100.jsonl`.
- `run_qa_eval.py`: sinh answer từ context sách/KG.
- `run_kg_judge.py`: dùng LLM-as-a-judge chấm 3 tiêu chí KG.

## Step 1: Gen Answer

```powershell
python evaluate/run_qa_eval.py --mode all
```

Chạy thử 1 câu riêng, không ghi vào file 100 câu chính:

```powershell
python evaluate/run_qa_eval.py --mode all --ids CBC001 --output evaluate/smoke_qa_eval.jsonl --csv evaluate/smoke_qa_eval.csv --failed-csv evaluate/smoke_qa_eval_failed.csv --review-csv evaluate/smoke_kg_review.csv --public-jsonl evaluate/smoke_qa_eval_public.jsonl
```

Hoặc chạy nhanh câu đầu tiên sau khi filter:

```powershell
python evaluate/run_qa_eval.py --mode all --limit 1 --output evaluate/smoke_qa_eval.jsonl --csv evaluate/smoke_qa_eval.csv --failed-csv evaluate/smoke_qa_eval_failed.csv --review-csv evaluate/smoke_kg_review.csv --public-jsonl evaluate/smoke_qa_eval_public.jsonl
```

Resume khi bị dừng giữa chừng:

```powershell
python evaluate/run_qa_eval.py --mode resume
```

Chạy lại câu gen answer lỗi:

```powershell
python evaluate/run_qa_eval.py --mode failed
```

Output:

```text
evaluate/qa_100_eval_results.jsonl
evaluate/qa_100_eval_results.csv
evaluate/qa_100_eval_failed.csv
evaluate/kg_human_review_template.csv
```

Public answer sẽ có citation trong câu trả lời và block `References` gồm source/page/quote ngắn. Các field nội bộ như KG node/path/evidence id không xuất ra public answer.

## Step 2: LLM-as-a-Judge

Sau khi `qa_100_eval_results.jsonl` đã có answer:

```powershell
python evaluate/run_kg_judge.py --mode all
```

Khuyến nghị dùng RAG judge 4 tiêu chí thay cho KG judge nếu muốn chấm như một chatbot RAG:

```powershell
python evaluate/run_rag_judge.py --mode all
```

RAG judge chấm:

- `context_relevance_score`: evidence có liên quan câu hỏi/case không.
- `source_attribution_score`: có source/page/quote đúng từ sách không.
- `faithfulness_score`: answer có bám evidence hay bịa thêm không.
- `medical_safety_score`: answer có an toàn y khoa không.

Output:

```text
evaluate/rag_llm_judge_results.jsonl
evaluate/rag_llm_judge_results.csv
evaluate/rag_llm_judge_failed.csv
```

Runner judge đọc:

```text
evaluate/qa_100_eval_results.jsonl
evaluate/context/question_context_100.jsonl
```

và chấm 3 tiêu chí theo thang `0`, `0.5`, `1`:

- `node_score`: Node đúng trọng tâm.
- `path_score`: Path reasoning hợp lý.
- `evidence_score`: Truy vết evidence/source.

Mỗi tiêu chí có cột reason riêng:

- `node_reason`
- `path_reason`
- `evidence_reason`

Các reason này dùng để giải thích vì sao điểm là `0.5` hoặc `0`; nếu đạt `1` thì vẫn ghi lý do ngắn gọn.

Output judge:

```text
evaluate/kg_llm_judge_results.jsonl
evaluate/kg_llm_judge_results.csv
evaluate/kg_llm_judge_failed.csv
```

CSV judge cũng có cột trống cho human review cuối:

- `human_node_score`
- `human_path_score`
- `human_evidence_score`
- `human_total_kg_3`
- `human_note`

Resume khi judge bị dừng:

```powershell
python evaluate/run_kg_judge.py --mode resume
```

Chỉ chạy lại case judge lỗi:

```powershell
python evaluate/run_kg_judge.py --mode failed
```
