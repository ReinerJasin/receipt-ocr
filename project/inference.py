# command to install packages
# pip install -U "paddleocr[doc-parser]"
# pip install paddlepaddle==3.2.1

import time

from paddleocr import PaddleOCRVL
# from paddleocr import PaddleOCR

# Timer start
total_start = time.perf_counter()

# STEP 1: LOADING MODEL
t_model_start = time.perf_counter()

pipeline = PaddleOCRVL(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=True,
    device="cpu",
    precision="fp16"
)

t_model_end = time.perf_counter()
print(f"[TIMER] Model loading time: {t_model_end - t_model_start:.3f} seconds")

# STEP 2: DOING PREDICTION
t_pred_start = time.perf_counter()

output = pipeline.predict("/Users/reiner/Downloads/sample_receipt.jpeg")

t_pred_end = time.perf_counter()
print(f"[TIMER] Prediction time: {t_pred_end - t_pred_start:.3f} seconds")

# STEP 3: OUTPUT
for res in output:
    
    # PRINT RESULT
    t_print_start = time.perf_counter()
    
    res.print()                 # debug view
    
    t_print_end = time.perf_counter()
    print(f"[TIMER] Print result: {t_print_end - t_print_start:.3f} seconds")
    
    # SAVE JSON
    t_json_start = time.perf_counter()
    
    res.save_to_json(save_path="output")
    
    t_json_end = time.perf_counter()
    print(f"[TIMER] Save JSON: {t_json_end - t_json_start:.3f} seconds")
    
    # SAVE AS MARKDOWN
    t_md_start = time.perf_counter()
    
    res.save_to_markdown(save_path="output") 
    
    t_md_end = time.perf_counter()
    print(f"[TIMER] Save Markdown: {t_md_end - t_md_start:.3f} seconds")
    
total_end = time.perf_counter()
print(f"\n[TIMER] Total end-to-end time: {total_end - total_start:.3f} seconds")