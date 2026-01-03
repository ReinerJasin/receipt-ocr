# command to install packages
# pip install -U "paddleocr[doc-parser]"
# pip install paddlepaddle==3.2.1

from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=True
)

output = pipeline.predict("/Users/reiner/Downloads/sample_receipt.jpeg")

for res in output:
    res.print()                 # debug view
    structured = res.json       # structured output
