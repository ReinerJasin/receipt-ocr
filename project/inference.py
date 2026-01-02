# command to install packages
# pip install -U "paddleocr[doc-parser]"
# pip install paddlepaddle==3.2.1

from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    device="cpu"
)