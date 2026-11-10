# 라이브러리 import
import os
import time
import pickle
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from collections import Counter

print("임포트 완료")

# 경로 설정
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name in ("invest", "practice", "src") else Path.cwd()

fname = "[보도자료] 경제상황 평가(2026.10월)_FFF.pdf"
pdf_dir = Path("./data")
pdf_path = pdf_dir / fname

fpath = str(pdf_dir) + os.sep

print("현재 위치 : " , Path.cwd())
print("프로젝트 루트:", PROJECT_ROOT)
print("PDF 폴더     :", pdf_dir ,pdf_path.resolve())
print("PDF 존재 여부:", pdf_path.exists())

# pdf element 추출 함수 정의
def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path : 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """

    print("===", path, "===")

    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        #image_output_dir_path=path,       # 이미지 저장 경로
        extract_image_block_output_dir=path+"/figures"  # 이미지 저장 경로
    )

print("extract_pdf_elements 정의 완료")

# 요소 분류 함수 정의
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.

    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))     # TODO 5: 표는 어느 리스트로?
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))     # TODO 6: 텍스트는 어느 리스트로?

    return texts, tables

print("categorize_elements 정의 완료")

# 추출
start = time.time()

raw_pdf_elements = extract_pdf_elements(fpath, fname)

print(f"추출 완료 — {time.time() - start:.1f}초 소요")
print("추출된 전체 요소 수:", len(raw_pdf_elements))

texts, tables = categorize_elements(raw_pdf_elements)

print("텍스트 블록 수:", len(texts))
print("표 블록 수    :", len(tables))

# 추출된 요소 집계
type_counts = Counter(str(type(el)).split(".")[-1].strip("'>") for el in raw_pdf_elements)

for element_type, count in type_counts.most_common():
    print(f"{element_type:22s}: {count}개")

# 추출된 이미지 확인
figures_dir = Path(fpath) / "figures"
print(figures_dir)
if figures_dir.exists():
    image_files = sorted(
        f for f in figures_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"추출된 이미지: {len(image_files)}개\n")
    for f in image_files[:10]:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
else:
    print("✗ figures 폴더가 없습니다.")
    print("  → TODO 2, TODO 3 을 다시 확인하세요.")

# 텍스트 분할
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,       # 각 텍스트 블록의 최대 크기
    chunk_overlap=200,    # TODO 8: 청크 간 겹칠 글자 수
)

joined_texts = " ".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

print("원본 블록 수  :", len(texts))
print("분할 후 블록 수:", len(texts_2k_token))
print("전체 글자 수  :", len(joined_texts))

# 분할 결과 비교
print("=" * 60)
print("[원본 texts[0]] 앞 300자")
print("=" * 60)
print(texts[0][:300])
print()

print("=" * 60)
print("[분할 texts_2k_token[0]] 앞 300자")
print("=" * 60)
print(texts_2k_token[0][:300])

# 결과 저장
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "extracted.pkl", "wb") as f:
    pickle.dump(
        {"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token},
        f,
    )

print("저장 완료:", output_dir / "extracted.pkl")