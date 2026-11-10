import os, time, pickle
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI, AuthenticationError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import base64
from langchain_core.messages import HumanMessage
import tiktoken

# 텍스트 요약 함수 정의
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 요약본을 생성합니다.

    texts          : 텍스트 리스트
    tables         : 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부
    """

    # Prompt 한국어 버전
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

    # Prompt 영어 버전 (한국어 결과가 부정확할 때 교체)
    # prompt_text_eng = """You are an assistant tasked with summarizing tables and text for retrieval.
    # These summaries will be embedded and used to retrieve the raw text or table elements.
    # Give a concise summary of the table or text that is well optimized for retrieval.
    # Table or text: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트 요약을 활성화 여부에 따라 요약하거나 bypass
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # 표 데이터 요약
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})
    return text_summaries, table_summaries

# 이미지 base64 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 이미지 요약 함수 정의
def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "지시문"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])])
    
    return msg.content

# 폴더 내 이미지 요약 함수 정의
def generate_img_summaries(path):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성

    path: Unstructured에 의해 추출된 이미지 파일들의 경로
    """
    img_base64_list = []
    image_summaries = []

    prompt_kor = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
do not describe only about the meaning about the words in the imgae but the visual status about the image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""

    # 영어 요약이 필요하면 이쪽을 사용
    # prompt_eng = """You are an assistant tasked with summarizing images for retrieval.
    # These summaries will be embedded and used to retrieve the raw image.
    # Provide a concise summary of the image that is well optimized for retrieval."""

    for img_file in sorted(os.listdir(path)):
        # .jpg, .png, .jpeg 확장자만 처리합니다
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kor))

    return img_base64_list, image_summaries

if __name__ == "__main__":
    # API 키 로드 및 유효성 검사
    cwd = Path.cwd()
    PROJECT_ROOT = cwd.parent if cwd.name in ("invest", "practice", "src") else cwd

    print(PROJECT_ROOT/".env")
    load_dotenv(PROJECT_ROOT/".env")

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError

    print(f"{openai_api_key[:7]} ... {openai_api_key[-4:]}")

    try:
        print(OpenAI().models.list())
    except:
        print("유효하지 않은 OpenAI API Key입니다.")
        raise AuthenticationError

    # pdf 추출 데이터 로드
    pkl_path = PROJECT_ROOT / "data" / "processed" / "extracted.pkl"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    texts, tables, texts_2k_token = data["texts"], data["tables"], data["texts_2k_token"]

    print(f"texts 개수: {len(texts)}\ntables 개수: {len(tables)}\ntexts_2k_token: {len(texts_2k_token)}")

    figures_dir = PROJECT_ROOT / "data" / "figures"
    print(str(figures_dir)+"이 존재"+("합" if figures_dir.exists() else "하지 않습")+"니다.")

    # 텍스트, 표 요약 실행
    start = time.time()
    text_summaries , table_summaries = generate_text_summaries(texts_2k_token, tables,summarize_texts=True)
    end = time.time()
    print(f"요약 완료 => {end - start :.1f} 초 걸림")
    print(f"텍스트 요약 수 =>" , len(text_summaries))
    print(f"표 요약 수 =>", len(table_summaries))

    # 요약 결과 확인
    print("-" *60)
    print("원본임 - 앞 300자")
    print(texts_2k_token[0][:300])
    print()
    print("-" *60)
    print("요약본임")
    print(text_summaries[0])
    print()
    origin = len(texts_2k_token[0])
    summ = len(text_summaries[0])
    print(f"압축률 : {origin}자 ->{summ} 자 ({summ/origin *100:.1f}%) ")

    # table_summaries가 비어있는 경우 원인 파악
    if table_summaries:
        print(f"표 요약 => {len(table_summaries)} 건 생성됨 \n")
        print("[표 요약 예시]")
        print(table_summaries[0])
    else:
        print("표 요약이 비어 있습니다. " \
        "      원인은 1. strategy = 'hi_res' 2. 진짜 표인지, 그림으로 그랜 표인지 확인")

    for i, summary in enumerate(table_summaries[:3]):
        print("="*60)
        print(f"[표 요약 {i+1}]")
        print("="*60)
        print(summary)
        print()

    for i, (table, summary) in enumerate(zip(tables[:2], table_summaries)):
        print("="*60)
        print(f"[표 {i+1} 원본 (앞 300자)]")
        print("="*60)
        print(table[:300])
        print()
        print("="*60)
        print(f"[표 {i+1} 요약]")
        print("="*60)
        print(summary)
        print()


    image_files = sorted(
        f for f in figures_dir.iterdir() if f.suffix.lower() in (".jpg", "jpeg", ".png")
    )

    sample_b64 = encode_image(image_files[0])
    print("원본파일 =>", image_files[0].name)
    print("문자열 길이 :", len(sample_b64), "자")
    print("앞 100자 :", sample_b64[:100])



    test_prompt = """You are an assistant tasked with summarizing images for retrieval.
    These summaries will be embedded and used to retrieve the raw image.
    Provide a concise summary of the image that is well optimized for retrieval.
    The summary should be written in Korean (Hangul)."""

    print(image_summarize(sample_b64, test_prompt))


    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(str(figures_dir))
    end = time.time()
    print(f"이미지 요약 완료 — {end - start:.1f}초 소요")
    print("요약된 이미지 수:", len(image_summaries))

    # 요약 품질 검수
    from IPython.display import Image, display
    for i , (img_file, summary) in enumerate(zip(image_files[:2], image_summaries[:2])):
        print("="*60)
        print(f"[{i}] {img_file.name}")
        print("="*60)
        display(Image(filename=str(img_file), width=450))
        print("요약:", summary)
        print()

    # 토큰 사용량 추정
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    input_tokens = sum(len(enc.encode(t)) for t in texts_2k_token)
    output_tokens = sum(len(enc.encode(s)) for s in text_summaries)

    print("텍스트 요약 토큰 사용량 (추정)")
    print(f"  입력 : {input_tokens:,} 토큰")
    print(f"  출력 : {output_tokens:,} 토큰")
    print(f"  합계 : {input_tokens + output_tokens:,} 토큰")
    print()
    print("요약한 이미지 수:", len(image_summaries), "장")
    print()
    print("※ 실제 비용 = 토큰 수 x 단가. 단가는 OpenAI 공식 가격 페이지에서 확인하세요.")
    print("※ 이미지 토큰은 해상도에 따라 달라지므로 위 수치에 포함되지 않았습니다.")

    # 저장
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summaries.pkl", "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list,
            },
            f,
        )

    print("저장 완료:", output_dir / "summaries.pkl")
    print("7교시에서 extracted.pkl 과 summaries.pkl 을 함께 불러옵니다.")