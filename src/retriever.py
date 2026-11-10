import os, re, time, uuid, pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# langchain 0.3, 1.x 버전의 패키지 구조 변경에 따른 예외처리
try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except:
    print("MultiVectorRetriever 클래스가 필요합니다.")
try:
    from langchain_core.stores import InMemoryStore
except:
    print("InMemoryStore 클래스가 필요합니다.")

# 요약과 원본 연결하여 저장
def add_documents(retriever, doc_summaries, doc_contents, id_key):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [Document(page_content=s, metadata={id_key:doc_ids[i]})
                    for i, s in enumerate(doc_summaries)]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

# 검색 결과를 텍스트와 이미지로 분리
def split_image_text_types(docs):
    b64_images = list()
    texts_out = list()

    for doc in docs:
        doc.page_content if isinstance(doc, Document) else doc
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$",doc):
            b64_images.append(doc)
        else:
            texts_out.append(doc)

    return {"images" : b64_images, "texts": texts_out}

# 최종 멀티모달 체인
def img_prompt_func(data_dict):
    messages = list()

    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })

        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages.append({
            "type": "text",
            "text": ("자료를 근거로 질문에 한국어로 답하세요\n"
                    "그래프로 나타난 수치는 이미지를 분석하여 대략적인 값을 요약에 포함하세요\n"
                    "자료에 없는 내용은 모른다고 답하세요.\n\n"
                    f"질문: {data_dict['question']}\n\n"
                    f"참고 텍스트 :\n {formatted_texts}"),
        })
        return [HumanMessage(content=messages)]

if __name__ == "__main__":
    # 프로젝트 경로, 환경 변수 로드
    PROJECT_ROOT =  next(
        ( p for p in [Path.cwd(), *Path.cwd().parents] if(p / ".env").exists()), Path.cwd(),
    )
    print(PROJECT_ROOT)

    load_dotenv(PROJECT_ROOT/".env")
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        if not openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY 값이 비어 있습니다. .env파일 확인해 주세요")
    except KeyError:
        raise ValueError("OPENAI_API_KEY 키가 없습니다. .env파일을 확인해 주세요")

    processed = PROJECT_ROOT / "data" / "processed"

    # 결과 복원
    with open(processed/"extracted.pkl", "rb") as f:
        doc = pickle.load(f)
        texts = doc["texts"]
        tables = doc["tables"]
        texts_2k_token = doc["texts_2k_token"]
        
    with open(processed/"summaries.pkl", "rb") as f:
        doc = pickle.load(f)
        text_summaries = doc["text_summaries"]
        table_summaries = doc["table_summaries"]
        image_summaries = doc["image_summaries"]
        img_base64_list = doc["img_base64_list"]

    print("복원 완료")
    print(f" 텍스트 원본 {len(texts_2k_token):3d}건  / 요약{len(text_summaries):3d} 건")
    print(f" 테이블 원본 {len(tables):3d}건  / 요약{len(table_summaries):3d} 건")
    print(f" 이미지 원본 {len(img_base64_list):3d}건  / 요약{len(image_summaries):3d} 건")

    vectorstore = Chroma(
        collection_name="multimodal_rag" , embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore= vectorstore, docstore= store, id_key = id_key
    )

    if text_summaries:
        add_documents(retriever, text_summaries, texts_2k_token, id_key)
    if table_summaries:
        add_documents(retriever, table_summaries, tables, id_key)
    if image_summaries:
        add_documents(retriever, image_summaries, img_base64_list, id_key)

    print("색인 완료")
    print("  벡터 DB에 저장된 요약 수:", len(vectorstore.get()["ids"]))

    # 검색 테스트
    query = "24년말 대비 26년 1분기의 누적 평가이익을 알려줘"

    found = retriever.invoke(query)
    print(len(found))

    for i, doc in enumerate(found):
        content = doc.page_content if hasattr(doc, "page_content") else doc
        is_image = bool(re.match("^[A-Za-z0-9+/]{100,}={0,2}$", content)) # [A-Za-z0-9+/]가 100개 이상 이어지며, =가 0~2개까지인 경우를 검색(base64 포맷)
        kind = "이미지(Base64)" if is_image else "텍스트/표"
        print(f"[{i}] {kind} — 길이 {len(content):,}자")
        print("   ", content[:120].replace("\n", " "), "...")
        print()

    split_data = split_image_text_types(found)
    print("이미지:" , len(split_data["images"]), "건")
    print("표:" , len(split_data["texts"]), "건")

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        } | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    print("최종 체인 조립 완료")

    # 질의 실행
    start = time.time()
    answer = chain.invoke(query)
    print(f"소요시간: {time.time() - start:.1f}s",
        f"질문(Q): {query}",
        ("="*60),
        f"답변: {answer}", sep='\n')

    # before / after 비교
    vectorstore_text = Chroma(
        collection_name="text_only_rag" , embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    prompt_text = ChatPromptTemplate.from_template(
        "자료를 근거로 질문에 한국어로 답하세요\n"
        "그래프로 나타난 수치는 이미지를 분석하여 대략적인 값을 요약에 포함하세요\n"
        "자료에 없는 내용은 모른다고 답하세요.\n\n"
        "질문: {question}\n\n"
        "참고 텍스트 :\n {context}")

    chain_text = (
        {
            "context": vectorstore_text.as_retriever(search_kwargs={"k": 4}) | RunnableLambda(lambda docs: '\n'.join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        } | prompt_text
        | model
        | StrOutputParser()
    )

    print("비교용 텍스트 전용 RAG 준비 완료")

    print("="*60)
    print("질문", query)

    before = chain_text.invoke(query)
    print("======== before ========")
    print(before)

    after = chain.invoke(query)
    print("======== after ========")
    print(after)

    print("이 출력 전체를 캡처해 README.md에 넣으세요.")