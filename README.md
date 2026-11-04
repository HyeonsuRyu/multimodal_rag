```
git clone <저장소 주소>
cd multimodal-rag
```

# uv를 사용하는 경우
## 가상환경 생성 및 의존성 설치
```bash
uv sync
```
## 커널 등록
```bash
uv run python -m ipykernel install --user --name multimodal-rag
```
---

# uv를 사용하지 않는 경우
## 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate
```

## 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt
```

## 커널 등록
```bash
python -m ipykernel install --user --name multimodal-rag
```

---

# 공통
## .env 작성
```
cp .env.example .env                # 윈도우: copy .env.example .env
```
## .env 파일을 열어 실제 API 키를 입력