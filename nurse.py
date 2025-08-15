import os
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. API Key 설정
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY가 없습니다. Streamlit secrets 또는 환경변수에 추가하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# 2. CSV 불러오기 + 임베딩 처리
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # CSV 파일명 수정
    if "embedding" not in df.columns:
        with st.spinner("🔍 텍스트 임베딩 생성 중..."):
            df["embedding"] = df["content"].apply(lambda x: 
                client.embeddings.create(
                    input=x,
                    model="text-embedding-3-large"
                ).data[0].embedding
            )
    else:
        df["embedding"] = df["embedding"].apply(eval)
    return df

df = load_data()

# =========================
# 3. 유사도 검색 함수
# =========================
def find_similar_docs(query, top_k=3):
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    ).data[0].embedding

    similarities = cosine_similarity(
        [query_embedding],
        df["embedding"].to_list()
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return df.iloc[top_indices]

# =========================
# 4. Streamlit UI
# =========================
st.title("💬 Nurse Chatbot")
user_input = st.text_input("질문을 입력하세요")

if user_input:
    results = find_similar_docs(user_input, top_k=3)

    context = "\n".join(results["content"].to_list())

    with st.spinner("🤖 답변 생성 중..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 간호 도우미입니다. 제공된 문서 내용을 바탕으로 답변하세요."},
                {"role": "user", "content": f"다음 내용을 참고해서 질문에 답하세요.\n\n{context}\n\n질문: {user_input}"}
            ]
        )

    st.subheader("💡 답변")
    st.write(response.choices[0].message.content)

    st.subheader("📄 참고 문서")
    for idx, row in results.iterrows():
        st.markdown(f"- {row['content']}")


