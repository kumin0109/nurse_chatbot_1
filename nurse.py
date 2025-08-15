import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. OpenAI 클라이언트 초기화
# =========================
api_key = st.secrets["OPENAI_API_KEY"]  # Streamlit secrets에서 API 키 읽기
client = OpenAI(api_key=api_key)

# =========================
# 2. CSV 데이터 로드 + 임베딩 변환
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_2_with_embeddings.csv")
    # 문자열로 저장된 리스트를 다시 리스트로 변환
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    # 카테고리 정보 합치기
    df["Etc"] = df[["Category1", "Category2", "Department"]].fillna("").astype(str).agg(";".join, axis=1)
    return df

df = load_data()

# =========================
# 3. 텍스트 → 임베딩
# =========================
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# =========================
# 4. 유사도 계산
# =========================
def find_most_similar(user_embedding, df):
    all_embeddings = np.array(df["Embedding"].to_list())
    sims = cosine_similarity([user_embedding], all_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return df.iloc[best_idx], sims[best_idx]

# =========================
# 5. GPT로 채점
# =========================
def grade_answer(user_answer, correct_answer):
    prompt = f"""
다음은 간호학 문제의 정답 예시와 사용자의 답변입니다.
정답 예시: {correct_answer}
사용자 답변: {user_answer}

0점~100점 사이에서 채점하고, 짧게 이유를 설명하세요.
형식: "점수: XX, 이유: ..."
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# =========================
# 6. Streamlit UI
# =========================
st.title("🧑‍⚕️ 간호사 시험 챗봇")
st.write("CSV 임베딩을 기반으로 가장 유사한 정답을 찾고, 답안을 채점합니다.")

question = st.text_input("질문을 입력하세요:")
user_answer = st.text_area("당신의 답변:")

if st.button("검색 및 채점") and question and user_answer:
    # 1) 질문 임베딩
    q_emb = embed_text(question)

    # 2) 가장 유사한 문제 찾기
    best_match, sim = find_most_similar(q_emb, df)

    st.subheader("📌 가장 유사한 문제")
    st.write(f"**문제:** {best_match['Question']}")
    st.write(f"**정답 예시:** {best_match['Answer']}")
    st.write(f"**카테고리:** {best_match['Etc']}")
    st.write(f"**유사도:** {sim:.2f}")

    # 3) 채점
    st.subheader("📝 채점 결과")
    grade_result = grade_answer(user_answer, best_match["Answer"])
    st.write(grade_result)

