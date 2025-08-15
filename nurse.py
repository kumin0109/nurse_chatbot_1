import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from collections import defaultdict

# ======================
# 🔐 OpenAI API 키 설정
# ======================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.strip():
    st.error("❌ OPENAI_API_KEY가 없습니다.\n\n"
             "Streamlit Cloud에서는 **Settings → Secrets** 탭에서 다음과 같이 입력하세요:\n"
             '```\nOPENAI_API_KEY="sk-..."\n```')
    st.stop()

client = OpenAI(api_key=api_key)

# ======================
# 📥 CSV 불러오기 (캐싱)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_2_with_embeddings.csv")
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    df["Etc"] = df[["Category1", "Category2", "Department"]].fillna("").astype(str).agg(";".join, axis=1)
    return df

# ======================
# 텍스트 → 벡터 변환
# ======================
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# ======================
# 유사도 계산
# ======================
def find_most_similar(user_embedding, df):
    all_embeddings = np.array(df["Embedding"].to_list())
    sims = cosine_similarity([user_embedding], all_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return df.iloc[best_idx], sims[best_idx]

# ======================
# Streamlit UI 설정
# ======================
st.set_page_config(page_title="간호사 상황극 문제은행", page_icon="🩺")
st.title("🩺 간호사 100문 100답 - 카테고리 선택 문제은행")

# === 세션 초기화 ===
if "raw_df" not in st.session_state:
    st.session_state.raw_df = load_data()
if "category_selected" not in st.session_state:
    st.session_state.category_selected = "전체"
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = st.session_state.raw_df.copy()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False
if "results" not in st.session_state:
    st.session_state.results = None
if "category_stats" not in st.session_state:
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# === 카테고리 목록 ===
all_categories = set()
for etc in st.session_state.raw_df["Etc"]:
    all_categories.update([e.strip() for e in str(etc).split(";") if e.strip()])

category_options = ["전체"] + sorted(list(all_categories))
selected = st.selectbox("📂 푸실 문제 카테고리를 선택하세요:", category_options)

# === 카테고리 변경 시 필터링 ===
if selected != st.session_state.category_selected:
    st.session_state.category_selected = selected
    if selected == "전체":
        st.session_state.filtered_df = st.session_state.raw_df.sample(frac=1).reset_index(drop=True)
    else:
        mask = st.session_state.raw_df["Etc"].apply(lambda x: selected in str(x))
        st.session_state.filtered_df = st.session_state.raw_df[mask].sample(frac=1).reset_index(drop=True)
    st.session_state.current_idx = 0
    st.session_state.answers = {}
    st.session_state.quiz_finished = False
    st.session_state.results = None
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

df = st.session_state.filtered_df
idx = st.session_state.current_idx

# ===== 퀴즈 완료 시 =====
if st.session_state.quiz_finished:
    correct_count = st.session_state.results["correct"]
    total_count = len(st.session_state.answers)
    st.success("🎉 채점이 완료되었습니다!")
    st.markdown(f"- 총 푼


