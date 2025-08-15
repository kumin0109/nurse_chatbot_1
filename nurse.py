import os
import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from collections import defaultdict

# ===== OpenAI API 키 설정 (secrets → 환경변수 순서로 시도) =====
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("❌ OpenAI API Key가 없습니다. .streamlit/secrets.toml 또는 환경변수에 설정하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 📥 CSV 불러오기 (캐싱)
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_2_with_embeddings.csv")
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    df["Etc"] = df[["Category1", "Category2", "Department"]].fillna("").astype(str).agg(";".join, axis=1)
    return df

# 텍스트 → 벡터(임베딩)
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# 유사도 계산
def find_most_similar(user_embedding, df):
    all_embeddings = np.array(df["Embedding"].to_list())
    sims = cosine_similarity([user_embedding], all_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return df.iloc[best_idx], sims[best_idx]

# 페이지 설정
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

# === 카테고리 목록 ===
all_categories = set()
for etc in st.session_state.raw_df["Etc"]:
    all_categories.update([e.strip() for e in str(etc).split(";") if e.strip()])

category_options = ["전체"] + sorted(list(all_categories))
selected = st.selectbox("📂 푸실 문제 카테고리를 선택하세요:", category_options)

# === 카테고리 변경 시 데이터 필터링 ===
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

df = st.session_state.filtered_df
idx = st.session_state.current_idx

# ===== 퀴즈 완료 시 =====
if st.session_state.quiz_finished:
    correct_count = st.session_state.results["correct"]
    total_count = len(st.session_state.answers)
    st.success("🎉 채점이 완료되었습니다!")
    st.markdown(f"- 총 푼 문제 수: **{total_count}**")
    st.markdown(f"- 맞힌 문제 수: **{correct_count}**")
    st.markdown(f"- 정답률: **{(correct_count/total_count)*100:.1f}%**")

    st.subheader("🧾 카테고리별 정답 통계")
    for cat, stat in st.session_state.results["category_stats"].items():
        if stat["total"] > 0:
            rate = stat["correct"] / stat["total"] * 100
            st.write(f"- **{cat}**: {stat['correct']} / {stat['total']} 정답 ({rate:.1f}%)")

    if st.button("🔁 처음부터 다시 시작하기"):
        st.session_state.clear()
        st.rerun()

# ===== 진행 중 =====
else:
    row = df.iloc[idx]
    st.markdown(f"**문제 {idx + 1}/{len(df)}:** {row['Question']}")

    answer = st.text_area(
        "🧑‍⚕️ 당신의 간호사 응답은?",
        value=st.session_state.answers.get(idx, ""),
        key=f"input_{idx}"
    )
    st.session_state.answers[idx] = answer

    col1, col2 = st.columns(2)

    # 다음 문제 버튼
    with col1:
        if idx < len(df) - 1:
            if st.button("➡ 다음 문제"):
                st.session_state.current_idx += 1
                st.rerun()
        else:
            st.write("마지막 문제입니다.")

    # 정답 제출 버튼
    with col2:
        if st.button("✅ 정답 제출"):
            correct_count = 0
            category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

            for i, user_ans in st.session_state.answers.items():
                user_embedding = embed_text(user_ans)
                best_match, similarity = find_most_similar(user_embedding, df)

                is_correct = similarity >= 0.8
                if is_correct:
                    correct_count += 1

                for category in best_match["Etc"].split(";"):
                    category = category.strip()
                    category_stats[category]["total"] += 1
                    if is_correct:
                        category_stats[category]["correct"] += 1

            st.session_state.results = {
                "correct": correct_count,
                "category_stats": category_stats
            }
            st.session_state.quiz_finished = True
            st.rerun()


