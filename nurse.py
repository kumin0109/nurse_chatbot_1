import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from collections import defaultdict

# 🔐 OpenAI API 키 설정 (st.secrets 없으면 환경변수 사용)
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("❌ OpenAI API Key가 설정되지 않았습니다. secrets 또는 환경변수를 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 📥 CSV 불러오기 (캐싱)
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_2_with_embeddings.csv")
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    df["Etc"] = df[["Category1", "Category2", "Department"]].fillna("").astype(str).agg(";".join, axis=1)
    return df

# 🧠 텍스트 → 임베딩 변환
def embed_text(text: str) -> list:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {e}")
        return None

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
    st.markdown(f"- 총 푼 문제 수: **{total_count}**")
    st.markdown(f"- 맞힌 문제 수: **{correct_count}**")
    st.markdown(f"- 정답률: **{(correct_count/total_count)*100:.1f}%**")

    st.subheader("🧾 카테고리별 정답 통계")
    for cat, stat in st.session_state.results["category_stats"].items():
        if stat["total"] > 0:
            rate = stat["correct"] / stat["total"] * 100
            st.write(f"- **{cat}**: {stat['correct']} / {stat['total']} 정답 ({rate:.1f}%)")

    if st.button("🔁 처음부터 다시 시작하기"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ===== 진행 중 =====
else:
    row = df.iloc[idx]
    st.markdown(f"**문제 {idx + 1}/{len(df)}:** {row['Question']}")

    answer = st.text_area("🧑‍⚕️ 당신의 간호사 응답은?",
                          value=st.session_state.answers.get(idx, ""),
                          key=f"input_{idx}")
    st.session_state.answers[idx] = answer

    col1, col2, col3 = st.columns(3)

    # 정답 제출
    with col1:
        if st.button("✅ 제출하고 채점"):
            if answer.strip():
                with st.spinner("AI가 채점 중입니다..."):
                    user_embedding = embed_text(answer)
                    if user_embedding:
                        # ✅ 현재 문제(row)의 정답과만 비교
                        correct_embedding = np.array(row["Embedding"]).reshape(1, -1)
                        similarity = cosine_similarity([user_embedding], correct_embedding)[0][0]

                        # 채점 결과 표시
                        if similarity >= 0.65:
                            st.success(f"✅ 정답입니다! (유사도: {similarity:.2f})")
                        elif similarity >= 0.55:
                            st.info(f"🟡 거의 맞았습니다. (유사도: {similarity:.2f})")
                        else:
                            st.error(f"❌ 오답입니다. (유사도: {similarity:.2f})")

                        # ✅ 항상 현재 문제의 정답 예시만 출력
                        st.markdown(f"**정답 예시:** {row['Answer']}")
                        st.caption(f"🗂️ 카테고리: {row['Etc']}")

                        # 카테고리별 통계
                        st.session_state.category_stats[row["Etc"]]["total"] += 1
                        if similarity >= 0.65:
                            st.session_state.category_stats[row["Etc"]]["correct"] += 1

    # 다음 문제
    with col2:
        if idx < len(df) - 1:
            if st.button("➡ 다음 문제"):
                st.session_state.current_idx += 1
                st.rerun()
        else:
            st.write("마지막 문제입니다.")

    # 최종 결과
    with col3:
        if st.button("📊 최종 결과 보기"):
            correct_count = 0
            for i, ans in st.session_state.answers.items():
                if ans.strip():
                    emb = embed_text(ans)
                    if emb is not None:
                        sim = cosine_similarity(
                            [emb], np.array(df.iloc[i]["Embedding"]).reshape(1, -1)
                        )[0][0]
                        if sim >= 0.65:
                            correct_count += 1

            st.session_state.results = {
                "correct": correct_count,
                "category_stats": st.session_state.category_stats
            }
            st.session_state.quiz_finished = True
            st.rerun()

