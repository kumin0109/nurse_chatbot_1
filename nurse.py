import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from collections import defaultdict

# ==========================
# 🔐 OpenAI API 키 설정
# ==========================
try:
    api_key = st.secrets["OPENAI_API_KEY"].strip()
except KeyError:
    st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다.\nStreamlit Cloud에서는 Secrets에 설정하세요.")
    st.stop()

# ✅ 키 유효성 검사
if not api_key.startswith("sk-") or len(api_key) < 40:
    st.error("❌ OPENAI_API_KEY 값이 유효하지 않습니다. 다시 확인하세요.")
    st.stop()

# 환경변수 등록 (openai 기본 인증 방식)
os.environ["OPENAI_API_KEY"] = api_key

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# ==========================
# 🔍 API 연결 테스트
# ==========================
with st.spinner("🔍 OpenAI API 인증 확인 중..."):
    try:
        models = client.models.list()
        st.success(f"✅ API 인증 성공! 모델 {len(models.data)}개 확인됨 (예: {models.data[0].id})")
        st.caption(f"🔑 키 앞부분: {api_key[:7]}..., 길이: {len(api_key)}")
    except Exception as e:
        st.error(f"❌ API 인증 실패: {e}")
        st.stop()

# ==========================
# 📥 CSV 불러오기 (캐싱)
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_2_with_embeddings.csv")
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    df["Etc"] = df[["Category1", "Category2", "Department"]].fillna("").astype(str).agg(";".join, axis=1)
    return df

# 텍스트 → 벡터 변환
def embed_text(text: str):
    text = text.strip()
    if not text:
        return None
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

# ==========================
# 🖥 페이지 설정
# ==========================
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
        st.experimental_rerun()

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

    col1, col2, col3 = st.columns(3)

    # 정답 제출 (문제별 채점)
    with col1:
        if st.button("✅ 제출하고 채점"):
            if answer.strip():
                with st.spinner("AI가 채점 중입니다..."):
                    user_embedding = embed_text(answer)
                    if user_embedding is None:
                        st.warning("⚠️ 답변이 비어 있습니다.")
                    else:
                        best_match, similarity = find_most_similar(user_embedding, df)

                        is_correct = similarity >= 0.65
                        if is_correct:
                            st.success(f"✅ 정답입니다! (유사도: {similarity:.2f})")
                        elif similarity >= 0.55:
                            st.info(f"🟡 거의 맞았습니다. (유사도: {similarity:.2f})")
                        else:
                            st.error(f"❌ 오답입니다. (유사도: {similarity:.2f})")

                        st.markdown(f"**정답 예시:** {best_match['Answer']}")
                        st.caption(f"🗂️ 카테고리: {best_match['Etc']}")

                        # 카테고리별 통계 업데이트
                        st.session_state.category_stats[best_match["Etc"]]["total"] += 1
                        if is_correct:
                            st.session_state.category_stats[best_match["Etc"]]["correct"] += 1

    # 다음 문제 버튼
    with col2:
        if idx < len(df) - 1:
            if st.button("➡ 다음 문제"):
                st.session_state.current_idx += 1
                st.experimental_rerun()
        else:
            st.write("마지막 문제입니다.")

    # 모든 문제 채점 종료 버튼
    with col3:
        if st.button("📊 최종 결과 보기"):
            correct_count = sum(
                1 for i, ans in st.session_state.answers.items()
                if ans.strip() and cosine_similarity(
                    [embed_text(ans)], np.array(df["Embedding"].to_list())
                )[0].max() >= 0.65
            )
            st.session_state.results = {
                "correct": correct_count,
                "category_stats": st.session_state.category_stats
            }
            st.session_state.quiz_finished = True
            st.experimental_rerun()


