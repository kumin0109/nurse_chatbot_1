import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ====================== 공통 설정 ======================
st.set_page_config(page_title="간호사 상황극 문제은행", page_icon="🩺")

# 🔐 OpenAI 키: 환경변수 우선 → 없으면 Streamlit secrets
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets(⋮ → Settings → Secrets) 또는 환경변수로 추가하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ====================== 데이터 로딩 ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nurse_2_with_embeddings.csv")   # ✅ 파일명 고정
    except FileNotFoundError:
        st.error("CSV 파일 'nurse_2_with_embeddings.csv' 를 찾을 수 없습니다. 파일을 앱 루트에 업로드하세요.")
        st.stop()

    if "Embedding" not in df.columns:
        st.error("CSV에 'Embedding' 컬럼이 없습니다. 임베딩 컬럼명을 확인하세요.")
        st.stop()

    # Embedding 컬럼: 문자열 → 리스트
    def to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x

    df["Embedding"] = df["Embedding"].apply(to_list)

    if len(df) == 0:
        st.error("CSV가 비어 있습니다.")
        st.stop()

    first = df["Embedding"].iloc[0]
    if not isinstance(first, (list, tuple)):
        st.error("Embedding 컬럼이 리스트 형태가 아닙니다. 예: [0.1, 0.2, ...]")
        st.stop()

    embed_dim = len(first)

    # 행별 길이 불일치 제거(있다면 경고)
    bad = df["Embedding"].apply(lambda v: len(v) != embed_dim)
    if bad.any():
        st.warning(f"임베딩 길이가 다른 행 {bad.sum()}개를 발견했습니다. 해당 행은 제외합니다.")
        df = df.loc[~bad].reset_index(drop=True)

    return df, embed_dim

# ====================== 임베딩 함수 ======================
def embed_text(text: str, target_dim: int):
    """
    CSV에 저장된 임베딩 차원(target_dim)에 맞춰 새 임베딩 생성.
    - 1536: text-embedding-3-small (기본 1536)
    - 3072: text-embedding-3-large (기본 3072)
    - 기타: large + dimensions=target_dim 로 맞춤
    """
    if target_dim == 1536:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
    elif target_dim == 3072:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=[text]
        )
    else:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
            dimensions=target_dim
        )
    return resp.data[0].embedding

# ====================== 채점(질문 단위) ======================
def score_for_question(user_embedding, raw_df, current_row, target_dim, q_col, a_col):
    """
    사용자 답변 임베딩을 '현재 질문'의 정답(들)과만 비교.
    - 같은 질문 텍스트가 여러 행에 있으면 모두 후보로 사용.
    - 없을 경우 현재 행만 사용.
    """
    user = np.asarray(user_embedding, dtype=np.float32).reshape(1, -1)

    mask_same_q = raw_df[q_col].astype(str) == str(current_row[q_col])
    subset = raw_df.loc[mask_same_q]

    if subset.empty:
        subset = pd.DataFrame([current_row])

    cand_mat = np.vstack(subset["Embedding"].apply(lambda v: np.asarray(v, dtype=np.float32)))
    if cand_mat.shape[1] != target_dim or user.shape[1] != target_dim:
        raise ValueError(
            f"임베딩 차원 불일치: CSV={cand_mat.shape[1]}, QUERY={user.shape[1]}"
        )

    sims = cosine_similarity(user, cand_mat)[0]
    best_idx = int(np.argmax(sims))
    best_row = subset.iloc[best_idx]
    return best_row, float(sims[best_idx])

# ====================== 앱 상태 초기화 ======================
st.title("🩺 간호사 100문 100답 - 카테고리 선택 문제은행")

if "raw_df" not in st.session_state:
    st.session_state.raw_df, st.session_state.embed_dim = load_data()

if "category_selected" not in st.session_state:
    st.session_state.category_selected = "전체"
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = st.session_state.raw_df.copy()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "solved_ids" not in st.session_state:
    st.session_state.solved_ids = []
if "category_stats" not in st.session_state:
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False

# ====================== 카테고리 필터 ======================
# 👉 Category1 기준으로 카테고리 선택
all_categories = set(st.session_state.raw_df["Category1"].dropna().unique())
category_options = ["전체"] + sorted(list(all_categories))
selected = st.selectbox("📂 푸실 문제 카테고리를 선택하세요:", category_options)

# 카테고리 변경 시 필터링 & 상태 리셋
if selected != st.session_state.category_selected:
    st.session_state.category_selected = selected
    if selected == "전체":
        st.session_state.filtered_df = (
            st.session_state.raw_df.sample(frac=1, random_state=None).reset_index(drop=True)
        )
    else:
        mask = st.session_state.raw_df["Category1"] == selected
        st.session_state.filtered_df = (
            st.session_state.raw_df[mask].sample(frac=1, random_state=None).reset_index(drop=True)
        )
    st.session_state.current_idx = 0
    st.session_state.correct_count = 0
    st.session_state.total_count = 0
    st.session_state.solved_ids = []
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    st.session_state.quiz_finished = False

df = st.session_state.filtered_df
idx = st.session_state.current_idx

# 퀴즈 완료 여부
if idx >= len(df):
    st.session_state.quiz_finished = True

# ====================== 문제 풀이 ======================
if not st.session_state.quiz_finished:
    row = df.iloc[idx]

    # 컬럼 이름 방어코드
    q_col = "Question" if "Question" in df.columns else df.columns[0]
    a_col = "Answer"   if "Answer"   in df.columns else df.columns[1]
    e_col = "Category1"  # ✅ Category1을 카테고리로 사용

    st.markdown(f"**문제 {idx + 1}:** {row[q_col]}")
    user_input = st.text_area("🧑‍⚕️ 당신의 간호사 응답은?", key=f"input_{idx}_{selected}")

    col1, col2 = st.columns(2)
    with col1:
        submit_clicked = st.button("정답 제출", type="primary")
    with col2:
        next_clicked = st.button("다음 문제")

    if submit_clicked and user_input.strip():
        with st.spinner("AI가 채점 중입니다..."):
            try:
                # 1) 사용자 답변 임베딩(차원 자동 맞춤)
                user_embedding = embed_text(user_input, st.session_state.embed_dim)

                # 2) '현재 질문'의 정답(들)과만 비교
                best_match, similarity = score_for_question(
                    user_embedding,
                    st.session_state.raw_df,  # 전체 원본에서 같은 질문 행을 모음
                    row,
                    st.session_state.embed_dim,
                    q_col,
                    a_col
                )

                st.session_state.total_count += 1
                st.session_state.solved_ids.append(idx)

                is_correct = False
                if similarity >= 0.8:
                    st.session_state.correct_count += 1
                    st.success(f"✅ 정답입니다! 유사도 {similarity:.2f}")
                    is_correct = True
                elif similarity >= 0.6:
                    st.info(f"🟡 거의 맞았어요! 유사도 {similarity:.2f}")
                else:
                    st.error(f"❌ 오답입니다. 유사도 {similarity:.2f}")

                # 항상 '현재 질문'의 정답 예시를 보여줌(동일 질문 중 가장 가까운 것)
                st.markdown(f"**정답 예시:**\n> {best_match[a_col]}")
                st.caption(f"🗂️ 카테고리: {str(row[e_col])}")

                # 카테고리 통계 집계(현재 문제 기준)
                st.session_state.category_stats[row[e_col]]["total"] += 1
                if is_correct:
                    st.session_state.category_stats[row[e_col]]["correct"] += 1

            except Exception as e:
                st.error(f"채점 중 오류가 발생했습니다: {e}")

    if next_clicked:
        st.session_state.current_idx += 1

# ====================== 퀴즈 완료 ======================
else:
    st.success("🎉 모든 문제를 완료했습니다!")

    st.subheader("📊 최종 결과 요약")
    correct = st.session_state.correct_count
    total = st.session_state.total_count
    st.markdown(f"- 총 문제 수: **{total}**")
    st.markdown(f"- 맞힌 문제 수: **{correct}**")
    if total > 0:
        st.markdown(f"- 정답률: **{(correct/total)*100:.1f}%**")
    else:
        st.markdown("- 정답률: **0.0%**")

    st.markdown("---")
    st.subheader("🧾 카테고리별 정답 통계")
    stats = st.session_state.category_stats
    for cat, stat in stats.items():
        if stat["total"] > 0:
            rate = stat["correct"] / stat["total"] * 100
            st.write(f"- **{cat}**: {stat['correct']} / {stat['total']} 정답 ({rate:.1f}%)")

    st.markdown("---")
    if st.button("🔁 처음부터 다시 시작하기"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()



