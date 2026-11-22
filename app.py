import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import COMMITTEE_MERGE, clean_title, extract_words

CSV_PATH = "assembly_bills_clean.csv"

# 검색 예시 버튼에 쓸 검색어
EXAMPLES = ["전기차 화재 안전", "육아휴직 확대", "학교폭력 예방", "전세사기 피해자 지원"]

# 평가에서 나온 구간별 관련 비율을 근거로 나눈 유사도 등급
LEVELS = [(40, "매우 유사", "green"), (20, "유사", "blue"), (10, "일부 유사", "orange"), (0, "낮음", "gray")]

st.set_page_config(page_title="법안 분석 대시보드", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["TITLE_WORDS"] = df["TITLE_WORDS"].fillna("")
    df["SUMMARY_WORDS"] = df["SUMMARY_WORDS"].fillna("")
    df["SUMMARY"] = df["SUMMARY"].fillna("")
    df["PROPOSER"] = df["PROPOSER"].fillna("제안자 미상")
    return df


# 제목과 본문 단어를 합쳐서 법안 전체를 벡터로 바꿈
@st.cache_resource
def build_vectors(docs):
    # 절반 넘는 법안에 나오는 단어는 구분력이 없어서 제외
    vectorizer = TfidfVectorizer(max_df=0.5)
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


# 법안마다 가진 단어 목록
@st.cache_resource
def build_word_sets(docs):
    return [set(d.split()) for d in docs]


# 내용이 같은 법안을 묶을 때 쓰는 본문만의 벡터
@st.cache_resource
def build_body_vectors(docs):
    return TfidfVectorizer(max_df=0.5).fit_transform(docs)


def get_level(score):
    for bound, name, color in LEVELS:
        if score >= bound:
            return name, color


# 검색어와 법안에 함께 나온 단어 중 비중이 큰 것
def common_words(query_vec, i, count=5):
    both = query_vec.multiply(matrix[i]).toarray().flatten()
    order = both.argsort()[::-1]
    found = [terms[k] for k in order if both[k] > 0]
    # 인공지능이 있으면 인공, 지능은 빼고 보여줌
    found = [w for w in found if not any(w != x and w in x for x in found)]
    return found[:count]


# 점수가 높은 순으로 뽑되 내용이 거의 같은 법안은 한 건으로 묶음
def pick_top(scores, query_vec, top_n=5):
    order = scores.argsort()[::-1][:30]
    picked = []
    same_count = {}
    for i in order:
        if scores[i] == 0:
            break
        merged = False
        for j in picked:
            if cosine_similarity(body_matrix[i], body_matrix[j])[0, 0] >= 0.95:
                same_count[j] += 1
                merged = True
                break
        if not merged and len(picked) < top_n:
            picked.append(i)
            same_count[i] = 0

    result = df.iloc[picked].copy()
    result["유사도(%)"] = (scores[picked] * 100).round(1)
    result["동일 내용 법안"] = [same_count[i] for i in picked]
    result["공통 단어"] = [", ".join(common_words(query_vec, i)) for i in picked]
    return result.reset_index(drop=True)


# 검색어와 비슷한 법안 찾기
def search(query, mask, min_score):
    words = extract_words(clean_title(query))
    if words == "":
        return None
    query_vec = vectorizer.transform([words])
    scores = cosine_similarity(query_vec, matrix).flatten()

    # 검색어 단어를 많이 가진 법안일수록 점수를 덜 깎음
    query_words = words.split()
    ratio = np.array([sum(w in s for w in query_words) / len(query_words) for s in word_sets])
    scores = scores * ratio

    # 필터에 안 맞거나 기준보다 낮은 법안은 0점 처리
    scores = np.where(mask, scores, 0).astype(float)
    scores[scores < min_score / 100] = 0
    return pick_top(scores, query_vec)


# 고른 법안 자체와 비슷한 법안 찾기
def search_by_bill(i, min_score):
    query_vec = matrix[i]
    scores = cosine_similarity(query_vec, matrix).flatten()
    scores[i] = 0
    scores[scores < min_score / 100] = 0
    return pick_top(scores, query_vec)


def bill_info(row):
    return (
        f"제{row['AGE']}대 / 제안자: {row['PROPOSER']} / 발의일: {row['PROPOSE_DT']} / "
        f"소관: {row['CURR_COMMITTEE']} / 처리결과: {row['PROC_RESULT_CD']}"
    )


# 검색 결과를 그래프와 펼침 목록으로 보여줌
def show_results(result, min_score, by_bill=False):
    if result is None:
        st.warning("검색어에서 단어를 찾지 못했습니다. 조금 더 구체적으로 입력해 주세요.")
        return
    if len(result) == 0:
        st.info(f"조건에 맞고 유사도가 {min_score}% 이상인 법안이 없습니다.")
        return

    result["등급"] = [get_level(s)[0] for s in result["유사도(%)"]]
    # 같은 이름의 법안이 많아서 순위와 제안자를 같이 표시
    result["법안"] = [f"{i + 1}. {row['BILL_NAME']} · {row['PROPOSER']}" for i, row in result.iterrows()]
    result["유사도"] = result["유사도(%)"]
    names = [name for _, name, _ in LEVELS]
    colors = ["#2e9e5b", "#3b6fd8", "#e08a1e", "#9aa0a6"]
    chart = (
        alt.Chart(result)
        .mark_bar()
        .encode(
            x=alt.X("유사도", title="유사도(%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("법안", sort=None, title=None, axis=alt.Axis(labelLimit=500)),
            color=alt.Color("등급", scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(title=None, orient="top")),
            tooltip=["법안", "유사도", "등급"],
        )
    )
    st.altair_chart(chart)

    for _, row in result.iterrows():
        name, color = get_level(row["유사도(%)"])
        title = f":{color}[**{name}**] {row['유사도(%)']}% · {row['BILL_NAME']}"
        if row["IS_ALT"]:
            title += " · 위원회 대안"
        with st.expander(title):
            st.write(bill_info(row))
            st.write(f"겹친 단어: {row['공통 단어']}")
            if row["동일 내용 법안"] > 0:
                st.write(f"내용이 거의 같은 법안이 {row['동일 내용 법안']}건 더 있습니다.")
            st.write(row["SUMMARY"] if row["SUMMARY"] else "제안이유 및 주요내용이 없는 법안입니다.")
            st.markdown(f"[의안정보시스템에서 보기]({row['LINK_URL']})")

    if by_bill:
        how = "고른 법안과 각 법안의 TF-IDF 코사인 유사도입니다."
    else:
        how = "TF-IDF 코사인 유사도에 검색어 단어 포함 비율을 곱한 값입니다."
    st.caption(f"유사도는 {how} 등급 기준은 40% 이상 매우 유사, 20% 이상 유사, 10% 이상 일부 유사입니다.")


# 예시 버튼을 누르면 검색창을 채우고 바로 검색
def use_example(text):
    st.session_state["query"] = text
    st.session_state["run_search"] = True


df = load_data()
docs = tuple(df["TITLE_WORDS"] + " " + df["SUMMARY_WORDS"])
vectorizer, matrix = build_vectors(docs)
word_sets = build_word_sets(docs)
body_matrix = build_body_vectors(tuple(df["SUMMARY_WORDS"]))
terms = vectorizer.get_feature_names_out()

st.title("법안 분석 대시보드")
st.write("법안 제목이나 정책 내용을 입력하면, 내용이 비슷한 기존 법안을 찾아드립니다.")
ages = "·".join(str(a) for a in sorted(df["AGE"].unique()))
st.caption(
    f"열린국회정보 Open API 수집 자료 · 제{ages}대 국회 · "
    f"발의일 {df['PROPOSE_DT'].min()} ~ {df['PROPOSE_DT'].max()}"
)

col1, col2, col3 = st.columns(3)
col1.metric("수집 법안 수", f"{len(df):,}건")
col2.metric("소관 위원회 수", f"{df['CURR_COMMITTEE'].nunique()}개")
col3.metric("가결 법안 수", f"{df['PROC_RESULT_CD'].str.contains('가결').sum():,}건")

tab_find, tab_title, tab_stats = st.tabs(["유사 법안 찾기", "법안 검색", "통계"])

with tab_find:
    st.write("검색 예시")
    buttons = st.columns(len(EXAMPLES))
    for col, text in zip(buttons, EXAMPLES):
        col.button(text, on_click=use_example, args=(text,), width="stretch")

    st.session_state.setdefault("query", "인구감소지역 기업 세제 지원")

    # 폼으로 묶어서 Enter만 눌러도 검색되게 함
    with st.form("search_form"):
        query = st.text_input("법안 제목이나 내용을 입력하세요", key="query")

        # 아무것도 고르지 않으면 전체에서 검색
        f1, f2, f3 = st.columns(3)
        pick_age = f1.multiselect("대수", sorted(df["AGE"].unique()))
        pick_committee = f2.multiselect("소관 위원회", sorted(df["CURR_COMMITTEE"].unique()))
        pick_result = f3.multiselect("처리 결과", sorted(df["PROC_RESULT_CD"].unique()))

        # 평가에서 10% 미만 결과는 거의 관련이 없었음
        min_score = st.slider("이 값보다 유사도가 낮은 법안은 숨기기(%)", 0, 50, 10)
        submitted = st.form_submit_button("검색", type="primary")

    if submitted or st.session_state.pop("run_search", False):
        mask = np.ones(len(df), dtype=bool)
        if pick_age:
            mask &= df["AGE"].isin(pick_age).to_numpy()
        if pick_committee:
            mask &= df["CURR_COMMITTEE"].isin(pick_committee).to_numpy()
        if pick_result:
            mask &= df["PROC_RESULT_CD"].isin(pick_result).to_numpy()
        show_results(search(query, mask, min_score), min_score)

with tab_title:
    keyword = st.text_input("법안 제목에 들어간 단어를 입력하세요", value="개인정보")

    if keyword.strip():
        found = df[df["BILL_NAME"].str.contains(keyword.strip(), regex=False)]
        found = found.sort_values("PROPOSE_DT", ascending=False)
        st.write(f"{len(found):,}건을 찾았습니다.")

        if len(found) > 0:
            table = found[["BILL_NAME", "AGE", "PROPOSER", "PROPOSE_DT", "CURR_COMMITTEE", "PROC_RESULT_CD", "LINK_URL"]]
            table.columns = ["법안명", "대수", "제안자", "발의일", "소관 위원회", "처리 결과", "원문"]
            st.dataframe(
                table,
                hide_index=True,
                column_config={"원문": st.column_config.LinkColumn("원문", display_text="보기")},
            )

            # 고른 법안의 상세 내용 보여주기
            labels = [f"{r.BILL_NAME} · {r.PROPOSER} · {r.PROPOSE_DT}" for r in found.itertuples()]
            pick = st.selectbox("자세히 볼 법안", range(len(found)), format_func=lambda i: labels[i])
            row = found.iloc[pick]
            with st.container(border=True):
                st.markdown(f"**{row['BILL_NAME']}**")
                st.write(bill_info(row))
                st.write(row["SUMMARY"] if row["SUMMARY"] else "제안이유 및 주요내용이 없는 법안입니다.")
                st.markdown(f"[의안정보시스템에서 보기]({row['LINK_URL']})")

            # 고른 법안의 제목과 본문 전체로 비슷한 법안을 찾음
            if st.button("이 법안과 비슷한 법안 찾기", type="primary"):
                st.subheader("비슷한 법안")
                show_results(search_by_bill(found.index[pick], 10), 10, by_bill=True)

with tab_stats:
    stat1, stat2, stat3 = st.tabs(["소관 위원회별", "처리 결과별", "월별 발의"])
    with stat1:
        st.bar_chart(df["CURR_COMMITTEE"].value_counts(), horizontal=True)
        merged = ", ".join(f"{old} → {new}" for old, new in COMMITTEE_MERGE.items() if " " not in old)
        st.caption(f"22대 국회에서 이름이 바뀐 위원회는 바뀐 이름으로 합쳐서 셌습니다. ({merged})")
    with stat2:
        st.bar_chart(df["PROC_RESULT_CD"].value_counts(), horizontal=True)
    with stat3:
        st.bar_chart(df["MONTH"].value_counts().sort_index())
