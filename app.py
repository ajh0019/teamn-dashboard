import itertools
import re
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import COMMITTEE_MERGE, clean_summary, clean_title, extract_words

SERVICE_NAME = "POLICY INSIGHT"
SERVICE_SUB = "국회 법안 AI 유사도 분석 플랫폼"
SERVICE_SLOGAN = "국회 법안, 데이터로 읽다."
SERVICE_INTRO = "법안 검색부터 유사 법안 분석까지, {n}건의 국회 데이터를 한눈에."

CSV_PATH = "assembly_bills_clean.csv"
GITHUB_URL = "https://github.com/ProgrammingTeamN/TeamN"
TEAM = "인공지능프로그래밍 N조 · 안정훈, 홍진수, 장소희"

# 검색 예시 버튼에 쓸 검색어
EXAMPLES = ["전기차 화재 안전", "육아휴직 확대", "학교폭력 예방", "전세사기 피해자 지원"]

# 평가에서 나온 구간별 관련 비율을 근거로 나눈 유사도 등급
LEVELS = [(40, "매우 유사", "green"), (20, "유사", "blue"), (10, "일부 유사", "orange"), (0, "낮음", "gray")]

# 유사도 막대 색. 등급 배지 색과 맞춤
BAR_COLORS = {"green": "#16a34a", "blue": "#2563eb", "orange": "#f59e0b", "gray": "#94a3b8"}

# 그래프에서 너무 긴 위원회 이름은 줄여서 보여줌
SHORT_NAMES = {
    "12.29여객기참사진상규명과피해자및유가족의피해구제를위한특별위원회": "12.29 여객기참사 특별위원회",
}

NAVY = "#1e3a8a"
ALL = "전체"

st.set_page_config(page_title=SERVICE_NAME, page_icon="⚖️", layout="wide")

# 카드와 상단 배너 모양
st.markdown(
    """
    <style>
    [class*="st-key-card"] { background: #ffffff; border-radius: 14px; box-shadow: 0 1px 4px rgba(15, 31, 75, 0.07); }
    .st-key-band { background: linear-gradient(120deg, #0f1f4b 0%, #1e40af 100%); border-radius: 14px; padding: 20px 24px 8px 24px; }
    .st-key-band label p, .st-key-band [data-testid="stWidgetLabel"] p { color: #e5eaf5 !important; font-weight: 600; }
    .hero .badge { display: inline-block; background: rgba(255,255,255,0.14); color: #e0e7ff; border-radius: 999px;
                   padding: 4px 14px; font-size: 0.9rem; margin-bottom: 14px; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e3e8f2; border-radius: 12px; padding: 14px 18px; }
    .hero { background: linear-gradient(120deg, #0f1f4b 0%, #1e40af 100%); color: #ffffff;
            border-radius: 16px; padding: 34px 38px; margin-bottom: 18px; }
    .hero h1 { color: #ffffff; font-size: 2.6rem; margin: 0 0 8px 0; padding: 0; }
    .hero .sub { color: #c7d2fe; font-size: 1.05rem; margin-bottom: 14px; }
    .hero .intro { color: #ffffff; font-size: 1.2rem; }
    .page-title { font-size: 1.7rem; font-weight: 700; color: #0f1f4b; margin-bottom: 0; }
    .page-desc { color: #5b6476; margin-bottom: 14px; }
    .count-box { background: #0f1f4b; color: #ffffff; border-radius: 12px; padding: 14px 20px; display: inline-block; }
    .count-box b { font-size: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="법안 자료를 불러오는 중입니다...")
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["TITLE_WORDS"] = df["TITLE_WORDS"].fillna("")
    df["SUMMARY_WORDS"] = df["SUMMARY_WORDS"].fillna("")
    df["SUMMARY"] = df["SUMMARY"].fillna("")
    df["PROPOSER"] = df["PROPOSER"].fillna("제안자 미상")
    df["대수"] = "제" + df["AGE"].astype(str) + "대"
    df["위원회"] = df["CURR_COMMITTEE"].replace(SHORT_NAMES)
    return df


# 제목과 본문 단어를 합쳐서 법안 전체를 벡터로 바꿈
@st.cache_resource(show_spinner="검색 준비 중입니다...")
def build_vectors(docs):
    # 절반 넘는 법안에 나오는 단어는 구분력이 없어서 제외
    vectorizer = TfidfVectorizer(max_df=0.5)
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


# 법안마다 가진 단어 목록
@st.cache_resource(show_spinner=False)
def build_word_sets(docs):
    return [set(d.split()) for d in docs]


# 내용이 같은 법안을 묶을 때 쓰는 본문만의 벡터
@st.cache_resource(show_spinner=False)
def build_body_vectors(docs):
    return TfidfVectorizer(max_df=0.5).fit_transform(docs)


def get_level(score):
    for bound, name, color in LEVELS:
        if score >= bound:
            return name, color


# 처리 결과에 맞는 배지 색
def result_color(result):
    if "가결" in result:
        return "green"
    if result == "계류중":
        return "blue"
    if result == "철회":
        return "orange"
    return "gray"


# 본문에 마크다운 기호가 섞여 있으면 화면이 깨져서 막아둠
def md_escape(text):
    return re.sub(r"([\\`*_~\[\]$#<>])", r"\\\1", text)


# 겹친 단어에 형광펜 표시
def highlight(text, words):
    text = md_escape(text)
    if not words:
        return text
    words = sorted(words, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
    return pattern.sub(lambda m: f":orange-background[{m.group(0)}]", text)


# 두 벡터에 함께 나온 단어와 그 단어가 점수에서 차지하는 비율(%)
# 코사인 유사도는 단어별 곱의 합이라 단어마다 몫을 나눌 수 있음
def word_shares(vec_a, vec_b, count=5):
    both = vec_a.multiply(vec_b).toarray().flatten()
    total = both.sum()
    if total == 0:
        return []
    found = {terms[k]: both[k] for k in both.argsort()[::-1] if both[k] > 0}
    # 인공지능이 있으면 인공, 지능의 몫은 인공지능에 합침
    shares = {}
    for w, v in found.items():
        bigger = [x for x in found if x != w and w in x]
        key = bigger[0] if bigger else w
        shares[key] = shares.get(key, 0) + v
    top = sorted(shares.items(), key=lambda x: -x[1])[:count]
    return [(w, round(v / total * 100)) for w, v in top]


# 점수가 높은 순으로 뽑되 내용이 거의 같은 법안은 한 건으로 묶음
def pick_top(scores, query_vec, top_n=5):
    order = scores.argsort()[::-1][:30]
    picked = []
    same = {}
    for i in order:
        if scores[i] == 0:
            break
        merged = False
        for j in picked:
            if cosine_similarity(body_matrix[i], body_matrix[j])[0, 0] >= 0.95:
                same[j].append(i)
                merged = True
                break
        if not merged and len(picked) < top_n:
            picked.append(i)
            same[i] = []

    result = df.iloc[picked].copy()
    result["번호"] = picked
    result["유사도(%)"] = (scores[picked] * 100).round(1)
    result["동일 내용 법안"] = [len(same[i]) for i in picked]
    result["동일 목록"] = [same[i] for i in picked]
    result["단어 기여"] = [word_shares(query_vec, matrix[i]) for i in picked]
    result["공통 단어"] = [[w for w, _ in s] for s in result["단어 기여"]]
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


def bill_meta(row):
    return f"제{row['AGE']}대 · {row['PROPOSER']} · {row['PROPOSE_DT']} 발의 · {row['CURR_COMMITTEE']}"


# 대수, 위원회, 처리 결과 조건에 맞는 법안만 True
def make_mask(data, ages, committees, results):
    mask = np.ones(len(data), dtype=bool)
    if ages:
        mask &= data["AGE"].isin(ages).to_numpy()
    if committees:
        mask &= data["CURR_COMMITTEE"].isin(committees).to_numpy()
    if results:
        mask &= data["PROC_RESULT_CD"].isin(results).to_numpy()
    return mask


# 흰 카드 모양의 테두리 상자. 키가 겹치지 않게 번호를 붙임
CARD_NO = itertools.count()


def card(parent=st):
    return parent.container(border=True, key=f"card_{next(CARD_NO)}")


def page_header(title, desc):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-desc">{desc}</div>', unsafe_allow_html=True)


# 등급 색으로 칠한 유사도 막대
def score_bar(score, color):
    width = min(score, 100)
    st.markdown(
        f"""
        <div style="font-size:0.9rem; margin-bottom:4px;">유사도 <b>{score}%</b></div>
        <div style="background:#e8edf7; border-radius:6px; height:10px; margin-bottom:12px;">
          <div style="width:{width}%; background:{BAR_COLORS[color]}; height:10px; border-radius:6px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 결과 한 건을 카드로 보여줌
def show_card(rank, row):
    with card():
        name, color = get_level(row["유사도(%)"])
        head = f":{color}-badge[{name}] :{result_color(row['PROC_RESULT_CD'])}-badge[{row['PROC_RESULT_CD']}]"
        if row["IS_ALT"]:
            head += " :violet-badge[위원회 대안]"
        st.markdown(head)
        st.markdown(f"#### {rank}. {md_escape(row['BILL_NAME'])}")
        score_bar(row["유사도(%)"], color)
        st.caption(bill_meta(row))

        words = row["공통 단어"]
        if words:
            st.markdown("겹친 단어 " + " ".join(f":orange-background[{w}]" for w in words))
            with st.popover("왜 검색됐나요?"):
                show_shares(row["단어 기여"])

        # 내용이 거의 같은 법안 목록
        if row["동일 내용 법안"] > 0:
            with st.expander(f"내용이 거의 같은 법안 {row['동일 내용 법안']}건 더 보기"):
                for j in row["동일 목록"]:
                    other = df.iloc[j]
                    st.markdown(
                        f":{result_color(other['PROC_RESULT_CD'])}-badge[{other['PROC_RESULT_CD']}] "
                        f"[{md_escape(other['BILL_NAME'])}]({other['LINK_URL']})"
                    )
                    st.caption(bill_meta(other))

        summary = row["SUMMARY"]
        if summary:
            # 미리보기에서는 제안이유 및 주요내용 머리말을 뺌
            body = clean_summary(summary)
            preview = body[:150] + ("..." if len(body) > 150 else "")
            st.markdown(highlight(preview, words))
            with st.expander("본문 전체 보기"):
                st.markdown(highlight(summary, words))
                st.markdown(f"[의안정보시스템에서 원문 보기]({row['LINK_URL']})")
        else:
            st.caption("제안이유 및 주요내용이 없는 법안입니다.")
            st.markdown(f"[의안정보시스템에서 원문 보기]({row['LINK_URL']})")


# 단어별로 점수에서 차지한 몫을 막대로 보여줌
def show_shares(shares):
    st.caption("겹친 단어가 유사도 점수에서 차지하는 비율입니다.")
    for w, pct in shares:
        st.progress(min(pct / 100, 1.0), text=f"{w} {pct}%")


# 결과 법안과 같은 내용 법안까지 합쳐 처리 결과와 발의 시점을 요약
def show_summary(result):
    rows = list(result["번호"]) + [j for group in result["동일 목록"] for j in group]
    found = df.iloc[rows]

    with card():
        st.markdown(f"#### 한눈에 보기 · 관련 법안 {len(found)}건")
        if len(found) > len(result):
            st.caption(f"결과 {len(result)}건과 내용이 거의 같은 법안 {len(found) - len(result)}건을 합친 수입니다.")

        counts = found["PROC_RESULT_CD"].value_counts()
        st.markdown(" ".join(f":{result_color(k)}-badge[{k} {v}건]" for k, v in counts.items()))

        # 같은 주제 법안이 언제 나왔는지 점으로 표시
        points = found.assign(발의일=pd.to_datetime(found["PROPOSE_DT"]), 처리결과=found["PROC_RESULT_CD"])
        shown = list(counts.index)
        chart = (
            alt.Chart(points)
            .mark_circle(size=140, opacity=0.85)
            .encode(
                x=alt.X("발의일:T", title=None, axis=alt.Axis(format="%Y-%m")),
                y=alt.Y("대수:N", title=None),
                color=alt.Color(
                    "처리결과:N",
                    scale=alt.Scale(domain=shown, range=[RESULT_COLORS.get(k, "#94a3b8") for k in shown]),
                    legend=alt.Legend(title=None, orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("BILL_NAME", title="법안명"),
                    alt.Tooltip("PROPOSER", title="제안자"),
                    alt.Tooltip("PROPOSE_DT", title="발의일"),
                    alt.Tooltip("처리결과", title="처리 결과"),
                ],
            )
            .properties(height=110)
        )
        draw(chart)


# 두 법안을 나란히 놓고 겹친 단어를 표시
def show_compare(result, key):
    with card():
        st.markdown("#### 두 법안 나란히 비교")
        names = [f"{i + 1}. {n}" for i, n in enumerate(result["BILL_NAME"])]
        c1, c2 = st.columns(2)
        a = c1.selectbox("왼쪽 법안", range(len(result)), format_func=lambda i: names[i], key=f"cmp_a_{key}")
        b = c2.selectbox("오른쪽 법안", range(len(result)), index=1, format_func=lambda i: names[i], key=f"cmp_b_{key}")
        if a == b:
            st.info("서로 다른 법안 두 건을 골라 주세요.")
            return

        i, j = result.loc[a, "번호"], result.loc[b, "번호"]
        sim = cosine_similarity(matrix[i], matrix[j])[0, 0] * 100
        shared = [w for w, _ in word_shares(matrix[i], matrix[j], 8)]
        st.markdown(f"두 법안의 유사도 **{sim:.1f}%**")
        if shared:
            st.markdown("함께 나온 단어 " + " ".join(f":orange-background[{w}]" for w in shared))

        left, right = st.columns(2)
        for col, k in ((left, i), (right, j)):
            row = df.iloc[k]
            with col:
                st.markdown(f":{result_color(row['PROC_RESULT_CD'])}-badge[{row['PROC_RESULT_CD']}]")
                st.markdown(f"**{md_escape(row['BILL_NAME'])}**")
                st.caption(bill_meta(row))
                with st.container(height=320, border=False):
                    body = clean_summary(row["SUMMARY"])
                    st.markdown(highlight(body, shared) if body else "제안이유 및 주요내용이 없는 법안입니다.")


# 검색 결과 전체를 보여주고 내려받기 버튼을 붙임
def show_results(result, key, by_bill=False):
    result = result.copy()
    result["순위"] = range(1, len(result) + 1)

    show_summary(result)

    top, order, down = st.columns([3, 2, 1], vertical_alignment="center")
    top.markdown(f"**비슷한 법안 {len(result)}건**")
    sort = order.segmented_control(
        "정렬", ["유사도순", "최신 발의순"], default="유사도순", key=f"sort_{key}", label_visibility="collapsed"
    )
    if sort == "최신 발의순":
        result = result.sort_values("PROPOSE_DT", ascending=False)

    out = result[["BILL_NAME", "AGE", "PROPOSER", "PROPOSE_DT", "CURR_COMMITTEE", "PROC_RESULT_CD", "유사도(%)", "LINK_URL"]]
    out.columns = ["법안명", "대수", "제안자", "발의일", "소관 위원회", "처리 결과", "유사도(%)", "원문"]
    down.download_button(
        "결과 내려받기",
        out.to_csv(index=False).encode("utf-8-sig"),
        file_name="유사법안_검색결과.csv",
        mime="text/csv",
        key=f"down_{key}",
        width="stretch",
    )

    for _, row in result.iterrows():
        show_card(row["순위"], row)

    if len(result) >= 2:
        show_compare(result.sort_values("순위").reset_index(drop=True), key)

    if by_bill:
        how = "고른 법안과 각 법안의 TF-IDF 코사인 유사도입니다."
    else:
        how = "TF-IDF 코사인 유사도에 검색어 단어 포함 비율을 곱한 값입니다."
    st.caption(f"유사도는 {how} 등급 기준은 40% 이상 매우 유사, 20% 이상 유사, 10% 이상 일부 유사입니다.")


# 지금 검색 결과를 여는 링크
def show_share_link(query):
    base = st.context.url.split("?")[0]
    with st.popover("🔗 검색 결과 링크"):
        st.caption("이 주소를 열면 같은 검색 결과가 바로 나옵니다.")
        st.code(f"{base}?q={quote(query)}", language=None)


# 결과가 없을 때 다시 찾아볼 방법을 알려줌
def show_empty(query, mask, min_score, key):
    words = extract_words(clean_title(query), compound=False).split()
    if not words:
        st.warning("검색어에서 단어를 찾지 못했습니다. 정책 내용을 나타내는 명사를 넣어 다시 검색해 보세요.")
    else:
        st.info(f"조건에 맞고 유사도가 {min_score}% 이상인 법안이 없습니다.")
        lower = search(query, mask, 0)
        if min_score > 0 and lower is not None and len(lower) > 0:
            st.caption(f"검색 설정에서 유사도 기준을 낮추면 {len(lower)}건을 볼 수 있습니다.")

    # 단어를 하나씩 떼어 보거나 예시로 검색
    tries = [w for w in dict.fromkeys(words) if len(words) > 1] + [e for e in EXAMPLES if e != query]
    st.pills("이렇게 찾아보세요", tries, key=f"retry_{key}", on_change=pick_chip, args=(f"retry_{key}",))


# 검색어를 정하고 최근 검색어에 남김
def run_query(text):
    text = text.strip()
    st.session_state["query"] = text
    st.session_state["active_query"] = text
    # 주소에 검색어를 남겨 링크로 같은 결과를 열 수 있게 함
    if text:
        st.query_params["q"] = text
    else:
        st.query_params.pop("q", None)
    if text:
        history = [h for h in st.session_state.get("history", []) if h != text]
        st.session_state["history"] = ([text] + history)[:5]


# 칩을 누르면 그 단어로 검색하고 선택 표시는 바로 지움
def pick_chip(key):
    text = st.session_state[key]
    if text:
        run_query(text)
    st.session_state[key] = None


def on_submit():
    run_query(st.session_state["query"])


def find_similar(bill_id):
    st.session_state["similar_bill"] = bill_id


# 통계 그래프 위에 붙는 조건 선택칸. 전체를 고르면 거르지 않음
def stat_filter(label, options, key):
    pick = st.selectbox(label, [ALL] + list(options), key=key)
    return None if pick == ALL else pick


# 흰 카드 위에 그래프 배경이 회색으로 떠 보이지 않게 함
def draw(chart):
    st.altair_chart(chart.configure(background="#ffffff"))


def bar_height(n):
    return max(160, n * 28)


# ---------------- 페이지 ----------------

def page_home():
    st.markdown(
        f"""
        <div class="hero">
          <div class="badge">⚖️ {SERVICE_NAME} · {SERVICE_SUB}</div>
          <h1>{SERVICE_SLOGAN}</h1>
          <div class="intro">{SERVICE_INTRO.format(n=f"{len(df):,}")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("전체 법안", f"{len(df):,}건")
    m2.metric("분석 국회 (제21·22대)", f"{df['AGE'].nunique()}개")
    m3.metric("소관 위원회", f"{df['CURR_COMMITTEE'].nunique()}개")
    m4.metric("가결 법안", f"{df['PROC_RESULT_CD'].str.contains('가결').sum():,}건")
    m5.metric("계류 중", f"{(df['PROC_RESULT_CD'] == '계류중').sum():,}건")

    st.write("")
    with card():
        st.markdown("#### 🔍 바로 유사 법안 찾기")
        with st.form("home_form", border=False):
            c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
            text = c1.text_input(
                "검색어", placeholder="예: 인구감소지역 기업 세제 지원", label_visibility="collapsed"
            )
            go = c2.form_submit_button("검색", type="primary", width="stretch")
        if go and text.strip():
            st.session_state["pending_query"] = text
            st.switch_page(pages["search"])

    st.write("")
    c1, c2, c3 = st.columns(3)
    with card(c1):
        st.markdown("#### 🔍 AI 유사 법안 검색")
        st.write("법안 제목이나 정책 내용을 입력하면 내용이 비슷한 법안 5건을 유사도와 함께 보여줍니다.")
        st.page_link(pages["search"], label="바로가기", icon="➡️")
    with card(c2):
        st.markdown("#### 📄 법안 통합 검색")
        st.write("제목으로 법안을 찾고, 상세 내용과 원문을 확인하고, 비슷한 법안까지 이어서 볼 수 있습니다.")
        st.page_link(pages["bills"], label="바로가기", icon="➡️")
    with card(c3):
        st.markdown("#### 📊 국회 통계 분석")
        st.write("위원회별 분포, 처리 결과, 월별 발의 추이를 조건별로 살펴보고 21대와 22대를 비교합니다.")
        st.page_link(pages["stats"], label="바로가기", icon="➡️")


def page_search():
    page_header("AI 유사 법안 검색", "법안 제목이나 정책 내용을 입력하면, 내용이 비슷한 기존 법안을 찾아드립니다.")

    # 홈에서 넘어온 검색어
    if "pending_query" in st.session_state:
        run_query(st.session_state.pop("pending_query"))

    # 공유받은 링크로 들어오면 주소의 검색어로 바로 검색
    shared = st.query_params.get("q", "")
    if shared and shared != st.session_state["active_query"]:
        run_query(shared)

    with card():
        with st.form("search_form", border=False):
            c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
            c1.text_input(
                "검색어", key="query", placeholder="예: 인구감소지역 기업 세제 지원", label_visibility="collapsed"
            )
            c2.form_submit_button("검색", type="primary", on_click=on_submit, width="stretch")

        st.pills("검색 예시", EXAMPLES, key="ex_pick", on_change=pick_chip, args=("ex_pick",))

        history = [h for h in st.session_state["history"] if h not in EXAMPLES]
        if history:
            st.pills("최근 검색어", history, key="his_pick", on_change=pick_chip, args=("his_pick",))

        # 검색 설정은 접어 두고 필요할 때 펼침
        with st.expander("검색 설정 (대수 · 위원회 · 처리 결과 · 유사도 기준)"):
            f1, f2, f3 = st.columns(3)
            ages = f1.multiselect("대수", sorted(df["AGE"].unique()), format_func=lambda a: f"제{a}대", placeholder=ALL, key="s_age")
            committees = f2.multiselect("소관 위원회", sorted(df["CURR_COMMITTEE"].unique()), placeholder=ALL, key="s_com")
            results = f3.multiselect("처리 결과", sorted(df["PROC_RESULT_CD"].unique()), placeholder=ALL, key="s_res")
            # 평가에서 10% 미만 결과는 거의 관련이 없었음
            min_score = st.slider("유사도가 이보다 낮은 법안은 숨기기(%)", 0, 50, 10, key="s_min")
            st.caption("등급: :green-badge[매우 유사] 40% 이상 · :blue-badge[유사] 20% 이상 · :orange-badge[일부 유사] 10% 이상")

    active = st.session_state["active_query"]
    if active:
        st.markdown(f"### \"{md_escape(active)}\" 검색 결과")
        mask = make_mask(df, ages, committees, results)
        result = search(active, mask, min_score)
        if result is None or len(result) == 0:
            show_empty(active, mask, min_score, key="find")
        else:
            show_share_link(active)
            show_results(result, key="find")
    else:
        st.info("검색어를 입력하거나 예시를 눌러 보세요.")


def page_bills():
    page_header("법안 통합 검색", "법안 제목에 들어간 단어로 법안을 찾고, 상세 내용과 비슷한 법안을 확인합니다.")

    with st.container(key="band"):
        c1, c2, c3, c4 = st.columns([3, 1, 2, 2])
        keyword = c1.text_input("법안 제목에 들어간 단어", value="개인정보")
        ages = c2.multiselect("대수", sorted(df["AGE"].unique()), format_func=lambda a: f"제{a}대", placeholder=ALL)
        committees = c3.multiselect("소관 위원회", sorted(df["CURR_COMMITTEE"].unique()), placeholder=ALL)
        results = c4.multiselect("처리 결과", sorted(df["PROC_RESULT_CD"].unique()), placeholder=ALL)

    view = df[make_mask(df, ages, committees, results)]
    if keyword.strip():
        view = view[view["BILL_NAME"].str.contains(keyword.strip(), regex=False)]
    found = view.sort_values("PROPOSE_DT", ascending=False)

    word = f'"{keyword.strip()}" ' if keyword.strip() else ""
    st.markdown(
        f'<div class="count-box">{word}검색 결과 <b>{len(found):,}</b> 건</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if len(found) == 0:
        st.info("조건에 맞는 법안이 없습니다. 다른 단어로 찾아보세요.")
        return

    table = found[["BILL_NAME", "대수", "PROPOSER", "PROPOSE_DT", "CURR_COMMITTEE", "PROC_RESULT_CD", "LINK_URL"]]
    table.columns = ["법안명", "대수", "제안자", "발의일", "소관 위원회", "처리 결과", "원문"]
    st.dataframe(
        table,
        hide_index=True,
        height=min(400, 38 + 35 * len(table)),
        column_config={"원문": st.column_config.LinkColumn("원문", display_text="보기")},
    )

    # 고른 법안의 상세 내용 보여주기
    labels = [f"{r.BILL_NAME} · {r.PROPOSER} · {r.PROPOSE_DT}" for r in found.itertuples()]
    pick = st.selectbox("자세히 볼 법안", range(len(found)), format_func=lambda i: labels[i])
    row = found.iloc[pick]
    with card():
        st.markdown(f":{result_color(row['PROC_RESULT_CD'])}-badge[{row['PROC_RESULT_CD']}]")
        st.markdown(f"#### {md_escape(row['BILL_NAME'])}")
        st.caption(bill_meta(row))
        st.markdown(md_escape(row["SUMMARY"]) if row["SUMMARY"] else "제안이유 및 주요내용이 없는 법안입니다.")
        st.markdown(f"[의안정보시스템에서 원문 보기]({row['LINK_URL']})")

        # 고른 법안의 제목과 본문 전체로 비슷한 법안을 찾음
        st.button("이 법안과 비슷한 법안 찾기", type="primary", on_click=find_similar, args=(row["BILL_ID"],))

    if st.session_state.get("similar_bill") == row["BILL_ID"]:
        st.markdown("### 비슷한 법안")
        result = search_by_bill(found.index[pick], 10)
        if len(result) == 0:
            st.info("유사도가 10% 이상인 법안이 없습니다.")
        else:
            show_results(result, key="bill", by_bill=True)


# 처리 결과별 고정 색. 배지 색과 맞춤
RESULT_COLORS = {
    "계류중": "#2563eb",
    "원안가결": "#16a34a",
    "수정가결": "#4ade80",
    "대안반영폐기": "#94a3b8",
    "임기만료폐기": "#64748b",
    "수정안반영폐기": "#cbd5e1",
    "폐기": "#475569",
    "철회": "#f59e0b",
}
AGE_COLORS = alt.Scale(domain=["제21대", "제22대"], range=["#93a5cf", NAVY])


def page_stats():
    page_header("국회 통계 분석", f"수집한 법안 {len(df):,}건을 위원회, 처리 결과, 발의 시기별로 살펴봅니다.")

    m1, m2, m3 = st.columns(3)
    m1.metric("전체 법안", f"{len(df):,}건")
    passed = df["PROC_RESULT_CD"].str.contains("가결").mean() * 100
    m2.metric("가결 비율", f"{passed:.1f}%")
    m3.metric("가장 많은 위원회", df["위원회"].value_counts().index[0])

    committees = sorted(df["CURR_COMMITTEE"].unique())
    results = sorted(df["PROC_RESULT_CD"].unique())
    result_scale = alt.Scale(domain=list(RESULT_COLORS), range=list(RESULT_COLORS.values()))

    left, right = st.columns(2)

    # 1. 위원회별 분포
    with card(left):
        st.markdown("#### 소관 위원회별 법안 분포")
        st.caption("어느 위원회에 법안이 많이 몰렸는지 보여줍니다. 막대가 길수록 해당 위원회가 맡은 법안이 많습니다.")
        f1, f2 = st.columns(2)
        with f1:
            age = stat_filter("대수", ["제21대", "제22대"], "c_age")
        with f2:
            res = stat_filter("처리 결과", results, "c_res")
        data = df
        if age:
            data = data[data["대수"] == age]
        if res:
            data = data[data["PROC_RESULT_CD"] == res]
        if len(data) == 0:
            st.info("조건에 맞는 법안이 없습니다.")
        else:
            counts = data["위원회"].value_counts().reset_index()
            counts.columns = ["위원회", "건수"]
            chart = (
                alt.Chart(counts)
                .mark_bar(color=NAVY, cornerRadiusEnd=3)
                .encode(
                    x=alt.X("건수", title="법안 수"),
                    y=alt.Y("위원회", sort="-x", title=None, axis=alt.Axis(labelLimit=0, labelFontSize=11)),
                    tooltip=["위원회", "건수"],
                )
                .properties(height=bar_height(len(counts)))
            )
            text = chart.mark_text(align="left", dx=4, color="#334155").encode(text="건수")
            draw(chart + text)
            merged = ", ".join(f"{old} → {new}" for old, new in COMMITTEE_MERGE.items() if " " not in old)
            st.caption(f"{len(data):,}건 기준 · 22대에서 이름이 바뀐 위원회는 바뀐 이름으로 합쳐서 셌습니다. ({merged})")

    # 2. 처리 결과 도넛
    with card(right):
        st.markdown("#### 처리 결과별 현황")
        st.caption("발의된 법안이 가결, 폐기, 계류 중 어떤 결과로 이어졌는지 비율로 보여줍니다.")
        f1, f2 = st.columns(2)
        with f1:
            age = stat_filter("대수", ["제21대", "제22대"], "r_age")
        with f2:
            com = stat_filter("소관 위원회", committees, "r_com")
        data = df
        if age:
            data = data[data["대수"] == age]
        if com:
            data = data[data["CURR_COMMITTEE"] == com]
        if len(data) == 0:
            st.info("조건에 맞는 법안이 없습니다.")
        else:
            counts = data["PROC_RESULT_CD"].value_counts().reset_index()
            counts.columns = ["처리 결과", "건수"]
            counts["비율(%)"] = (counts["건수"] / counts["건수"].sum() * 100).round(1)
            donut = (
                alt.Chart(counts)
                .mark_arc(innerRadius=75, outerRadius=130)
                .encode(
                    theta=alt.Theta("건수", stack=True),
                    color=alt.Color("처리 결과", scale=result_scale, legend=alt.Legend(title=None, orient="right")),
                    order=alt.Order("건수", sort="descending"),
                    tooltip=["처리 결과", "건수", "비율(%)"],
                )
                .properties(height=300)
            )
            center = alt.Chart(pd.DataFrame({"t": [f"{len(data):,}건"]})).mark_text(
                fontSize=22, fontWeight="bold", color="#0f1f4b"
            ).encode(text="t")
            draw(donut + center)

            table = counts.copy()
            table["비율(%)"] = table["비율(%)"].map(lambda v: f"{v:.1f}%")
            st.dataframe(table, hide_index=True)
            st.caption("대안반영폐기는 다른 법안과 합쳐져 위원회 대안으로 반영된 경우입니다.")

    left, right = st.columns(2)

    # 3. 월별 발의 꺾은선
    with card(left):
        st.markdown("#### 월별 법안 발의 추이")
        st.caption("매달 몇 건의 법안이 발의됐는지 보여줍니다. 색으로 21대와 22대를 구분했습니다.")
        com = stat_filter("소관 위원회", committees, "m_com")
        data = df if not com else df[df["CURR_COMMITTEE"] == com]
        if len(data) == 0:
            st.info("조건에 맞는 법안이 없습니다.")
        else:
            counts = data.groupby(["MONTH", "대수"]).size().reset_index(name="건수")
            counts["월"] = pd.to_datetime(counts["MONTH"])
            line = (
                alt.Chart(counts)
                .mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=2)
                .encode(
                    x=alt.X("월", title="발의 월", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
                    y=alt.Y("건수", title="법안 수"),
                    color=alt.Color("대수", scale=AGE_COLORS, legend=alt.Legend(title=None, orient="top")),
                    tooltip=["MONTH", "대수", "건수"],
                )
                .properties(height=320)
            )
            draw(line)
            st.caption("수집한 표본 기준이라 실제 국회 전체 발의 건수와는 다릅니다.")

    # 4. 21대와 22대 비교
    with card(right):
        st.markdown("#### 제21대와 제22대 비교")
        st.caption("두 국회의 처리 결과 비율을 나란히 비교합니다. 22대는 임기 중이라 계류 중인 법안이 많습니다.")
        com = stat_filter("소관 위원회", committees, "v_com")
        data = df if not com else df[df["CURR_COMMITTEE"] == com]
        if len(data) == 0:
            st.info("조건에 맞는 법안이 없습니다.")
        else:
            counts = data.groupby(["대수", "PROC_RESULT_CD"]).size().reset_index(name="건수")
            counts["비율(%)"] = (counts["건수"] / counts.groupby("대수")["건수"].transform("sum") * 100).round(1)
            chart = (
                alt.Chart(counts)
                .mark_bar()
                .encode(
                    x=alt.X("비율(%)", title="비율(%)"),
                    y=alt.Y("PROC_RESULT_CD", title=None, sort="-x", axis=alt.Axis(labelLimit=0)),
                    yOffset="대수",
                    color=alt.Color("대수", scale=AGE_COLORS, legend=alt.Legend(title=None, orient="top")),
                    tooltip=["대수", "PROC_RESULT_CD", "건수", "비율(%)"],
                )
                .properties(height=max(320, counts["PROC_RESULT_CD"].nunique() * 40))
            )
            draw(chart)


def page_about():
    page_header("프로젝트 소개", "POLICY INSIGHT가 어떤 자료로, 어떤 방법으로 유사 법안을 찾는지 소개합니다.")

    with card():
        st.markdown("#### 왜 만들었나")
        st.write(
            "국회에는 같은 문제를 다루는 법안이 여러 의원에게서 반복해서 발의됩니다. "
            "새 법안을 준비하거나 정책을 살펴볼 때, 비슷한 법안이 이미 있었는지와 그 법안이 어떻게 처리됐는지를 "
            "빠르게 확인할 수 있도록 유사 법안 검색 서비스를 만들었습니다."
        )

    with card():
        st.markdown("#### 데이터")
        st.write(
            f"열린국회정보 Open API에서 제21대 국회 1,000건, 제22대 국회 1,000건, 총 {len(df):,}건의 법안을 수집했습니다. "
            "각 대수의 전체 기간에서 고르게 뽑았고, 법안 목록에 제안이유 및 주요내용을 합쳐서 사용했습니다."
        )

    with card():
        st.markdown("#### 분석 방법")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**1. 형태소 분석**  \nKiwi로 제목과 본문에서 명사를 뽑고, 붙여 쓴 단어(전기차, 학교폭력)도 함께 남깁니다.")
        c2.markdown("**2. TF-IDF**  \n법안마다 단어의 중요도를 계산해 벡터로 바꿉니다.")
        c3.markdown("**3. 코사인 유사도**  \n검색어와 각 법안 벡터가 얼마나 비슷한지 계산합니다.")
        c4.markdown("**4. 단어 포함 비율**  \n검색어 단어를 많이 가진 법안에 더 높은 점수를 줍니다.")

    with card():
        st.markdown("#### 검색 정확도 평가")
        st.write(
            "검색어 20개에 대해 각 방식이 찾은 상위 5건이 실제로 관련 있는 법안인지 팀원이 직접 판정했습니다. "
            "판정 기준은 검색어가 다루는 정책 문제를 그 법안이 직접 다루는지입니다."
        )
        score = pd.DataFrame(
            {
                "방식": ["제목만 사용", "제목 + 본문", "제목 + 본문 + 합성어 + 포함 비율 (현재)"],
                "상위 5건 정확도": ["28%", "53%", "64%"],
                "1위 정확도": ["47%", "90%", "85%"],
            }
        )
        st.dataframe(score, hide_index=True)
        st.caption("현재 방식은 같은 검색어 20개로 고른 설정이라, 새로운 검색어에서는 정확도가 이보다 낮을 수 있습니다.")

    with card():
        st.markdown("#### 한계와 앞으로")
        st.write(
            "단어가 일치해야 찾을 수 있어서 \"응급실\"과 \"응급의료기관\"처럼 표현이 다르면 찾지 못합니다. "
            "문장의 뜻을 비교하는 의미 검색을 더하면 보완할 수 있습니다. 또한 2,000건 표본이라 실제로 있는 "
            "비슷한 법안이 검색되지 않을 수 있습니다."
        )

    with card():
        st.markdown("#### 만든 사람")
        st.write(TEAM)
        st.markdown(f"[GitHub 저장소]({GITHUB_URL})")


# ---------------- 실행 ----------------

df = load_data()
docs = tuple(df["TITLE_WORDS"] + " " + df["SUMMARY_WORDS"])
vectorizer, matrix = build_vectors(docs)
word_sets = build_word_sets(docs)
body_matrix = build_body_vectors(tuple(df["SUMMARY_WORDS"]))
terms = vectorizer.get_feature_names_out()

st.session_state.setdefault("query", "")
st.session_state.setdefault("active_query", "")
st.session_state.setdefault("history", [])

pages = {
    "home": st.Page(page_home, title="홈", icon="🏠", default=True),
    "search": st.Page(page_search, title="AI 유사 법안 검색", icon="🔍", url_path="search"),
    "bills": st.Page(page_bills, title="법안 통합 검색", icon="📄", url_path="bills"),
    "stats": st.Page(page_stats, title="국회 통계 분석", icon="📊", url_path="stats"),
    "about": st.Page(page_about, title="프로젝트 소개", icon="ℹ️", url_path="about"),
}

nav = st.navigation({f"⚖️ {SERVICE_NAME}": list(pages.values())})

with st.sidebar:
    st.caption(SERVICE_SUB)
    st.divider()
    st.caption(f"자료: 열린국회정보 Open API  \n제21·22대 국회 법안 {len(df):,}건")
    st.caption(TEAM)

nav.run()

st.divider()
st.caption(
    f"{SERVICE_NAME} · 자료: 열린국회정보 Open API (발의일 {df['PROPOSE_DT'].min()} ~ {df['PROPOSE_DT'].max()}) · "
    f"{TEAM} · [GitHub]({GITHUB_URL})"
)
