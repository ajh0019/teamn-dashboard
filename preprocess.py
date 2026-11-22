import re
import sys
import pandas as pd
from kiwipiepy import Kiwi

kiwi = Kiwi()

# 제목마다 붙는 말이라 구분에 도움이 안 되는 단어
STOPWORDS = {"법률", "관한", "법안", "위한"}


NOUN_TAGS = ("NNG", "NNP", "SL")


# 명사와 영문 단어만 뽑기
# compound가 True면 붙여 쓴 명사를 합친 단어도 같이 남김 (전기차, 학교폭력 등)
def extract_words(text, compound=True):
    if not isinstance(text, str):
        return ""
    words = []
    run = []
    last_end = -1
    for token in kiwi.tokenize(text) + [None]:
        is_noun = token is not None and (token.tag in NOUN_TAGS or token.tag == "XPN")

        # 띄어쓰기 없이 이어진 명사가 끝나면 합친 단어를 추가
        if not is_noun or token.start != last_end:
            if compound and len(run) >= 2:
                words.append("".join(t.form for t in run))
            run = []
        if not is_noun:
            last_end = -1
            continue

        run.append(token)
        last_end = token.start + token.len
        if token.tag != "XPN" and len(token.form) > 1:
            words.append(token.form)
    return " ".join(w for w in words if w not in STOPWORDS)


# 제목 끝의 일부개정법률안 같은 꼬리말과 괄호 제거
def clean_title(title):
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"(일부개정법률안|전부개정법률안|법률안|법안)\s*$", "", title)
    return title.strip()


# 22대 국회에서 이름이 바뀐 위원회는 바뀐 이름으로 통일
COMMITTEE_MERGE = {
    "환경노동위원회": "기후에너지환경노동위원회",
    "기획재정위원회": "재정경제기획위원회",
    "여성가족위원회": "성평등가족위원회",
    "정치개혁 특별위원회": "정치개혁특별위원회",
}


# 본문 맨 앞의 제안이유 및 주요내용 머리말 제거
def clean_summary(summary):
    return re.sub(r"^(\d+\.\s*)?(대안의 )?제안이유( 및 주요내용)?", "", summary).strip()


def preprocess(df):
    df = df.drop_duplicates(subset=["BILL_ID"]).reset_index(drop=True)

    # 값이 전부 비어 있는 열은 버림
    df = df.dropna(axis=1, how="all")

    df["CURR_COMMITTEE"] = df["CURR_COMMITTEE"].fillna("미지정")
    df["ORIG_COMMITTEE"] = df["CURR_COMMITTEE"]
    df["CURR_COMMITTEE"] = df["CURR_COMMITTEE"].replace(COMMITTEE_MERGE)
    df["PROC_RESULT_CD"] = df["PROC_RESULT_CD"].fillna("계류중")
    df["SUMMARY"] = df["SUMMARY"].fillna("")

    # 위원장이 낸 법안은 여러 법안을 합친 대안
    df["IS_ALT"] = df["PROPOSER_KIND"] == "위원장"
    df["MONTH"] = df["PROPOSE_DT"].str[:7]

    df["TITLE_WORDS"] = df["BILL_NAME"].apply(clean_title).apply(extract_words)
    df["SUMMARY_WORDS"] = df["SUMMARY"].apply(clean_summary).apply(extract_words)
    return df


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    result = preprocess(pd.read_csv(in_path))
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(len(result), "건 저장:", out_path)
