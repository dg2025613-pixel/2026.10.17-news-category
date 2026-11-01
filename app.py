import streamlit as st
import pandas as pd
import requests
import trafilatura
import re

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="AI 뉴스 장르 분류 스튜디오",
    page_icon="📰",
    layout="wide"
)

GITHUB_EXCEL_PATH = "data/news_data.xlsx"

# KNN 이웃 수 변경
K_NEIGHBORS = 5


# ==========================================
# 2. 화면 디자인
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
    background-color: #f3f7fc;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.news-header {
    background: linear-gradient(120deg, #173b68, #3278b9);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.news-header h1 {
    color: white;
    font-size: 32px;
    font-weight: 800;
}

.news-header p {
    color: #e4efff;
    font-size: 15px;
}

.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e1eaf5;
    margin-bottom: 15px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dce7f3;
    text-align: center;
}

.result-label {
    color: #6b7c93;
    font-size: 15px;
}

.result-value {
    color: #1e5b96;
    font-size: 32px;
    font-weight: 800;
    margin-top: 10px;
}

.small-text {
    color: #718096;
    font-size: 13px;
}

h2, h3 {
    color: #203b5c;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e1eaf5;
}

.stButton > button {
    background-color: #2469a6;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #174d80;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 제목
# ==========================================

st.markdown("""
<div class="news-header">

    <h1>📰 AI NEWS CLASSIFICATION STUDIO</h1>

    <p>
    TF-IDF와 KNN 알고리즘을 활용한 인공지능 뉴스 장르 분류 시스템
    </p>

    <p>
    뉴스 기사를 입력하면 AI가 내용을 분석하여 장르를 예측합니다.
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. 엑셀 데이터 불러오기
# ==========================================

def load_github_data():

    file_path = Path(GITHUB_EXCEL_PATH)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{GITHUB_EXCEL_PATH} 파일을 찾을 수 없습니다."
        )

    df = pd.read_excel(
        file_path,
        engine="openpyxl"
    )

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    rename_dict = {
        "category": "genre",
        "label": "genre",
        "분류": "genre",
        "장르": "genre",

        "headline": "title",
        "제목": "title",

        "content": "text",
        "body": "text",
        "본문": "text",
        "내용": "text",
        "article": "text"
    }

    df = df.rename(columns=rename_dict)

    required_columns = ["genre", "title", "text"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"필수 열 {col}이 없습니다."
            )

    df = df[required_columns].copy()

    for col in required_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = df[
        (df["genre"] != "") &
        (df["title"] != "") &
        (df["text"] != "")
    ]

    return df


# ==========================================
# 5. 학습 데이터 정리
# ==========================================

def clean_training_data(df):

    df = df.copy()

    df["title_key"] = (
        df["title"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    genre_counts = (
        df.groupby("title_key")["genre"]
        .nunique()
    )

    conflicting_titles = genre_counts[
        genre_counts > 1
    ].index.tolist()

    df = df[
        ~df["title_key"].isin(conflicting_titles)
    ].copy()

    df = df.drop_duplicates(
        subset=["title_key"],
        keep="first"
    )

    df = df.drop(columns=["title_key"])

    return df


# ==========================================
# 6. 학습 데이터 불러오기 및 정리
# ==========================================

try:

    original_df = load_github_data()

    df = clean_training_data(original_df)

except Exception as e:

    st.error(
        f"학습 데이터를 불러오지 못했습니다: {e}"
    )

    st.stop()


if len(df) < K_NEIGHBORS:

    st.error("학습 데이터가 부족합니다.")

    st.stop()


# ==========================================
# 7. 학습 데이터 현황
# ==========================================

st.subheader("📊 학습 데이터 현황")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "원본 데이터 수",
        f"{len(original_df)}개"
    )

with col2:

    st.metric(
        "사용 중인 학습 파일",
        "news_data.xlsx"
    )


# ==========================================
# 8. TF-IDF 벡터화
# ==========================================

X = (
    df["title"] + " " + df["text"]
).tolist()

y = df["genre"].tolist()


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    sublinear_tf=True,
    max_features=50000
)

X_tfidf = vectorizer.fit_transform(X)


# ==========================================
# 9. KNN 모델 학습
# ==========================================

model = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    weights="distance",
    metric="cosine",
    algorithm="brute"
)

model.fit(
    X_tfidf,
    y
)


# ==========================================
# 10. URL에서 뉴스 추출
# ==========================================

def extract_article(url):

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "올바른 뉴스 URL을 입력해 주세요."
        )

    downloaded = trafilatura.fetch_url(url)

    if not downloaded:

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            },
            timeout=15
        )

        response.raise_for_status()

        downloaded = response.text

    article_text = trafilatura.extract(
        downloaded,
        url=url,
        include_comments=False,
        include_tables=False
    )

    if not article_text:

        raise ValueError(
            "기사 본문을 추출하지 못했습니다. "
            "기사 본문을 직접 입력해 주세요."
        )

    metadata = trafilatura.extract_metadata(
        downloaded
    )

    article_title = ""

    if metadata and metadata.title:
        article_title = metadata.title

    return article_title, article_text


# ==========================================
# 11. 뉴스 입력 화면
# ==========================================

st.markdown("---")

st.subheader("🔍 뉴스 장르 분석")

input_method = st.radio(
    "뉴스 입력 방식",
    ["기사 본문 직접 입력", "뉴스 URL 입력"],
    horizontal=True
)


article_title = ""
article_text = ""


with st.form("news_form"):

    if input_method == "기사 본문 직접 입력":

        article_title = st.text_input(
            "뉴스 제목",
            placeholder="뉴스 제목을 입력하세요."
        )

        article_text = st.text_area(
            "뉴스 본문",
            height=250,
            placeholder="분류하고 싶은 뉴스 기사의 본문을 입력하세요."
        )

    else:

        article_url = st.text_input(
            "뉴스 URL",
            placeholder="https://..."
        )

    predict_button = st.form_submit_button(
        "🔎 뉴스 장르 분석하기",
        use_container_width=True
    )


# ==========================================
# 12. 뉴스 분석
# ==========================================

if predict_button:

    try:

        if input_method == "뉴스 URL 입력":

            if not article_url.strip():

                st.error(
                    "뉴스 URL을 입력해 주세요."
                )

                st.stop()

            with st.spinner(
                "뉴스 기사를 가져오는 중입니다..."
            ):

                article_title, article_text = extract_article(
                    article_url
                )

        if not article_text.strip():

            st.error(
                "뉴스 본문을 입력해 주세요."
            )

            st.stop()

        prediction_input = (
            article_title + " " + article_text
        ).strip()

        input_vector = vectorizer.transform(
            [prediction_input]
        )

        prediction = model.predict(
            input_vector
        )[0]

        distances, indices = model.kneighbors(
            input_vector,
            n_neighbors=K_NEIGHBORS
        )

        similarities = 1 - distances[0]


        # ==================================
        # 13. 예측 결과
        # ==================================

        st.markdown("---")

        st.subheader("🎯 AI 분석 결과")

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
            AI가 예측한 뉴스 장르
            </div>

            <div class="result-value">
            {prediction}
            </div>

        </div>
        """, unsafe_allow_html=True)


        st.write("")

        st.info(
            "TF-IDF로 뉴스의 특징을 수치화한 뒤 "
            "KNN 알고리즘으로 가장 가까운 학습 데이터 "
            "5개를 비교하여 장르를 예측했습니다."
        )


        # ==================================
        # 14. 유사한 뉴스 표시
        # ==================================

        st.subheader("📚 가장 유사한 학습 뉴스")

        neighbor_rows = []

        for rank, (idx, similarity) in enumerate(
            zip(indices[0], similarities),
            start=1
        ):

            neighbor_rows.append({

                "순위": rank,

                "예측 장르": df.iloc[idx]["genre"],

                "뉴스 제목": df.iloc[idx]["title"],

                "유사도": f"{similarity * 100:.2f}%"

            })


        neighbor_df = pd.DataFrame(
            neighbor_rows
        )

        st.dataframe(
            neighbor_df,
            use_container_width=True,
            hide_index=True
        )


        # ==================================
        # 15. 뉴스 본문 확인
        # ==================================

        with st.expander(
            "분석에 사용된 뉴스 본문 확인"
        ):

            if article_title:

                st.markdown(
                    f"**제목:** {article_title}"
                )

            st.write(article_text)


    except Exception as e:

        st.error(
            f"뉴스 분석 중 오류가 발생했습니다: {e}"
        )


# ==========================================
# 16. 하단 안내
# ==========================================

st.markdown("---")

st.markdown("""
<div class="info-card">

    <h3>🤖 AI 뉴스 분류 시스템 안내</h3>

    <p>
    본 시스템은 TF-IDF와 KNN 알고리즘을 이용하여
    뉴스의 장르를 예측합니다.
    </p>

    <p>
    TF-IDF는 단어 및 문자 조합의 중요도를 수치화하며,
    KNN은 입력된 뉴스와 유사한 학습 데이터를 찾아
    가장 적절한 장르를 예측합니다.
    </p>

    <p class="small-text">
    예측 결과는 학습 데이터의 품질과
    뉴스 내용에 따라 달라질 수 있습니다.
    </p>

</div>
""", unsafe_allow_html=True)
