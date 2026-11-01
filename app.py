import streamlit as st
import pandas as pd
import requests
import trafilatura

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="AI 뉴스 장르 분류 스튜디오",
    page_icon="📰",
    layout="wide"
)

GITHUB_EXCEL_PATH = Path("data/news_data.xlsx")
K_NEIGHBORS = 5

st.title("📰 AI NEWS CLASSIFICATION STUDIO")
st.caption("TF-IDF와 KNN 알고리즘을 활용한 인공지능 뉴스 장르 분류 시스템")
st.write("뉴스 기사를 입력하면 AI가 내용을 분석하여 장르를 예측합니다.")
st.divider()


def load_github_data():
    if not GITHUB_EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 파일을 찾을 수 없습니다: {GITHUB_EXCEL_PATH}"
        )

    data = pd.read_excel(
        GITHUB_EXCEL_PATH,
        engine="openpyxl"
    )

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
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

    data = data.rename(columns=rename_dict)

    required_columns = ["genre", "title", "text"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "엑셀 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data[
        (data["genre"] != "")
        & (data["title"] != "")
        & (data["text"] != "")
    ].copy()

    if data.empty:
        raise ValueError("엑셀 파일에 학습할 데이터가 없습니다.")

    return data


def clean_training_data(data):
    cleaned = data.copy()

    cleaned["title_key"] = (
        cleaned["title"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    genre_counts = cleaned.groupby("title_key")["genre"].nunique()

    conflicting_titles = genre_counts[
        genre_counts > 1
    ].index

    cleaned = cleaned[
        ~cleaned["title_key"].isin(conflicting_titles)
    ].copy()

    cleaned = cleaned.drop_duplicates(
        subset=["title_key", "genre"],
        keep="first"
    )

    cleaned = cleaned.drop(columns=["title_key"])

    return cleaned.reset_index(drop=True)


try:
    original_df = load_github_data()
    training_df = clean_training_data(original_df)

    if training_df.empty:
        raise ValueError("학습 데이터가 남아 있지 않습니다.")

    if len(training_df) < K_NEIGHBORS:
        raise ValueError("학습 데이터가 5개보다 적습니다.")

    training_text = (
        training_df["title"] + " " + training_df["text"]
    ).tolist()

    training_labels = training_df["genre"].tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        max_features=50000
    )

    training_vectors = vectorizer.fit_transform(training_text)

    model = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="distance",
        metric="cosine",
        algorithm="brute"
    )

    model.fit(training_vectors, training_labels)

except Exception as error:
    st.error(f"학습 데이터를 불러오거나 모델을 학습하지 못했습니다: {error}")
    st.stop()


st.subheader("📊 학습 데이터 현황")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("원본 데이터 수", f"{len(original_df):,}개")

with metric_col2:
    st.metric("사용 중인 학습 파일", "news_data.xlsx")

st.divider()


def extract_article(url):
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError("올바른 뉴스 URL을 입력해 주세요.")

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
            timeout=20
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
            "기사 본문을 자동으로 가져오지 못했습니다. "
            "본문 직접 입력 방식으로 시도해 주세요."
        )

    metadata = trafilatura.extract_metadata(downloaded)

    article_title = ""

    if metadata and metadata.title:
        article_title = metadata.title

    return article_title, article_text


st.subheader("🔍 뉴스 장르 분석")

input_method = st.radio(
    "뉴스 입력 방식",
    ["기사 본문 직접 입력", "뉴스 URL 입력"],
    horizontal=True
)

with st.form("news_form"):
    article_title_input = ""
    article_text_input = ""
    article_url_input = ""

    if input_method == "기사 본문 직접 입력":
        article_title_input = st.text_input(
            "뉴스 제목",
            placeholder="뉴스 제목을 입력하세요."
        )

        article_text_input = st.text_area(
            "뉴스 본문",
            height=250,
            placeholder="분류하고 싶은 뉴스 기사의 본문을 입력하세요."
        )

    else:
        article_url_input = st.text_input(
            "뉴스 URL",
            placeholder="https://..."
        )

    predict_button = st.form_submit_button(
        "🔎 뉴스 장르 분석하기",
        use_container_width=True
    )


if predict_button:
    try:
        if input_method == "뉴스 URL 입력":
            if not article_url_input.strip():
                st.error("뉴스 URL을 입력해 주세요.")
                st.stop()

            with st.spinner("뉴스 기사를 가져오는 중입니다..."):
                article_title, article_text = extract_article(
                    article_url_input
                )
        else:
            article_title = article_title_input.strip()
            article_text = article_text_input.strip()

        if not article_text:
            st.error("뉴스 본문을 입력해 주세요.")
            st.stop()

        prediction_text = (
            article_title + " " + article_text
        ).strip()

        input_vector = vectorizer.transform([prediction_text])

        prediction = model.predict(input_vector)[0]

        distances, indices = model.kneighbors(
            input_vector,
            n_neighbors=K_NEIGHBORS
        )

        similarities = 1 - distances[0]

        st.divider()
        st.subheader("🎯 AI 분석 결과")

        st.success(f"예측된 뉴스 장르: **{prediction}**")

        st.info(
            f"TF-IDF로 뉴스의 특징을 수치화하고, "
            f"KNN이 가장 가까운 학습 데이터 {K_NEIGHBORS}개를 비교해 "
            "장르를 예측했습니다. 유사도는 정답 확률을 뜻하지 않습니다."
        )

        st.subheader("📚 가장 유사한 학습 뉴스")

        neighbor_rows = []

        for rank, (index, similarity) in enumerate(
            zip(indices[0], similarities),
            start=1
        ):
            neighbor_rows.append({
                "순위": rank,
                "학습 데이터 장르": training_df.iloc[index]["genre"],
                "뉴스 제목": training_df.iloc[index]["title"],
                "유사도": f"{max(0, min(1, similarity)) * 100:.2f}%"
            })

        neighbor_df = pd.DataFrame(neighbor_rows)

        st.dataframe(
            neighbor_df,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("분석에 사용된 뉴스 본문 확인"):
            if article_title:
                st.write(f"**제목:** {article_title}")

            st.write(article_text)

    except Exception as error:
        st.error(f"뉴스 분석 중 오류가 발생했습니다: {error}")


st.divider()
st.subheader("🤖 AI 뉴스 분류 시스템 안내")

st.write(
    "본 시스템은 TF-IDF와 KNN 알고리즘을 이용하여 뉴스의 장르를 예측합니다."
)

st.write(
    "TF-IDF는 뉴스 제목과 본문에 포함된 문자 조합의 중요도를 수치화하고, "
    "KNN은 입력 뉴스와 가까운 학습 데이터를 찾아 장르를 예측합니다."
)

st.caption(
    "예측 결과는 학습 데이터의 품질과 뉴스 내용에 따라 달라질 수 있습니다."
)
