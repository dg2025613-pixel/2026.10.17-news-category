
import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests
import streamlit as st
import trafilatura

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# -----------------------------
# 1. 앱 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뉴스 장르 분류 AI",
    page_icon="📰",
    layout="wide"
)

st.title("📰 AI 뉴스 장르 분류기")
st.write(
    "뉴스 원문 URL을 입력하면 TF-IDF와 KNN을 이용해 "
    "뉴스의 장르를 분류합니다."
)

GENRES = ["정치", "경제", "사회", "문화", "스포츠", "IT/과학"]


# -----------------------------
# 2. 테스트용 예시 학습 데이터
# 실제 실험에서는 충분한 뉴스 데이터로 교체하세요.
# -----------------------------
SAMPLE_DATA = [
    ("정치", "국회에서 예산안을 논의했다", "국회 예산안 심의 과정에서 여야 의원들이 정책과 예산 편성 방향을 논의했다."),
    ("정치", "정부의 새로운 정책 발표", "정부는 관계 부처 회의를 열고 주요 정책의 추진 일정과 세부 계획을 발표했다."),
    ("정치", "지방자치단체 정책 협의", "정당 관계자들은 지역 현안과 지방 행정 정책에 대한 의견을 교환했다."),

    ("경제", "국내 물가 동향 발표", "소비자물가와 생활필수품 가격의 변화가 가계 지출과 소비 심리에 영향을 미치고 있다."),
    ("경제", "기업 실적과 주식시장", "기업들의 분기 실적 발표 이후 주식시장에서는 매출과 영업이익 전망에 관심이 모였다."),
    ("경제", "금리와 가계 대출 변화", "금리 변동과 은행의 대출 조건이 가계의 이자 부담과 금융시장에 영향을 주고 있다."),

    ("사회", "지역 교통 안전 대책", "지자체는 보행자 사고를 줄이기 위해 교통 시설을 점검하고 안전 대책을 마련했다."),
    ("사회", "학교 교육 환경 개선", "교육 당국은 학교 시설과 학생들의 학습 환경을 개선하기 위한 지원 계획을 검토했다."),
    ("사회", "지역 주민 복지 사업", "지역 사회에서는 취약계층 지원과 복지 서비스 확대를 위한 사업이 진행되고 있다."),

    ("문화", "미술 전시회 관람객 증가", "새로운 미술 전시회가 열리면서 관람객들이 작품과 작가의 예술적 표현을 감상했다."),
    ("문화", "지역 문화 축제 개최", "지역 문화 축제에서는 전통 공연과 예술 프로그램이 열려 다양한 문화 콘텐츠를 소개했다."),
    ("문화", "영화제에서 신작 공개", "영화제에서 여러 감독의 신작이 공개됐으며 관객과 평론가들이 작품의 연출을 평가했다."),

    ("스포츠", "축구 경기에서 승리", "축구팀은 경기 후반 득점에 성공하며 상대 팀을 제치고 중요한 승리를 거뒀다."),
    ("스포츠", "프로야구 선수 기록 경신", "프로야구 선수는 경기에서 안타와 타점을 기록하며 팀의 공격을 이끌었다."),
    ("스포츠", "국제 대회 결승 진출", "선수단은 준결승 경기에서 좋은 경기력을 보여 국제 대회 결승전에 진출했다."),

    ("IT/과학", "인공지능 기술 연구 발표", "연구진은 인공지능 모델의 학습 성능을 개선하는 새로운 알고리즘을 발표했다."),
    ("IT/과학", "우주 탐사 연구 성과", "과학자들은 우주 탐사 자료를 분석해 행성과 우주 환경에 관한 새로운 정보를 확인했다."),
    ("IT/과학", "반도체 기술 개발", "연구팀은 반도체의 처리 속도와 에너지 효율을 개선하기 위한 새로운 기술을 연구했다."),
]

SAMPLE_DF = pd.DataFrame(
    SAMPLE_DATA,
    columns=["genre", "title", "text"]
)


# -----------------------------
# 3. 학습 데이터 선택
# -----------------------------
st.subheader("1. 학습 데이터")

st.caption(
    "CSV 열 이름은 genre, title, text로 설정하세요. "
    "장르에는 정치, 경제, 사회, 문화, 스포츠, IT/과학을 사용합니다."
)

uploaded_file = st.file_uploader(
    "학습용 뉴스 CSV 파일 (선택 사항)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        train_df = pd.read_csv(uploaded_file)
        required = {"genre", "title", "text"}

        if not required.issubset(train_df.columns):
            st.error("CSV에 genre, title, text 열이 모두 있어야 합니다.")
            st.stop()

        train_df = train_df[["genre", "title", "text"]].copy()
        train_df = train_df.dropna()

    except Exception as e:
        st.error(f"CSV를 읽을 수 없습니다: {e}")
        st.stop()
else:
    train_df = SAMPLE_DF.copy()
    st.info(
        "현재 테스트용 예시 데이터로 실행 중입니다. "
        "실제 분류 성능을 확인하려면 장르별 뉴스 데이터로 교체하세요."
    )

train_df["genre"] = train_df["genre"].astype(str).str.strip()
train_df["title"] = train_df["title"].astype(str).str.strip()
train_df["text"] = train_df["text"].astype(str).str.strip()

train_df["content"] = (
    train_df["title"] + " " + train_df["text"]
).str.strip()

train_df = train_df[
    train_df["genre"].isin(GENRES)
    & (train_df["content"].str.len() > 0)
].copy()

if train_df.empty:
    st.error("유효한 학습 데이터가 없습니다.")
    st.stop()

genre_counts = train_df["genre"].value_counts()

st.write("장르별 학습 데이터 수")
st.dataframe(
    genre_counts.rename_axis("장르").reset_index(name="기사 수"),
    use_container_width=True,
    hide_index=True
)

if train_df["genre"].nunique() < 2:
    st.error("학습 데이터에는 최소 2개 이상의 서로 다른 장르가 필요합니다.")
    st.stop()


# -----------------------------
# 4. TF-IDF + KNN 모델 학습
# -----------------------------
max_neighbors = min(5, len(train_df))

with st.sidebar:
    st.header("모델 설정")
    k = st.slider(
        "KNN의 이웃 수 (K)",
        min_value=1,
        max_value=max_neighbors,
        value=min(3, max_neighbors)
    )

    st.caption(
        "K가 작으면 소수의 유사 기사에, "
        "K가 크면 더 많은 유사 기사에 영향을 받습니다."
    )

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True,
            max_features=50000
        )
    ),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="cosine",
            algorithm="brute"
        )
    )
])

try:
    model.fit(train_df["content"], train_df["genre"])
except Exception as e:
    st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
    st.stop()

st.success("TF-IDF와 KNN 모델 학습이 완료되었습니다.")


# -----------------------------
# 5. 안전한 URL 검사 및 본문 추출
# -----------------------------
def validate_public_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("http 또는 https 주소를 입력하세요.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("올바른 뉴스 URL을 입력하세요.")

    host = parsed.hostname

    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError("사이트 주소를 확인할 수 없습니다.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("공개 웹사이트 주소만 사용할 수 있습니다.")

    return url


def extract_article(url):
    current_url = validate_public_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsGenreClassifier/1.0)"
    }

    with requests.Session() as session:
        for _ in range(5):
            validate_public_url(current_url)

            response = session.get(
                current_url,
                headers=headers,
                timeout=15,
                allow_redirects=False
            )

            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("사이트의 이동 주소를 확인할 수 없습니다.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                raise ValueError("HTML 형식의 뉴스 페이지가 아닙니다.")

            if len(response.content) > 5_000_000:
                raise ValueError("페이지 크기가 너무 큽니다.")

            html = response.content.decode(
                response.encoding or "utf-8",
                errors="replace"
            )

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True
            )

            if not text or len(text.strip()) < 100:
                raise ValueError(
                    "기사 본문을 추출하지 못했습니다. "
                    "사이트의 접근 제한 여부를 확인하거나 기사 내용을 직접 입력하세요."
                )

            return text.strip()

        raise ValueError("페이지 이동 횟수가 너무 많습니다.")


# -----------------------------
# 6. 뉴스 장르 분류 화면
# -----------------------------
st.divider()
st.subheader("2. 분류할 뉴스 입력")

input_mode = st.radio(
    "입력 방식",
    ["뉴스 URL", "기사 본문 직접 입력"],
    horizontal=True
)

article_text = ""

if input_mode == "뉴스 URL":
    url = st.text_input(
        "뉴스 원문 URL",
        placeholder="https://news.example.com/article/..."
    )

    if st.button("뉴스 분석하기", type="primary"):
        if not url.strip():
            st.warning("뉴스 URL을 입력하세요.")
        else:
            try:
                with st.spinner("뉴스 본문을 가져오는 중입니다..."):
                    article_text = extract_article(url.strip())
                st.session_state["article_text"] = article_text
                st.session_state["article_url"] = url.strip()
            except Exception as e:
                st.error(f"뉴스를 가져오지 못했습니다: {e}")

else:
    manual_text = st.text_area(
        "뉴스 제목과 본문",
        height=220,
        placeholder="뉴스 제목과 본문을 붙여 넣으세요."
    )

    if st.button("입력한 기사 분석하기", type="primary"):
        if len(manual_text.strip()) < 30:
            st.warning("분류를 위해 기사 내용을 조금 더 입력하세요.")
        else:
            st.session_state["article_text"] = manual_text.strip()
            st.session_state["article_url"] = ""

# URL 추출 결과 또는 직접 입력 결과를 이용해 분류
if "article_text" in st.session_state:
    article_text = st.session_state["article_text"]

    st.markdown("### 추출된 기사 내용")
    st.text_area(
        "기사 본문 확인",
        value=article_text,
        height=180,
        disabled=True
    )

    if st.button("이 본문으로 장르 분류", type="primary"):
        try:
            predicted_genre = model.predict([article_text])[0]

            probabilities = model.predict_proba([article_text])[0]
            classes = model.named_steps["knn"].classes_

            results = pd.DataFrame({
                "장르": classes,
                "비율(%)": probabilities * 100
            }).sort_values("비율(%)", ascending=False)

            st.divider()
            st.subheader("3. 분류 결과")

            st.metric("예측된 뉴스 장르", predicted_genre)

            st.caption(
                "아래 비율은 KNN의 이웃 가중치를 바탕으로 계산한 값이며, "
                "실제 정답일 확률이나 검증된 정확도를 의미하지 않습니다."
            )

            st.dataframe(
                results.style.format({"비율(%)": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                results.set_index("장르")["비율(%)"]
            )

            st.subheader("분류에 사용된 유사 뉴스")
            vectorizer = model.named_steps["tfidf"]
            knn = model.named_steps["knn"]

            query_vector = vectorizer.transform([article_text])
            distances, indices = knn.kneighbors(query_vector)

            similar_rows = train_df.iloc[indices[0]].copy()
            similar_rows["유사도(%)"] = (
                1 - distances[0]
            ).clip(0, 1) * 100

            st.dataframe(
                similar_rows[["genre", "title", "유사도(%)"]]
                .rename(columns={
                    "genre": "장르",
                    "title": "학습 뉴스 제목"
                })
                .assign(
                    **{"유사도(%)": lambda df: df["유사도(%)"].round(1)}
                ),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"분류 중 오류가 발생했습니다: {e}")

st.caption(
    "참고: 뉴스 사이트의 이용약관, robots 정책 및 접근 제한을 준수하세요. "
    "일부 사이트는 자동 본문 추출을 허용하지 않을 수 있습니다."
)
