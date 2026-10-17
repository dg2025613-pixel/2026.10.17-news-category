
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


# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="NEWS AI | 뉴스 장르 분석",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# KNN 이웃 수는 3으로 고정
K_NEIGHBORS = 3

GENRES = ["정치", "경제", "사회", "문화", "스포츠", "IT/과학"]


# =====================================
# 2. 뉴스 스튜디오 디자인
# =====================================
st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #18365b 0%, transparent 55%),
        linear-gradient(145deg, #07111f 0%, #0b1729 55%, #101c30 100%);
    color: #f3f6fc;
}

/* 기본 콘텐츠 너비 */
.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* 사이드바 숨기기 */
[data-testid="stSidebar"] {
    display: none;
}
[data-testid="collapsedControl"] {
    display: none;
}

/* 기본 텍스트 */
.stApp, .stApp p, .stApp label {
    color: #e4eaf5;
}
h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: -0.5px;
}

/* 뉴스 상단 바 */
.news-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #31415a;
    padding: 5px 0 17px 0;
    margin-bottom: 30px;
}
.brand {
    color: #ffffff;
    font-size: 19px;
    font-weight: 900;
    letter-spacing: 2px;
}
.brand span {
    color: #ff414d;
}
.live-tag {
    color: #ff5b65;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff414d;
    margin-right: 7px;
}

/* 메인 헤드라인 */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(115deg, #152c49, #0c192b 75%);
    border: 1px solid #2b4262;
    border-radius: 18px;
    padding: 32px 30px;
    margin-bottom: 25px;
}
.hero:after {
    content: "NEWS";
    position: absolute;
    right: -10px;
    bottom: -35px;
    font-size: 100px;
    font-weight: 900;
    color: rgba(255,255,255,0.035);
    letter-spacing: 5px;
    pointer-events: none;
}
.hero-label {
    display: inline-block;
    background: #e83242;
    color: white;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}
.hero-title {
    font-size: clamp(29px, 4vw, 45px);
    line-height: 1.25;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 12px;
}
.hero-desc {
    color: #b7c7dc;
    font-size: 15px;
    line-height: 1.8;
}

/* 섹션 구분 */
.section-title {
    font-size: 19px;
    font-weight: 800;
    color: white;
    border-left: 4px solid #ff414d;
    padding-left: 12px;
    margin: 28px 0 15px 0;
}

/* 카드 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 29, 48, 0.88);
    border-color: #2b3e59 !important;
    border-radius: 14px;
}

/* 입력창 */
.stTextInput input, .stTextArea textarea {
    background: #0a1423 !important;
    color: #ffffff !important;
    border: 1px solid #40536e !important;
    border-radius: 9px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #ff414d !important;
    box-shadow: 0 0 0 1px #ff414d !important;
}

/* 버튼 */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #e83242, #ff4b55);
    color: white !important;
    font-weight: 800;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    transition: 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #ff4b55, #ff6570);
    border: none;
    transform: translateY(-1px);
}

/* 라디오 버튼 */
.stRadio [role="radiogroup"] {
    gap: 15px;
}
.stRadio label {
    color: #e4eaf5 !important;
}

/* 지표 */
[data-testid="stMetric"] {
    background: #14263e;
    border: 1px solid #304968;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #aebed3 !important;
}
[data-testid="stMetricValue"] {
    color: #ff5b65 !important;
    font-weight: 900;
}

/* 표 */
[data-testid="stDataFrame"] {
    border: 1px solid #30415b;
    border-radius: 10px;
    overflow: hidden;
}

/* 하단 */
.footer {
    margin-top: 42px;
    border-top: 1px solid #293b54;
    padding-top: 18px;
    text-align: center;
    color: #8193ad;
    font-size: 12px;
    line-height: 1.8;
}

/* 작은 화면 */
@media (max-width: 600px) {
    .hero {
        padding: 23px 18px;
    }
    .hero:after {
        font-size: 65px;
    }
    .news-topbar {
        flex-wrap: wrap;
    }
}
</style>

<div class="news-topbar">
    <div class="brand">NEWS<span>AI</span> / INTELLIGENCE</div>
    <div class="live-tag"><span class="live-dot"></span>AI NEWS ANALYSIS</div>
</div>

<div class="hero">
    <div class="hero-label">NEWSROOM · AI ANALYTICS</div>
    <div class="hero-title">뉴스의 흐름을 읽다.<br>데이터로 장르를 분석하다.</div>
    <div class="hero-desc">
        뉴스 원문을 입력하면 AI가 텍스트의 특징을 분석합니다.<br>
        TF-IDF와 KNN 알고리즘으로 뉴스 장르를 분류합니다.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================
# 3. 학습용 예시 데이터
# 실제 프로젝트에서는 충분한 뉴스 데이터로 교체
# =====================================
SAMPLE_DATA = [
    ("정치", "국회 예산안 심의", "국회에서 여야 의원들이 예산안과 주요 정책의 추진 방향을 논의했다."),
    ("정치", "정부 정책 발표", "정부는 관계 부처 회의를 열고 새로운 정책의 세부 계획을 발표했다."),
    ("정치", "지방 행정 정책 논의", "정당 관계자들은 지역 현안과 지방 행정 정책에 대한 의견을 교환했다."),

    ("경제", "국내 물가 동향", "소비자물가와 생활필수품 가격의 변화가 가계 지출과 소비 심리에 영향을 미치고 있다."),
    ("경제", "기업 실적과 주식시장", "기업들의 분기 실적 발표 이후 주식시장에서는 매출과 영업이익 전망에 관심이 모였다."),
    ("경제", "금리와 가계 대출", "금리 변동과 은행의 대출 조건이 가계의 이자 부담과 금융시장에 영향을 주고 있다."),

    ("사회", "지역 교통 안전 대책", "지자체는 보행자 사고를 줄이기 위해 교통 시설을 점검하고 안전 대책을 마련했다."),
    ("사회", "학교 교육 환경 개선", "교육 당국은 학교 시설과 학생들의 학습 환경을 개선하기 위한 지원 계획을 검토했다."),
    ("사회", "지역 주민 복지", "지역 사회에서는 복지 서비스 확대와 주민 지원을 위한 사업이 진행되고 있다."),

    ("문화", "미술 전시회 개최", "새로운 미술 전시회가 열리면서 관람객들이 작품과 작가의 예술적 표현을 감상했다."),
    ("문화", "지역 문화 축제", "지역 문화 축제에서는 전통 공연과 예술 프로그램이 열려 다양한 문화 콘텐츠를 소개했다."),
    ("문화", "영화제 신작 공개", "영화제에서 감독들의 신작이 공개됐으며 관객과 평론가들이 작품의 연출을 평가했다."),

    ("스포츠", "축구 경기 승리", "축구팀은 경기 후반 득점에 성공하며 상대 팀을 제치고 중요한 승리를 거뒀다."),
    ("스포츠", "프로야구 선수 기록", "프로야구 선수는 경기에서 안타와 타점을 기록하며 팀의 공격을 이끌었다."),
    ("스포츠", "국제 대회 결승 진출", "선수단은 준결승 경기에서 좋은 경기력을 보여 국제 대회 결승전에 진출했다."),

    ("IT/과학", "인공지능 기술 연구", "연구진은 인공지능 모델의 학습 성능을 개선하는 새로운 알고리즘을 발표했다."),
    ("IT/과학", "우주 탐사 연구 성과", "과학자들은 우주 탐사 자료를 분석해 행성과 우주 환경에 관한 새로운 정보를 확인했다."),
    ("IT/과학", "반도체 기술 개발", "연구팀은 반도체의 처리 속도와 에너지 효율을 개선하기 위한 새로운 기술을 연구했다."),
]

SAMPLE_DF = pd.DataFrame(
    SAMPLE_DATA,
    columns=["genre", "title", "text"]
)


# =====================================
# 4. 학습 데이터 불러오기
# =====================================
st.markdown(
    '<div class="section-title">01 / 학습 데이터</div>',
    unsafe_allow_html=True
)

with st.expander("학습용 뉴스 데이터 CSV 업로드", expanded=False):
    st.write(
        "CSV에 genre, title, text 열이 있어야 합니다. "
        "실제 분류 성능을 높이려면 장르별 뉴스 데이터를 충분히 준비하세요."
    )
    uploaded_file = st.file_uploader(
        "뉴스 데이터 파일",
        type=["csv"],
        key="news_csv"
    )

if uploaded_file is not None:
    try:
        train_df = pd.read_csv(uploaded_file)

        if not {"genre", "title", "text"}.issubset(train_df.columns):
            st.error("CSV에는 genre, title, text 열이 필요합니다.")
            st.stop()

        train_df = train_df[["genre", "title", "text"]].dropna().copy()

    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
        st.stop()
else:
    train_df = SAMPLE_DF.copy()
    st.info(
        "현재 테스트용 예시 데이터로 실행 중입니다. "
        "실제 뉴스 분류를 위해서는 충분한 학습 데이터가 필요합니다."
    )

for col in ["genre", "title", "text"]:
    train_df[col] = train_df[col].astype(str).str.strip()

train_df["content"] = (
    train_df["title"] + " " + train_df["text"]
).str.strip()

train_df = train_df[
    train_df["genre"].isin(GENRES)
    & (train_df["content"].str.len() > 0)
].copy()

if len(train_df) < K_NEIGHBORS:
    st.error("K=3으로 분류하려면 유효한 학습 뉴스가 최소 3개 필요합니다.")
    st.stop()

if train_df["genre"].nunique() < 2:
    st.error("학습 데이터에는 최소 2개 이상의 뉴스 장르가 필요합니다.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.metric("학습 뉴스 수", f"{len(train_df):,}개")

with col2:
    st.metric("분류 장르 수", f"{train_df['genre'].nunique()}개")


# =====================================
# 5. TF-IDF + KNN 학습
# KNN 이웃 수는 항상 3
# =====================================
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
            n_neighbors=K_NEIGHBORS,
            weights="distance",
            metric="cosine",
            algorithm="brute"
        )
    )
])

try:
    model.fit(train_df["content"], train_df["genre"])
except Exception as e:
    st.error(f"모델 학습 오류: {e}")
    st.stop()


# =====================================
# 6. 뉴스 URL 검사
# =====================================
def validate_public_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("http 또는 https 주소를 입력하세요.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("올바른 뉴스 URL을 입력하세요.")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError("사이트 주소를 확인할 수 없습니다.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("공개 웹사이트 주소만 사용할 수 있습니다.")

    return url


def extract_article(url):
    current_url = url

    with requests.Session() as session:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; NewsGenreClassifier/1.0)"
            )
        }

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
                    raise ValueError("이동할 주소를 확인할 수 없습니다.")

                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            if "text/html" not in content_type.lower():
                raise ValueError("HTML 뉴스 페이지가 아닙니다.")

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
                    "기사 본문을 가져오지 못했습니다. "
                    "사이트 접근 제한 여부를 확인하거나 본문을 직접 입력하세요."
                )

            return text.strip()

        raise ValueError("페이지 이동 횟수가 너무 많습니다.")


# =====================================
# 7. 뉴스 입력 화면
# =====================================
st.markdown(
    '<div class="section-title">02 / 뉴스 입력</div>',
    unsafe_allow_html=True
)

input_mode = st.radio(
    "뉴스 입력 방법",
    ["뉴스 URL 입력", "본문 직접 입력"],
    horizontal=True
)

if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

if "article_url" not in st.session_state:
    st.session_state["article_url"] = ""

with st.container(border=True):
    if input_mode == "뉴스 URL 입력":
        st.write("🔗 **분석할 뉴스 원문 주소를 입력하세요.**")

        url = st.text_input(
            "뉴스 URL",
            placeholder="https://뉴스사이트/기사주소",
            label_visibility="collapsed"
        )

        if st.button("뉴스 본문 가져오기", type="primary"):
            if not url.strip():
                st.warning("뉴스 URL을 입력하세요.")
            else:
                try:
                    with st.spinner("뉴스 본문을 분석할 준비를 하고 있습니다..."):
                        text = extract_article(url.strip())

                    st.session_state["article_text"] = text
                    st.session_state["article_url"] = url.strip()
                    st.success("뉴스 본문을 가져왔습니다.")

                except Exception as e:
                    st.error(f"본문 추출 실패: {e}")

    else:
        st.write("📝 **뉴스 제목과 본문을 붙여 넣으세요.**")

        manual_text = st.text_area(
            "뉴스 본문",
            height=220,
            placeholder="뉴스 제목과 본문을 입력하세요...",
            label_visibility="collapsed"
        )

        if st.button("입력한 본문 사용하기", type="primary"):
            if len(manual_text.strip()) < 30:
                st.warning("분류할 수 있도록 기사 내용을 더 입력하세요.")
            else:
                st.session_state["article_text"] = manual_text.strip()
                st.session_state["article_url"] = ""
                st.success("입력한 기사를 사용할 준비가 되었습니다.")


# =====================================
# 8. 장르 분류 결과
# =====================================
article_text = st.session_state["article_text"]

if article_text:
    st.markdown(
        '<div class="section-title">03 / 분석 결과</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.write("**분석 대상 기사**")

        if st.session_state["article_url"]:
            st.caption(st.session_state["article_url"])

        st.text_area(
            "기사 내용 확인",
            value=article_text,
            height=160,
            disabled=True,
            label_visibility="collapsed",
            key="preview_text"
        )

        if st.button("AI 뉴스 장르 분석", type="primary"):
            try:
                predicted_genre = model.predict([article_text])[0]

                probabilities = model.predict_proba([article_text])[0]
                classes = model.named_steps["knn"].classes_

                results = pd.DataFrame({
                    "장르": classes,
                    "비율(%)": probabilities * 100
                }).sort_values("비율(%)", ascending=False)

                st.session_state["prediction"] = predicted_genre
                st.session_state["results"] = results

                vectorizer = model.named_steps["tfidf"]
                knn = model.named_steps["knn"]

                query_vector = vectorizer.transform([article_text])
                distances, indices = knn.kneighbors(query_vector)

                similar_rows = train_df.iloc[indices[0]].copy()
                similar_rows["유사도(%)"] = (
                    1 - distances[0]
                ).clip(0, 1) * 100

                st.session_state["similar_rows"] = similar_rows[
                    ["genre", "title", "유사도(%)"]
                ].copy()

            except Exception as e:
                st.error(f"분류 오류: {e}")

    if "prediction" in st.session_state:
        predicted_genre = st.session_state["prediction"]
        results = st.session_state["results"]
        similar_rows = st.session_state["similar_rows"]

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(110deg, #172e4b, #101d30);
                border: 1px solid #e83b49;
                border-left: 6px solid #ff414d;
                border-radius: 14px;
                padding: 25px;
                margin-top: 18px;
            ">
                <div style="
                    color: #ff626b;
                    font-size: 12px;
                    font-weight: 900;
                    letter-spacing: 2px;
                    margin-bottom: 10px;
                ">AI CLASSIFICATION REPORT</div>
                <div style="
                    color: #c4d1e3;
                    font-size: 14px;
                    margin-bottom: 8px;
                ">분석된 뉴스 장르</div>
                <div style="
                    color: white;
                    font-size: 36px;
                    font-weight: 900;
                ">{predicted_genre}</div>
                <div style="
                    color: #aabbd0;
                    font-size: 13px;
                    margin-top: 12px;
                ">TF-IDF 텍스트 분석 · KNN 이웃 3개 기준</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 장르별 모델 점수")

        st.caption(
            "아래 값은 KNN 이웃의 가중치를 바탕으로 계산한 비율이며, "
            "검증된 정답 확률이나 모델 정확도를 의미하지 않습니다."
        )

        st.dataframe(
            results.style.format({"비율(%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

        chart_data = results.set_index("장르")[["비율(%)"]]
        st.bar_chart(chart_data)

        st.markdown("#### 가장 비슷한 학습 뉴스 3개")

        similar_rows = similar_rows.rename(columns={
            "genre": "뉴스 장르",
            "title": "학습 뉴스 제목"
        })

        similar_rows["유사도(%)"] = similar_rows["유사도(%)"].round(1)

        st.dataframe(
            similar_rows,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "유사도는 TF-IDF 벡터의 코사인 유사도를 바탕으로 계산한 참고값입니다."
        )


# =====================================
# 9. 하단 안내
# =====================================
st.markdown("""
<div class="footer">
    NEWS AI · INTELLIGENCE DESK<br>
    TF-IDF TEXT ANALYSIS &nbsp; | &nbsp; K-NEAREST NEIGHBORS<br>
    이 서비스의 결과는 학습 데이터에 따라 달라질 수 있습니다.
</div>
""", unsafe_allow_html=True)
