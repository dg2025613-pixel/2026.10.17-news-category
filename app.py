
import html
import re
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import trafilatura

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. 페이지 기본 설정
# =========================================================

st.set_page_config(
    page_title="AI 뉴스 장르 분류 시스템",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. 밝은 뉴스 스튜디오 디자인
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #f4f8ff 0%,
            #eaf2fc 55%,
            #f9fbff 100%
        );
        color: #172b4d;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #17365d !important;
    }

    p, label, li, .stMarkdown {
        color: #263b57;
    }

    .news-header {
        background: linear-gradient(110deg, #ffffff, #e4efff);
        border: 1px solid #c9d9ef;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 5px 18px rgba(40, 75, 120, 0.08);
    }

    .news-label {
        color: #d92739;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 2px;
    }

    .news-title {
        color: #17365d;
        font-size: 34px;
        font-weight: 900;
        margin-top: 8px;
    }

    .news-subtitle {
        color: #526781;
        font-size: 15px;
        margin-top: 10px;
    }

    .result-card {
        background: linear-gradient(110deg, #ffffff, #e7f0fc);
        border: 1px solid #b9cde6;
        border-left: 6px solid #e83242;
        border-radius: 14px;
        padding: 25px;
        margin-top: 18px;
        box-shadow: 0 4px 14px rgba(40, 75, 120, 0.07);
    }

    .result-label {
        color: #d92739;
        font-weight: 900;
        letter-spacing: 1px;
        font-size: 13px;
    }

    .result-category {
        color: #15365d;
        font-size: 36px;
        font-weight: 900;
        margin: 10px 0;
    }

    .result-caption {
        color: #526781;
        font-size: 13px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #d4e0ef;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        color: #172b4d;
        border: 1px solid #c7d5e7;
        border-radius: 8px;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b9cde6;
        border-radius: 10px;
        padding: 12px;
    }

    div.stButton > button {
        background: #173e70;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #245792;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d4e0ef;
        border-radius: 10px;
        padding: 14px;
    }

    hr {
        border-color: #d4e0ef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 뉴스 장르 및 기본 학습 데이터
# =========================================================

GENRES = ["정치", "경제", "사회", "문화", "스포츠", "IT/과학"]

SAMPLE_DATA = [
    # 정치
    ("정치", "국회 본회의 법안 처리", "국회가 본회의를 열고 주요 법안의 처리 방안을 논의했다. 여야 의원들은 법안의 주요 내용과 시행 시기를 놓고 의견을 나눴다."),
    ("정치", "정부 정책 추진 계획 발표", "정부는 관계 부처 회의를 열고 올해 정책 추진 계획을 발표했다. 정책의 실행 일정과 예산 배분 방안도 함께 검토했다."),
    ("정치", "지방자치단체 정책 협의", "지방자치단체장들이 지역 현안 해결을 위한 협의에 나섰다. 참석자들은 교통과 주거 문제에 대한 대응 방안을 논의했다."),
    ("정치", "정당 정책 토론회 개최", "정당 관계자들이 정책 토론회에서 주요 공약과 입법 방향을 설명했다. 토론에서는 복지와 교육 정책이 주요 쟁점으로 다뤄졌다."),
    ("정치", "국회 상임위원회 회의", "국회 상임위원회가 소관 부처의 업무 현황을 점검했다. 의원들은 정책 집행 과정과 관련 예산에 관해 질의했다."),

    # 경제
    ("경제", "국내 물가 동향 발표", "통계 자료에 따르면 소비자물가의 변동 폭이 주요 경제 지표로 주목받고 있다. 시장에서는 식료품과 에너지 가격의 움직임을 살펴보고 있다."),
    ("경제", "기업 실적과 매출 분석", "기업들이 분기 실적을 발표하면서 매출과 영업이익 변화에 관심이 모이고 있다. 업계에서는 수요와 원가 변화가 실적에 영향을 미쳤다고 분석했다."),
    ("경제", "한국 증시 거래 동향", "국내 증시에서 주요 종목의 주가가 움직이며 투자자들의 관심을 끌었다. 시장에서는 금리와 기업 실적, 해외 증시 흐름을 주요 변수로 보고 있다."),
    ("경제", "중소기업 지원 정책", "중소기업의 경영 부담을 줄이기 위한 지원 방안이 논의됐다. 관계 기관은 자금 조달과 판로 확대를 위한 정책을 검토하고 있다."),
    ("경제", "금리와 가계 대출 변화", "금리 변동이 가계 대출과 소비 심리에 미치는 영향에 관심이 쏠리고 있다. 금융권에서는 대출 규모와 상환 부담의 변화를 살펴보고 있다."),

    # 사회
    ("사회", "지역 교통 안전 대책", "지역 관계 기관이 교통사고 예방을 위한 안전 대책을 마련하고 있다. 보행자 통행이 많은 구역의 시설과 안내 체계를 점검할 예정이다."),
    ("사회", "학교 주변 안전 점검", "관계 기관이 학교 주변 통학로의 안전 시설을 점검했다. 점검 과정에서 보행 환경과 교통안전 시설의 개선 필요성이 논의됐다."),
    ("사회", "시민 자원봉사 활동", "지역 주민들이 환경 정화와 취약계층 지원을 위한 봉사활동에 참여했다. 참가자들은 지속적인 지역사회 참여의 중요성을 강조했다."),
    ("사회", "대중교통 이용 환경 개선", "대중교통 이용자의 편의를 높이기 위한 시설 개선 방안이 검토되고 있다. 관계 기관은 혼잡 시간대와 이용자 의견을 조사하고 있다."),
    ("사회", "지역 복지 서비스 확대", "지역사회 복지 서비스를 확대하기 위한 논의가 진행됐다. 관계자들은 돌봄 지원과 복지 정보 접근성을 높이는 방안을 살펴봤다."),

    # 문화
    ("문화", "지역 미술 전시회 개막", "지역 미술관에서 현대 미술 작품을 소개하는 전시회가 열렸다. 관람객들은 다양한 작품을 감상하며 작가의 표현 방식과 전시 주제를 살펴봤다."),
    ("문화", "도서관 독서 문화 행사", "공공도서관이 독서 문화를 확산하기 위한 행사를 마련했다. 행사에는 작가와 독자가 참여해 책의 주제와 독서 경험을 나눴다."),
    ("문화", "전통문화 체험 프로그램", "문화시설에서 전통 공예와 생활문화를 소개하는 체험 프로그램이 진행됐다. 참가자들은 전통문화의 특징과 제작 과정을 배웠다."),
    ("문화", "영화제 상영작 공개", "영화제가 올해 상영작과 주요 프로그램을 공개했다. 주최 측은 다양한 장르의 작품을 통해 관객과 창작자가 만나는 자리를 마련할 계획이다."),
    ("문화", "지역 음악 공연 개최", "지역 공연장에서 음악 공연이 열려 관객들의 관심을 모았다. 공연은 여러 음악 장르를 아우르며 지역 문화예술 활동을 소개했다."),

    # 스포츠
    ("스포츠", "프로축구 경기 결과", "프로축구 경기에서 양 팀이 치열한 공방을 펼쳤다. 선수들은 공격과 수비에서 집중력을 보였으며 경기 종료 후 감독들이 경기 내용을 평가했다."),
    ("스포츠", "프로야구 선수 기록", "프로야구 경기에서 타자와 투수의 활약이 승부에 영향을 미쳤다. 구단은 선수들의 경기 기록과 다음 경기 준비 상황을 살펴보고 있다."),
    ("스포츠", "국제 수영 대회 개최", "국제 수영 대회에서 여러 나라의 선수들이 종목별 경기에 참가했다. 선수들은 예선과 결승에서 기록을 겨루며 기량을 선보였다."),
    ("스포츠", "농구 리그 경기 분석", "농구 리그 경기에서 빠른 공격 전개와 수비 전략이 주요 관전 요소로 꼽혔다. 양 팀 감독은 경기 후 선수들의 움직임과 전술을 분석했다."),
    ("스포츠", "선수 훈련과 대회 준비", "선수들이 다가오는 대회를 앞두고 훈련 강도와 경기 전략을 조정하고 있다. 코칭스태프는 체력과 기술 훈련을 함께 진행하고 있다."),

    # IT/과학
    ("IT/과학", "인공지능 기술 연구 동향", "연구진이 인공지능 모델의 학습 효율을 높이는 방법을 연구하고 있다. 이번 연구에서는 데이터 처리와 모델 성능 평가 방식이 주요하게 다뤄졌다."),
    ("IT/과학", "반도체 기술 개발", "반도체 업계가 처리 속도와 에너지 효율을 개선하기 위한 기술 개발에 나서고 있다. 연구진은 소자 구조와 제조 공정의 개선 가능성을 검토했다."),
    ("IT/과학", "우주 탐사 연구 진행", "과학자들이 우주 환경을 관측하기 위한 연구를 진행하고 있다. 관측 자료는 천체의 특성과 우주 환경을 이해하는 데 활용될 예정이다."),
    ("IT/과학", "로봇 센서 기술 연구", "연구진이 로봇의 주변 환경 인식 능력을 높이기 위한 센서 기술을 개발하고 있다. 센서에서 수집한 정보는 장애물 인식과 이동 판단에 활용된다."),
    ("IT/과학", "기후 변화 데이터 분석", "연구진이 장기간의 기상 자료를 분석해 기후 변화의 특징을 살펴보고 있다. 데이터 분석 결과는 환경 변화의 원인과 영향을 연구하는 데 활용된다."),
]


# =========================================================
# 4. 뉴스 URL에서 본문 가져오기
# =========================================================

def extract_news_from_url(url):
    """뉴스 URL에서 제목과 본문을 추출한다."""

    url = url.strip()

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("올바른 뉴스 URL을 입력해 주세요. (http:// 또는 https://)")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    downloaded = response.text

    extracted_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else "제목을 가져오지 못했습니다."

    if not extracted_text or len(extracted_text.strip()) < 30:
        raise ValueError(
            "뉴스 본문을 자동으로 가져오지 못했습니다. "
            "해당 언론사가 자동 수집을 차단했을 수 있으니 본문을 직접 입력해 주세요."
        )

    return title, extracted_text.strip()


# =========================================================
# 5. 학습 데이터 불러오기
# =========================================================

def normalize_training_data(df):
    """업로드한 CSV의 열 이름을 표준 형식으로 정리한다."""

    df = df.copy()

    # 열 이름의 앞뒤 공백 제거
    df.columns = [str(col).strip().lower() for col in df.columns]

    # 자주 사용하는 열 이름을 표준 이름으로 변환
    rename_map = {}

    for col in df.columns:
        if col in ["category", "label", "분류", "장르"]:
            rename_map[col] = "genre"
        elif col in ["content", "body", "본문", "내용", "article"]:
            rename_map[col] = "text"
        elif col in ["headline", "제목"]:
            rename_map[col] = "title"

    df = df.rename(columns=rename_map)

    if "genre" not in df.columns:
        raise ValueError(
            "CSV에 genre 또는 category 열이 필요합니다."
        )

    if "text" not in df.columns:
        raise ValueError(
            "CSV에 text 또는 content 열이 필요합니다."
        )

    if "title" not in df.columns:
        df["title"] = ""

    df["genre"] = df["genre"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # 제목도 분류에 활용할 수 있도록 본문과 결합
    df["combined_text"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    df = df[
        (df["genre"] != "")
        & (df["text"] != "")
        & (df["combined_text"] != "")
    ].copy()

    if len(df) < 1:
        raise ValueError("사용할 수 있는 학습 데이터가 없습니다.")

    if df["genre"].nunique() < 2:
        raise ValueError(
            "학습 데이터에는 최소 2개 이상의 서로 다른 장르가 필요합니다."
        )

    return df.reset_index(drop=True)


# =========================================================
# 6. TF-IDF + KNN 모델 학습
# =========================================================

@st.cache_data(show_spinner=False)
def prepare_sample_data():
    df = pd.DataFrame(
        SAMPLE_DATA,
        columns=["genre", "title", "text"],
    )
    return normalize_training_data(df)


def train_model(df):
    """TF-IDF로 텍스트를 수치화하고 KNN 모델을 학습한다."""

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        max_features=50000,
    )

    X = vectorizer.fit_transform(df["combined_text"])
    y = df["genre"]

    # 이웃 수는 3으로 고정하되 데이터가 3개 미만이면 조정
    k = min(3, len(df))

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="cosine",
        algorithm="brute",
    )

    model.fit(X, y)

    return vectorizer, model, X, k


# =========================================================
# 7. 화면 상단
# =========================================================

st.markdown(
    """
    <div class="news-header">
        <div class="news-label">AI NEWS ANALYTICS</div>
        <div class="news-title">📰 AI 뉴스 장르 분류 시스템</div>
        <div class="news-subtitle">
            TF-IDF와 KNN을 활용해 뉴스의 텍스트 특징을 분석하고
            장르를 예측하는 인공지능 프로젝트
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, middle, right = st.columns(3)

with left:
    st.metric("분류 장르", "6개")

with middle:
    st.metric("분석 방법", "TF-IDF")

with right:
    st.metric("KNN 이웃 수", "3개")


# =========================================================
# 8. 학습 데이터 업로드
# =========================================================

with st.expander("📂 학습 데이터 CSV 업로드 (선택 사항)", expanded=False):
    st.write(
        "직접 만든 뉴스 데이터가 있다면 CSV 파일을 업로드할 수 있어. "
        "업로드하지 않으면 기본 예시 데이터로 분석해."
    )

    st.caption(
        "필수 열: genre, text / 선택 열: title. "
        "category, content 같은 열 이름도 지원해."
    )

    uploaded_file = st.file_uploader(
        "뉴스 학습 데이터 CSV",
        type=["csv"],
        key="training_csv",
    )

    training_df = None

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            training_df = normalize_training_data(uploaded_df)

            st.success(
                f"학습 데이터 {len(training_df)}개를 불러왔어."
            )

            st.dataframe(
                training_df[["genre", "title", "text"]].head(10),
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"CSV 파일을 읽는 중 오류가 발생했어: {e}")


# =========================================================
# 9. 입력 방식 선택
# =========================================================

st.subheader("🔎 분석할 뉴스 입력")

input_mode = st.radio(
    "뉴스 입력 방법",
    ["뉴스 URL 입력", "본문 직접 입력"],
    horizontal=True,
)

news_title = ""
news_text = ""
news_url = ""

if input_mode == "뉴스 URL 입력":
    news_url = st.text_input(
        "뉴스 기사 URL",
        placeholder="https://example.com/news/article",
    )

    st.caption(
        "일부 언론사는 자동 본문 추출을 차단할 수 있어. "
        "그런 경우에는 '본문 직접 입력'을 이용해 줘."
    )

else:
    news_title = st.text_input(
        "뉴스 제목 (선택 사항)",
        placeholder="뉴스 제목을 입력해 줘",
    )

    news_text = st.text_area(
        "뉴스 본문",
        placeholder="분석할 뉴스 본문을 여기에 붙여 넣어 줘.",
        height=240,
    )


# =========================================================
# 10. 뉴스 분석 실행
# =========================================================

analyze_button = st.button(
    "🔍 뉴스 장르 분석하기",
    use_container_width=True,
)

if analyze_button:

    # URL에서 본문 추출
    if input_mode == "뉴스 URL 입력":

        if not news_url.strip():
            st.warning("뉴스 URL을 입력해 줘.")
            st.stop()

        try:
            with st.spinner("뉴스 본문을 가져오는 중..."):
                news_title, news_text = extract_news_from_url(news_url)

            st.success("뉴스 본문을 가져왔어.")

        except Exception as e:
            st.error(f"뉴스를 가져오지 못했어: {e}")
            st.stop()

    # 직접 입력한 본문 확인
    if not news_text.strip():
        st.warning("분석할 뉴스 본문을 입력해 줘.")
        st.stop()

    if len(news_text.strip()) < 20:
        st.warning("조금 더 긴 뉴스 본문을 입력해 줘. (20자 이상 권장)")
        st.stop()

    # 학습 데이터 선택
    if uploaded_file is not None:
        if training_df is None:
            st.error("업로드한 CSV를 확인해 줘.")
            st.stop()
        data = training_df.copy()
    else:
        data = prepare_sample_data()

    # 모델 학습 및 분석
    try:
        with st.spinner("TF-IDF와 KNN으로 뉴스를 분석하는 중..."):

            vectorizer, model, X_train, k = train_model(data)

            combined_news = (
                str(news_title).strip() + " " + news_text.strip()
            ).strip()

            X_news = vectorizer.transform([combined_news])

            predicted_genre = model.predict(X_news)[0]

            # 가까운 학습 뉴스 찾기
            similarities = cosine_similarity(X_news, X_train).flatten()

            # 유사도가 높은 뉴스 순서로 정렬
            similar_indices = similarities.argsort()[::-1]

            # 유사도 0인 데이터는 제외
            similar_indices = [
                int(i)
                for i in similar_indices
                if similarities[i] > 0
            ][:5]

            # 모델 내부의 이웃 정보 확인
            distances, neighbor_indices = model.kneighbors(
                X_news,
                n_neighbors=k,
            )

            neighbor_details = []

            for idx, distance in zip(
                neighbor_indices[0],
                distances[0],
            ):
                neighbor_details.append(
                    {
                        "genre": data.iloc[int(idx)]["genre"],
                        "title": data.iloc[int(idx)]["title"],
                        "similarity": max(0.0, 1.0 - float(distance)),
                    }
                )

        # 결과 카드
        safe_genre = html.escape(str(predicted_genre))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    AI CLASSIFICATION REPORT
                </div>
                <div style="color: #526781; margin-top: 12px;">
                    분석된 뉴스 장르
                </div>
                <div class="result-category">
                    {safe_genre}
                </div>
                <div class="result-caption">
                    TF-IDF 텍스트 분석 · KNN 이웃 {k}개 기준
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📄 분석한 뉴스")

        if news_title:
            st.write(f"**제목:** {news_title}")

        if input_mode == "뉴스 URL 입력":
            st.write(f"**URL:** {news_url}")

        with st.expander("분석에 사용된 본문 보기"):
            st.write(news_text)

        # 유사 뉴스 표시
        st.markdown("### 🔗 가장 비슷한 학습 뉴스")

        if similar_indices:
            for rank, idx in enumerate(similar_indices, start=1):
                row = data.iloc[idx]
                similarity_percent = similarities[idx] * 100

                safe_title = html.escape(
                    str(row["title"]) or "제목 없음"
                )
                safe_row_genre = html.escape(str(row["genre"]))

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div style="color:#d92739;font-weight:800;">
                            유사 뉴스 {rank}
                        </div>
                        <div style="
                            color:#17365d;
                            font-size:18px;
                            font-weight:800;
                            margin:7px 0;
                        ">
                            {safe_title}
                        </div>
                        <div style="color:#526781;">
                            장르: {safe_row_genre}
                            &nbsp; | &nbsp;
                            텍스트 유사도: {similarity_percent:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"유사 뉴스 {rank} 본문 보기"):
                    st.write(row["text"])

        else:
            st.info(
                "학습 데이터에서 유사한 텍스트를 찾지 못했어. "
                "학습 데이터를 추가하면 비교 결과가 개선될 수 있어."
            )

        # KNN 이웃 결과
        st.markdown("### 🧠 KNN 이웃 분석")

        st.write(
            "KNN은 입력 뉴스와 텍스트 특징이 가까운 학습 뉴스 "
            "3개를 찾아 장르를 예측해. 아래는 실제로 선택된 이웃이야."
        )

        neighbor_df = pd.DataFrame(neighbor_details)

        if not neighbor_df.empty:
            neighbor_df["similarity"] = (
                neighbor_df["similarity"] * 100
            ).round(1)

            neighbor_df = neighbor_df.rename(
                columns={
                    "genre": "뉴스 장르",
                    "title": "학습 뉴스 제목",
                    "similarity": "유사도(%)",
                }
            )

            st.dataframe(
                neighbor_df,
                use_container_width=True,
                hide_index=True,
            )

        # 학습 데이터 분포
        st.markdown("### 📊 학습 데이터 장르 분포")

        genre_counts = (
            data["genre"]
            .value_counts()
            .rename_axis("장르")
            .reset_index(name="뉴스 개수")
        )

        st.bar_chart(
            genre_counts.set_index("장르")["뉴스 개수"]
        )

        st.caption(
            "주의: 분류 결과는 학습 데이터의 구성과 품질에 영향을 받아. "
            "이 모델은 뉴스의 진위 여부를 판별하는 것이 아니라 "
            "텍스트를 바탕으로 장르를 예측하는 모델이야."
        )

    except Exception as e:
        st.error(f"분석 중 오류가 발생했어: {e}")
        st.exception(e)


# =========================================================
# 11. 하단 설명
# =========================================================

st.divider()

st.markdown(
    """
    **분류 원리**

    1. **TF-IDF:** 뉴스에 등장하는 문자 조각의 중요도를 수치로 변환해.
    2. **KNN:** 새 뉴스와 가까운 학습 뉴스 3개를 찾아.
    3. **장르 예측:** 가까운 뉴스들의 장르를 바탕으로 결과를 예측해.

    이 앱은 학습 데이터 기반의 간단한 텍스트 분류 모델이야.
    결과는 참고용이며, 실제 뉴스 장르와 항상 일치하는 것은 아니야.
    """
)
