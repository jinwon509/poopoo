import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 한글 폰트 (Streamlit CSS)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 제목 · wide mode
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ─────────────────────────────────────────────
# 파일명에서 학교명 추출 (하드코딩 금지)
# ─────────────────────────────────────────────
def extract_school_name(filename: str | None) -> str:
    if filename is None:
        return "(업로드 없음)"
    base = os.path.basename(filename)
    # 첫 밑줄 또는 _ 전에 있는 부분을 학교명으로 사용
    for sep in ("_", "‑", "-"):
        if sep in base:
            name = base.split(sep)[0].strip()
            if name:
                return name
    return base.split(".")[0].strip() or "(파일명)"


# ─────────────────────────────────────────────
# CSV 읽기 (인코딩 2단계 시도)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv_bytes(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            return df
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 (제공된 예시 코드 그대로)
# ─────────────────────────────────────────────
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for col in df.columns:
        if col in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            num = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[col] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ─────────────────────────────────────────────
# 사이드바: CSV 업로드
# ─────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = std_cols(read_csv_bytes(env_file))
growth_df = std_cols(read_csv_bytes(growth_file))

# 숫자 변환 후 개체번호를 문자열 그대로 유지
if "개체번호" in growth_df.columns:
    growth_df["개체번호"] = growth_df["개체번호"].astype(str).str.strip()

# 생육 측정 항목 선택 (sidebar)
st.sidebar.header("📏 생육 측정 항목")
growth_metrics = []
if "잎 수" in growth_df.columns:
    growth_metrics.append("잎 수")
if "지상부 길이" in growth_df.columns:
    growth_metrics.append("지상부 길이")
if "지하부 길이" in growth_df.columns:
    growth_metrics.append("지하부 길이")
if "생중량" in growth_df.columns:
    growth_metrics.append("생중량")
selected_metrics = st.sidebar.multiselect(
    "비교할 항목 (일부 선택 가능)",
    options=growth_metrics,
    default=growth_metrics,
)

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ============================================================
# Tab 1 — 실험 개요
# ============================================================
with tab1:
    try:
        if env_df.empty and growth_df.empty:
            st.info("📎 환경 데이터 CSV와 생육 결과 데이터 CSV를 사이드바에서 업로드해 주세요.")
        else:
            # 연구 배경 및 목적
            school_name = extract_school_name(env_file.name if env_file else None)
            school_name = (
                school_name
                or extract_school_name(growth_file.name if growth_file else None)
                or "(업로드 없음)"
            )
            st.markdown(f"### 연구 배경 및 목적")
            st.markdown(
                """
- **극지식물**은 혹독한 환경에서 살아남기 위해 개체마다 다른 성장 전략을 보입니다.
- 같은 조건에서도 **잎 수, 줄기 길이, 생중량**이 크게 다를 수 있습니다.
- 이 대시보드는 우리 학교 `{school}`에서 측정한 개체별 데이터를 비교·분석해 **누가 더 잘 자랐는지, 어떤 특징을 가졌는지**를 한눈에 확인합니다.
 """.format(
                    school=school_name
                )
            )

            # 데이터 요약 표
            st.markdown("### 📋 데이터 요약")
            summary_rows = []
            # 환경 데이터 요약
            if not env_df.empty:
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                date_min = env_df["time"].min() if "time" in env_df.columns else None
                date_max = env_df["time"].max() if "time" in env_df.columns else None
                summary_rows.append(
                    {
                        "구분": f"{school_name} 환경 데이터",
                        "개체(측정 지점) 수": f"{len(env_df):,}행",
                        "측정 기간": f"{date_min.strftime('%Y-%m-%d') if date_min is not None else '—'} ~ {date_max.strftime('%Y-%m-%d') if date_max is not None else '—'}",
                        "측정 항목": ", ".join(env_cols) if env_cols else "—",
                    }
                )
            # 생육 데이터 요약
            if not growth_df.empty:
                g_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                summary_rows.append(
                    {
                        "구분": f"{school_name} 생육 결과",
                        "개체 수": f"{len(growth_df):,}개체",
                        "측정 기간": "단일 시점 (측정일 기준)",
                        "측정 항목": ", ".join(g_cols) if g_cols else "—",
                    }
                )
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", use_container_width=False)

            # 주요 지표 카드 4개
            st.markdown("### 📊 주요 지표")
            c1, c2, c3, c4 = st.columns(4)

            # 총 개체수
            if not growth_df.empty:
                n = len(growth_df)
                c1.metric("총 개체 수", f"{n}개체")
            else:
                c1.metric("총 개체 수", "—")

            # 평균 온도
            if "temperature" in env_df.columns:
                t_mean = env_df["temperature"].dropna().mean()
                if not pd.isna(t_mean):
                    c2.metric("평균 온도", f"{t_mean:.1f} ℃")
                else:
                    c2.metric("평균 온도", "—")
            else:
                c2.metric("평균 온도", "—")

            # 평균 습도
            if "humidity" in env_df.columns:
                h_mean = env_df["humidity"].dropna().mean()
                if not pd.isna(h_mean):
                    c3.metric("평균 습도", f"{h_mean:.1f} %")
                else:
                    c3.metric("평균 습도", "—")
            else:
                c3.metric("평균 습도", "—")

            # 평균 생중량
            if "생중량" in growth_df.columns:
                g_mean = growth_df["생중량"].dropna().mean()
                if not pd.isna(g_mean):
                    c4.metric("평균 생중량", f"{g_mean:.2f} g")
                else:
                    c4.metric("평균 생중량", "—")
            else:
                c4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================================================
# Tab 2 — 환경 데이터
# ============================================================
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 사이드바에서 업로드해 주세요. 업로드 전에는 이 탭의 내용을 보여줄 수 없습니다.")
        else:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]

            # 환경 요약 표 (평균·최소·최대)
            st.markdown("### 🌡️ 환경 요약 (평균 · 최소 · 최대)")
            if env_cols:
                summary_data = {}
                for col in env_cols:
                    s = env_df[col].dropna()
                    if len(s) > 0:
                        summary_data[col] = {
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                        }
                if summary_data:
                    st.dataframe(
                        pd.DataFrame(summary_data).T.rename_axis("항목").reset_index(),
                        width="stretch",
                    )
                else:
                    st.info("환경 수치 데이터가 충분하지 않아 요약 표를 표시할 수 없습니다.")
            else:
                st.info("환경 데이터 파일에 temperature, humidity, ph, ec 열이 없습니다. 다른 형식의 파일일 수 있습니다.")

            # 날짜별 시계열 (일별 평균 꺾은선)
            st.markdown("### 📈 날짜별 시계열 (일별 평균)")
            if "time" in env_df.columns:
                env_df_sorted = env_df.sort_values("time")
                daily = env_df_sorted.set_index("time").resample("D").mean(numeric_only=True)

                # 온도 변화
                if "temperature" in daily.columns:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(
                        go.Scatter(
                            x=daily.index,
                            y=daily["temperature"],
                            mode="lines",
                            name="온도 (℃)",
                            line=dict(color="#e74c3c", width=2),
                        )
                    )
                    fig_temp.update_layout(
                        title="일별 평균 온도 변화",
                        xaxis_title="날짜",
                        yaxis_title="온도 (℃)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_temp, width="stretch")

                # 습도 변화
                if "humidity" in daily.columns:
                    fig_hum = go.Figure()
                    fig_hum.add_trace(
                        go.Scatter(
                            x=daily.index,
                            y=daily["humidity"],
                            mode="lines",
                            name="습도 (%)",
                            line=dict(color="#3498db", width=2),
                        )
                    )
                    fig_hum.update_layout(
                        title="일별 평균 습도 변화",
                        xaxis_title="날짜",
                        yaxis_title="습도 (%)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_hum, width="stretch")

                # pH·EC 변화 (열이 있을 때만)
                pH_EC_present = [c for c in ["ph", "ec"] if c in daily.columns]
                if pH_EC_present:
                    fig_chem = make_subplots(
                        rows=1, cols=len(pH_EC_present), subplot_titles=["pH 변화", "EC 변화"]
                    )
                    colors = {"ph": "#27ae60", "ec": "#8e44ad"}
                    for i, col in enumerate(pH_EC_present, start=1):
                        fig_chem.add_trace(
                            go.Scatter(
                                x=daily.index,
                                y=daily[col].dropna(),
                                mode="lines",
                                name=col.upper(),
                                line=dict(color=colors.get(col, "#333"), width=2),
                            ),
                            row=1,
                            col=i,
                        )
                    fig_chem.update_layout(
                        title="일별 평균 pH · EC 변화",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_chem, width="stretch")
                else:
                    st.info("환경 데이터에 ph 또는 ec 열이 없어 pH·EC 그래프는 표시하지 않았습니다.")

            else:
                st.info("환경 데이터에 시간(time) 열이 없어 시계열 그래프를 표시할 수 없습니다.")

            # expander: 원본 테이블 + CSV 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                # CSV 다운로드
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 환경 데이터 CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{extract_school_name(env_file.name) if env_file else '환경데이터'}_환경CSV.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================================================
# Tab 3 — 생육 결과
# ============================================================
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 데이터 CSV를 사이드바에서 업로드해 주세요. 업로드 전에는 이 탭의 내용을 보여줄 수 없습니다.")
        else:
            # 생중량 열이 없으면 지상부+지하부 질량으로 생성 시도 (이미 std_cols에서 처리됨)
            if "생중량" not in growth_df.columns:
                st.info("생중량(g) 열이 없습니다. 지상부 질량 + 지하부 질량으로 생중량을 계산했습니다. (원본 파일에 지상부/지하부 질량 열이 있는 경우에만 가능)")
                st.stop()

            # 🥇 핵심 결과 카드: 평균 생중량, 최대·최소 개체
            st.markdown("### 🥇 핵심 결과 — 생중량")
            card1, card2, card3 = st.columns(3)
            g_clean = growth_df["생중량"].dropna()
            if len(g_clean) > 0:
                avg_w = g_clean.mean()
                max_w = g_clean.max()
                min_w = g_clean.min()
                max_idx = growth_df["생중량"].idxmax()
                min_idx = growth_df["생중량"].idxmin()
                max_id = growth_df.loc[max_idx, "개체번호"] if max_idx is not None and not pd.isna(max_idx) else "—"
                min_id = growth_df.loc[min_idx, "개체번호"] if min_idx is not None and not pd.isna(min_idx) else "—"
                card1.metric("평균 생중량", f"{avg_w:.2f} g")
                card2.metric("최대 생중량", f"{max_w:.2f} g", delta=f"개체 {max_id}")
                card3.metric("최소 생중량", f"{min_w:.2f} g", delta=f"개체 {min_id}")
            else:
                card1.metric("평균 생중량", "—")
                card2.metric("최대 생중량", "—")
                card3.metric("최소 생중량", "—")

            # ──────────────────────────────────────────
            # 개체별 생육 비교 (막대 그래프)
            # ──────────────────────────────────────────
            st.markdown("### 📊 개체별 생육 비교")
            if selected_metrics and len(selected_metrics) > 0:
                # 평균 생중량 막대 (⭐ 항상 포함하려고 시도)
                bar_metrics = selected_metrics.copy()
                if "생중량" not in bar_metrics and "생중량" in growth_df.columns:
                    bar_metrics = ["생중량"] + [m for m in bar_metrics if m != "생중량"]

                for metric in bar_metrics:
                    if metric not in growth_df.columns:
                        st.info(f"⚠️ '{metric}' 열이 없어 해당 그래프를 건너뛰었습니다.")
                        continue
                    s = growth_df[[ "개체번호", metric ]].dropna()
                    if len(s) < 2:
                        st.info(f"'{metric}' 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                        continue

                    fig_bar = go.Figure()
                    # 생중량이면 상위·하위 25% 강조
                    if metric == "생중량":
                        q75 = s[metric].quantile(0.75)
                        q25 = s[metric].quantile(0.25)
                        for _, row in s.iterrows():
                            color = "#e74c3c" if row[metric] >= q75 else (
                                "#f39c12" if row[metric] <= q25 else "#3498db"
                            )
                            fig_bar.add_trace(
                                go.Bar(
                                    x=[row["개체번호"]],
                                    y=[row[metric]],
                                    name=str(row["개체번호"]),
                                    marker_color=color,
                                    showlegend=False,
                                )
                            )
                        fig_bar.update_layout(
                            title=f"개체별 {metric} (빨강: 상위 25% / 주황: 하위 25% / 파랑: 중간)",
                            xaxis_title="개체번호",
                            yaxis_title=f"{metric}",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                    else:
                        fig_bar.add_trace(
                            go.Bar(
                                x=s["개체번호"],
                                y=s[metric],
                                marker_color="#3498db",
                                name=metric,
                            )
                        )
                        fig_bar.update_layout(
                            title=f"개체별 {metric}",
                            xaxis_title="개체번호",
                            yaxis_title=f"{metric}",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                    st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("비교할 측정 항목을 사이드바에서 선택해 주세요.")

            # ──────────────────────────────────────────
            # 생중량 분포 (박스플롯)
            # ──────────────────────────────────────────
            st.markdown("### 📦 생중량 분포 (박스플롯)")
            if "생중량" in growth_df.columns:
                s_box = growth_df[["개체번호", "생중량"]].dropna()
                if len(s_box) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=s_box["생중량"],
                            name="생중량 (g)",
                            fillcolor="#3498db",
                            marker=dict(color="#3498db"),
                        )
                    )
                    fig_box.update_layout(
                        title="개체별 생중량 분포",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 표시할 수 없습니다.")
            else:
                st.info("생중량(g) 열이 없어 박스플롯을 표시할 수 없습니다.")

            # ──────────────────────────────────────────
            # 상관관계 분석 (산점도 + 추세선) — 2개
            # ──────────────────────────────────────────
            st.markdown("### 🔗 상관관계 분석 (산점도 + 추세선)")

            def scatter_with_trendline(x_col: str, y_col: str, x_label: str, y_label: str):
                """산점도 + np.polyfit 추세선을 그리는 재사용 함수."""
                d = growth_df[[x_col, y_col]].dropna()
                if len(d) < 2:
                    st.info(f"'{x_col}'과 '{y_col}' 데이터가 충분하지 않아 산점도를 표시할 수 없습니다.")
                    return
                x = d[x_col].values
                y = d[y_col].values
                r = np.corrcoef(x, y)[0, 1]
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="개체",
                        marker=dict(color="#e74c3c", size=8, opacity=0.7),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"추세선 (r = {r:.2f})",
                        line=dict(color="#2c3e50", width=2, dash="dash"),
                    )
                )
                fig.update_layout(
                    title=f"{x_label} vs {y_label}  (상관계수 r = {r:.2f})",
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    hovermode="closest",
                )
                st.plotly_chart(fig, width="stretch")

            # 잎 수 vs 생중량
            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                scatter_with_trendline("잎 수", "생중량", "잎 수 (장)", "생중량 (g)")
            else:
                st.info("'잎 수' 또는 '생중량' 열이 없어 '잎 수 vs 생중량' 산점도를 표시할 수 없습니다.")

            # 지상부 길이 vs 생중량
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                scatter_with_trendline("지상부 길이", "생중량", "지상부 길이 (mm)", "생중량 (g)")
            else:
                st.info("'지상부 길이' 또는 '생중량' 열이 없어 '지상부 길이 vs 생중량' 산점도를 표시할 수 없습니다.")

            # ──────────────────────────────────────────
            # expander: 원본 + CSV 다운로드
            # ──────────────────────────────────────────
            with st.expander("📄 생육 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 생육 결과 CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{extract_school_name(growth_file.name) if growth_file else '생육결과'}_생육CSV.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
