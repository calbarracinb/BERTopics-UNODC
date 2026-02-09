from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_DATA_PATH = Path("data/processed/tweets_sentiment.csv")
SENTIMENT_ORDER = ["POS", "NEU", "NEG"]
SENTIMENT_COLORS = {"POS": "#00A878", "NEU": "#3A506B", "NEG": "#D1495B"}
SENTIMENT_LABEL_ES = {"POS": "Positivo", "NEU": "Neutral", "NEG": "Negativo"}
UI_FONT_FAMILY = "Manrope, Segoe UI, sans-serif"
EXCLUDED_HANDLES = {"@simciupp", "@uppachuca", "@cib_uppachuca"}
WEIGHTING_OPTIONS = {
    "Ponderado por tweets": ("raw", "cantidad de tweets"),
    "Ponderado por vistas": ("views", "peso por vistas"),
    "Ponderado por reposts": ("reposts", "peso por reposts"),
    "Ponderado por likes": ("likes", "peso por likes"),
}


def format_number_es(value: float | int, decimals: int = 0) -> str:
    fmt = f"{{:,.{decimals}f}}"
    text = fmt.format(float(value))
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_percent_es(value: float, decimals: int = 1) -> str:
    return f"{format_number_es(value, decimals)}%"


def _apply_spanish_number_separators(fig) -> None:
    fig.update_layout(separators=",.")


def _style_figure(fig) -> None:
    _apply_spanish_number_separators(fig)
    fig.update_layout(
        font={"family": UI_FONT_FAMILY, "size": 16, "color": "#243447"},
        title={"x": 0.5, "xanchor": "center"},
        legend_title_text=None,
        margin={"t": 72, "b": 38, "l": 26, "r": 18},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
    )
    fig.update_xaxes(title_standoff=14, gridcolor="rgba(41, 64, 84, 0.10)")
    fig.update_yaxes(title_standoff=14, gridcolor="rgba(41, 64, 84, 0.10)")
    fig.update_coloraxes(colorbar_tickformat="~s")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")

    numeric_cols = [
        "comments",
        "likes",
        "reposts",
        "views",
        "sentiment_positive",
        "sentiment_neutral",
        "sentiment_negative",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "tweet_date" in df.columns:
        df["tweet_date"] = pd.to_datetime(df["tweet_date"], errors="coerce")

    text_cols = ["tweet_text", "tweet_text_display", "tweet_text_model", "mentions", "hashtags", "author_handle"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Seguridad: elimina cuentas excluidas incluso si el CSV fue generado antes de la regla del pipeline.
    if "author_handle" in df.columns:
        excluded_pattern = "|".join(re.escape(h) for h in sorted(EXCLUDED_HANDLES))
        mentions = df["mentions"].str.lower() if "mentions" in df.columns else pd.Series("", index=df.index)
        text_display = (
            df["tweet_text_display"].str.lower()
            if "tweet_text_display" in df.columns
            else df["tweet_text"].str.lower()
        )
        df = df[
            (~df["author_handle"].str.lower().isin(EXCLUDED_HANDLES))
            & (~mentions.str.contains(excluded_pattern, na=False, regex=True))
            & (~text_display.str.contains(excluded_pattern, na=False, regex=True))
        ].copy()

    return df


def _sentiment_display_series(df: pd.DataFrame, sentiment_column: str) -> pd.Series:
    return df[sentiment_column].map(SENTIMENT_LABEL_ES).fillna("Neutral")


def _weight_series(df: pd.DataFrame, weight_mode: str) -> pd.Series:
    if weight_mode == "views":
        return pd.to_numeric(df.get("views", 0), errors="coerce").fillna(0).clip(lower=0)
    if weight_mode == "reposts":
        return pd.to_numeric(df.get("reposts", 0), errors="coerce").fillna(0).clip(lower=0)
    if weight_mode == "likes":
        return pd.to_numeric(df.get("likes", 0), errors="coerce").fillna(0).clip(lower=0)
    return pd.Series(1.0, index=df.index, dtype="float64")


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str, str]:
    st.sidebar.header("Filtros y opciones")

    sentiment_options: list[tuple[str, str]] = [("sentiment", "Tono general")]
    if "sentiment_simci_adjusted" in df.columns:
        sentiment_options.insert(0, ("sentiment_simci_adjusted", "Tono sobre SIMCI (ajustado)"))

    selected_source_label = st.sidebar.radio(
        "Cómo leer el tono",
        options=[label for _, label in sentiment_options],
        index=0,
    )
    sentiment_column = next(col for col, label in sentiment_options if label == selected_source_label)

    selected_sentiments = st.sidebar.multiselect(
        "Tipos de tono",
        options=SENTIMENT_ORDER,
        default=SENTIMENT_ORDER,
        format_func=lambda x: SENTIMENT_LABEL_ES.get(x, x),
    )

    selected_weighting_label = st.sidebar.radio(
        "Ponderación de resultados",
        options=list(WEIGHTING_OPTIONS.keys()),
        index=0,
    )
    weight_mode, weight_label = WEIGHTING_OPTIONS[selected_weighting_label]

    time_granularity = st.sidebar.radio(
        "Vista temporal",
        options=["Semanal", "Mensual"],
        index=0,
    )

    if "authors_filter" not in st.session_state:
        st.session_state["authors_filter"] = []
    if "author_chart_selection" not in st.session_state:
        st.session_state["author_chart_selection"] = []
    if "last_applied_chart_selection" not in st.session_state:
        st.session_state["last_applied_chart_selection"] = []

    available_authors = set(df["author_handle"].unique()) if "author_handle" in df.columns else set()
    chart_selection = [author for author in st.session_state["author_chart_selection"] if author in available_authors]
    if chart_selection != st.session_state["last_applied_chart_selection"]:
        st.session_state["authors_filter"] = chart_selection
        st.session_state["last_applied_chart_selection"] = chart_selection

    author_options = sorted([author for author in df["author_handle"].dropna().astype(str).unique() if author])
    selected_authors = st.sidebar.multiselect(
        "Cuentas",
        options=author_options,
        key="authors_filter",
    )

    filtered = df.copy()
    if sentiment_column in filtered.columns:
        filtered = filtered[filtered[sentiment_column].isin(selected_sentiments)]
    if selected_authors:
        filtered = filtered[filtered["author_handle"].isin(selected_authors)]

    if "tweet_date" in df.columns and df["tweet_date"].notna().any():
        min_date = df["tweet_date"].min().date()
        max_date = df["tweet_date"].max().date()
        start_date, end_date = st.sidebar.slider(
            "Rango de fechas",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["tweet_date"] >= start_ts) & (filtered["tweet_date"] <= end_ts)]

    return filtered, sentiment_column, weight_mode, weight_label, time_granularity


def render_kpis(df: pd.DataFrame, sentiment_column: str, weight_mode: str) -> None:
    total_tweets = len(df)
    total_comments = float(df["comments"].sum()) if "comments" in df.columns else 0.0
    total_likes = float(df["likes"].sum()) if "likes" in df.columns else 0.0
    total_reposts = float(df["reposts"].sum()) if "reposts" in df.columns else 0.0
    total_views = float(df["views"].sum()) if "views" in df.columns else 0.0

    avg_likes = float(df["likes"].mean()) if total_tweets and "likes" in df.columns else 0.0
    avg_reposts = float(df["reposts"].mean()) if total_tweets and "reposts" in df.columns else 0.0
    avg_views = float(df["views"].mean()) if total_tweets and "views" in df.columns else 0.0

    # Keep rows balanced: featured "Tweets" card + 6 standard cards.
    metric_cards = [
        ("Interacciones totales", format_number_es(total_likes + total_reposts + total_comments)),
        ("Me gusta totales", format_number_es(total_likes)),
        ("Reposts totales", format_number_es(total_reposts)),
        ("Vistas totales", format_number_es(total_views)),
        ("Me gusta promedio", format_number_es(avg_likes, 1)),
        ("Reposts promedio", format_number_es(avg_reposts, 1)),
    ]

    # If there are no comments at all, prioritize "Vistas promedio" as a more useful KPI.
    if total_comments <= 0:
        metric_cards[0] = ("Vistas promedio", format_number_es(avg_views, 1))

    cards_html = [
        (
            '<div class="kpi-card kpi-card-featured">'
            '<div class="kpi-label">Tweets seleccionados</div>'
            f'<div class="kpi-value">{format_number_es(total_tweets)}</div>'
            "</div>"
        )
    ]
    for label, value in metric_cards:
        cards_html.append(
            (
                '<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                "</div>"
            )
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def _build_timeline_period(df: pd.DataFrame, time_granularity: str) -> pd.Series:
    if time_granularity == "Mensual":
        return df["tweet_date"].dt.to_period("M").dt.to_timestamp().dt.date
    return df["tweet_date"].dt.to_period("W-MON").dt.start_time.dt.date


def render_charts(df: pd.DataFrame, sentiment_column: str, weight_mode: str, weight_label: str, time_granularity: str) -> None:
    if df.empty:
        st.warning("No hay datos después de aplicar filtros.")
        return

    df_plot = df.copy()
    df_plot["sentimiento_es"] = _sentiment_display_series(df_plot, sentiment_column)
    df_plot["peso"] = _weight_series(df_plot, weight_mode)

    has_dates = "tweet_date" in df_plot.columns and df_plot["tweet_date"].notna().any()
    timeline = pd.DataFrame()
    if has_dates:
        timeline = (
            df_plot.dropna(subset=["tweet_date"])
            .assign(periodo=lambda x: _build_timeline_period(x, time_granularity))
            .groupby(["periodo", "sentimiento_es"], as_index=False)["peso"]
            .sum()
            .rename(columns={"peso": "valor"})
        )
        timeline["periodo"] = pd.to_datetime(timeline["periodo"], errors="coerce")
        timeline = timeline.dropna(subset=["periodo"])

    chart_col_1, chart_col_2 = st.columns(2, gap="large")

    with chart_col_1:
        sentiment_counts = (
            df_plot.groupby(sentiment_column, as_index=False)["peso"]
            .sum()
            .rename(columns={"peso": "valor"})
        )
        sentiment_counts = sentiment_counts.set_index(sentiment_column).reindex(SENTIMENT_ORDER, fill_value=0).reset_index()
        sentiment_counts["sentimiento_es"] = sentiment_counts[sentiment_column].map(SENTIMENT_LABEL_ES)
        total_val = float(sentiment_counts["valor"].sum())
        sentiment_counts["etiqueta_pct"] = sentiment_counts["valor"].apply(
            lambda v: format_percent_es((float(v) / total_val * 100), 1) if total_val > 0 else "0,0%"
        )

        fig_sent = px.pie(
            sentiment_counts,
            names="sentimiento_es",
            values="valor",
            hole=0.55,
            color=sentiment_column,
            color_discrete_map=SENTIMENT_COLORS,
            title="Distribución del tono de los tweets",
        )
        fig_sent.update_traces(
            text=sentiment_counts["etiqueta_pct"],
            textposition="inside",
            texttemplate="%{label}<br>%{text}",
        )
        _style_figure(fig_sent)
        st.plotly_chart(fig_sent, use_container_width=True)
        if weight_mode != "raw":
            st.caption(f"Visualización ponderada por {weight_label}.")

    with chart_col_2:
        if not timeline.empty:
            fig_tl = px.area(
                timeline,
                x="periodo",
                y="valor",
                color="sentimiento_es",
                color_discrete_map={SENTIMENT_LABEL_ES[k]: v for k, v in SENTIMENT_COLORS.items()},
                title=f"Evolución del tono en el tiempo ({time_granularity.lower()}, apilado)",
                labels={"periodo": "Periodo", "valor": weight_label.capitalize(), "sentimiento_es": "Sentimiento"},
            )
            fig_tl.update_traces(stackgroup="one")
            fig_tl.update_yaxes(tickformat="~s")
            _style_figure(fig_tl)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("No hay fechas válidas para graficar la evolución temporal.")

    if not timeline.empty:
        sentiments = [SENTIMENT_LABEL_ES[key] for key in SENTIMENT_ORDER]
        all_periods = sorted(timeline["periodo"].unique())
        reindexed = (
            timeline.set_index(["periodo", "sentimiento_es"])
            .reindex(
                pd.MultiIndex.from_product([all_periods, sentiments], names=["periodo", "sentimiento_es"]),
                fill_value=0.0,
            )
            .reset_index()
        )
        reindexed["acumulado"] = reindexed.groupby("sentimiento_es")["valor"].cumsum()

        y_label = "Acumulado de tweets" if weight_mode == "raw" else f"{weight_label.capitalize()} acumulado"
        title = "Acumulado del tono en el tiempo (apilado)"
        if weight_mode != "raw":
            title = f"Acumulado del tono en el tiempo (apilado, {weight_label})"

        fig_cumulative = px.area(
            reindexed,
            x="periodo",
            y="acumulado",
            color="sentimiento_es",
            color_discrete_map={SENTIMENT_LABEL_ES[k]: v for k, v in SENTIMENT_COLORS.items()},
            title=title,
            labels={"periodo": "Periodo", "acumulado": y_label, "sentimiento_es": "Sentimiento"},
        )
        fig_cumulative.update_traces(stackgroup="one")
        fig_cumulative.update_yaxes(tickformat="~s")
        _style_figure(fig_cumulative)
        st.plotly_chart(fig_cumulative, use_container_width=True)

    chart_col_3, chart_col_4 = st.columns(2, gap="large")

    with chart_col_3:
        top_accounts = (
            df_plot.groupby("author_handle", as_index=False)["peso"]
            .sum()
            .rename(columns={"peso": "valor"})
            .sort_values("valor", ascending=False)
            .head(12)
        )
        fig_accounts = px.bar(
            top_accounts,
            x="valor",
            y="author_handle",
            orientation="h",
            title=f"Cuentas con mayor interacción ({weight_label})",
            color="valor",
            color_continuous_scale="Tealgrn",
            labels={"valor": weight_label.capitalize(), "author_handle": "Autor"},
        )
        fig_accounts.update_xaxes(tickformat="~s")
        _style_figure(fig_accounts)
        fig_accounts.update_layout(yaxis={"categoryorder": "total ascending"})

        event = None
        try:
            event = st.plotly_chart(
                fig_accounts,
                use_container_width=True,
                key="top_accounts_chart",
                on_select="rerun",
                selection_mode=("points",),
            )
        except TypeError:
            st.plotly_chart(fig_accounts, use_container_width=True, key="top_accounts_chart_fallback")

        if isinstance(event, dict):
            points = event.get("selection", {}).get("points", [])
            selected_authors = sorted({point.get("y") for point in points if point.get("y")})
            st.session_state["author_chart_selection"] = selected_authors
            if selected_authors:
                st.caption(f"Selección de autores desde gráfico: {', '.join(selected_authors)}")

    with chart_col_4:
        fig_scatter = px.scatter(
            df_plot,
            x="views",
            y="likes",
            color="sentimiento_es",
            size="reposts",
            color_discrete_map={SENTIMENT_LABEL_ES[k]: v for k, v in SENTIMENT_COLORS.items()},
            hover_data=["author_handle", "tweet_date_raw"],
            title="Relación entre vistas y likes (tamaño = reposts)",
            labels={
                "views": "Vistas",
                "likes": "Likes",
                "reposts": "Reposts",
                "sentimiento_es": "Sentimiento",
                "author_handle": "Autor",
                "tweet_date_raw": "Fecha original",
            },
        )
        fig_scatter.update_xaxes(tickformat="~s")
        fig_scatter.update_yaxes(tickformat="~s")
        _style_figure(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    if "emotion_label_es" in df_plot.columns:
        emotion_counts = (
            df_plot.groupby("emotion_label_es", as_index=False)["peso"]
            .sum()
            .rename(columns={"emotion_label_es": "emocion", "peso": "valor"})
            .sort_values("valor", ascending=False)
            .head(10)
        )
        fig_emotion = px.bar(
            emotion_counts,
            x="emocion",
            y="valor",
            color="valor",
            color_continuous_scale="Mint",
            title=f"Emociones detectadas ({weight_label})",
            labels={"emocion": "Emoción", "valor": weight_label.capitalize()},
        )
        fig_emotion.update_yaxes(tickformat="~s")
        _style_figure(fig_emotion)
        st.plotly_chart(fig_emotion, use_container_width=True)


def render_table(df: pd.DataFrame, sentiment_column: str) -> None:
    if df.empty:
        st.info("No hay tweets para mostrar en la tabla.")
        return

    sentiment_view = df[sentiment_column].map(SENTIMENT_LABEL_ES).fillna("Neutral")
    text_col = "tweet_text_display" if "tweet_text_display" in df.columns else "tweet_text"
    emotion_col = "emotion_label_es" if "emotion_label_es" in df.columns else ("emotion" if "emotion" in df.columns else None)

    base_cols = ["tweet_date_raw", "author_handle", "sentimiento"]
    if emotion_col:
        base_cols.append(emotion_col)
    metric_cols = [col for col in ["comments", "likes", "reposts", "views"] if col in df.columns]
    selected_cols = base_cols + metric_cols + [text_col, "tweet_url"]

    table_df = (
        df.assign(sentimiento=sentiment_view)
        .sort_values(["likes", "views"], ascending=False)
        .loc[:, selected_cols]
        .head(50)
        .rename(
            columns={
                "tweet_date_raw": "Fecha",
                "author_handle": "Autor",
                "sentimiento": "Sentimiento",
                "emotion_label_es": "Emoción",
                "emotion": "Emoción",
                "comments": "Comentarios",
                "likes": "Likes",
                "reposts": "Reposts",
                "views": "Vistas",
                text_col: "Texto",
                "tweet_url": "URL",
            }
        )
    )

    for col in ["Comentarios", "Likes", "Reposts", "Vistas"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").fillna(0).apply(format_number_es)

    st.subheader("Top 50 tweets con mayor interacción")
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("URL"),
            "Texto": st.column_config.TextColumn("Texto", width="large"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="SIMCI X Dashboard", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
          :root {
            --ui-font: 'Manrope', Segoe UI, sans-serif;
            --card-radius: 16px;
          }
          html, body, [data-testid="stAppViewContainer"], .stApp, .stMarkdown, .stDataFrame, .stSidebar {
            font-family: var(--ui-font);
          }
          .stApp {
            background: radial-gradient(circle at top right, #eaf5f7 0%, #ddeaf2 44%, #f7f6f2 100%);
          }
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1500px;
          }
          .dashboard-title {
            text-align: center;
            margin: 0.05rem 0 0.2rem 0;
            color: #0b3043;
            letter-spacing: 0.1px;
          }
          .dashboard-subtitle {
            text-align: center;
            margin: 0 auto 1rem auto;
            max-width: 980px;
            color: #33596f;
            font-size: 1.0rem;
            line-height: 1.4;
          }
          .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.2rem;
            margin-bottom: 0.25rem;
          }
          .kpi-card {
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: var(--card-radius);
            background: linear-gradient(140deg, #ffffff, #f6fbfb);
            padding: 0.9rem 1.05rem;
            min-height: 98px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
          }
          .kpi-card-featured {
            grid-row: span 2;
            background: linear-gradient(145deg, #ffffff, #eef7f9);
            min-height: 208px;
          }
          .kpi-label {
            font-size: 0.88rem;
            color: #2f4858;
            margin-bottom: 0.18rem;
          }
          .kpi-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #003049;
            line-height: 1.15;
          }
          [data-testid="stPlotlyChart"] {
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: var(--card-radius);
            background: rgba(255, 255, 255, 0.78);
            padding: 0.4rem 0.5rem 0.25rem 0.5rem;
          }
          [data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: var(--card-radius);
            overflow: hidden;
          }
          h3 {
            text-align: center;
          }
          @media (max-width: 1200px) {
            .kpi-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .kpi-card-featured {
              grid-row: span 1;
              min-height: 110px;
            }
          }
          @media (max-width: 720px) {
            .kpi-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="dashboard-title">Conversación sobre SIMCI en X</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="dashboard-subtitle">
        Analizamos las menciones a SIMCI en X para identificar tono, tendencias y narrativas clave usando inteligencia artificial (procesamiento del lenguaje natural).
        </p>
        """,
        unsafe_allow_html=True,
    )

    data_path = DEFAULT_DATA_PATH
    if not data_path.exists():
        st.error(
            f"No existe el archivo `{data_path}`. Ejecuta primero: "
            "`python3 run_pipeline.py --input x-2026-02-08.csv x-2026-02-09.csv`"
        )
        st.stop()

    df = load_data(data_path)
    filtered, sentiment_column, weight_mode, weight_label, time_granularity = apply_filters(df)

    render_kpis(filtered, sentiment_column, weight_mode)
    st.divider()
    render_charts(filtered, sentiment_column, weight_mode, weight_label, time_granularity)
    st.divider()
    render_table(filtered, sentiment_column)


if __name__ == "__main__":
    main()
