import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(page_title="A/B Chart Experiment", layout="wide")
st.title("A/B Chart Experiment – Movies")

st.sidebar.header("Dataset & Variables")

uploaded_file = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])

@st.cache_data
def load_default():
    url = (
        "https://raw.githubusercontent.com/LearnDataSci/articles/master/"
        "Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/"
        "IMDB-Movie-Data.csv"
    )
    return pd.read_csv(url)

@st.cache_data
def load_uploaded(raw_bytes: bytes):
    import io
    return pd.read_csv(io.BytesIO(raw_bytes))

if uploaded_file is not None:
    df = load_uploaded(uploaded_file.read())
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    df = load_default()
    st.sidebar.info("Using default: IMDB Movies (2006-2016)")

# Variable selectors
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols     = df.select_dtypes(include="number").columns.tolist()

default_cat = "Genre"   if "Genre"             in categorical_cols else categorical_cols[0]
default_num = "Revenue (Millions)" if "Revenue (Millions)" in numeric_cols else numeric_cols[0]

cat_col = st.sidebar.selectbox("Categorical variable (X)", categorical_cols,
                                index=categorical_cols.index(default_cat))
num_col = st.sidebar.selectbox("Numeric variable  (Y)", numeric_cols,
                                index=numeric_cols.index(default_num))


QUESTION = f"Which **{cat_col}** generates the highest **{num_col}**?"
st.info(f"### Business Question\n{QUESTION}")


@st.cache_data
def prepare(df: pd.DataFrame, cat: str, num: str) -> pd.DataFrame:
    data = df[[cat, num]].dropna()
    if data[cat].astype(str).str.contains(",").any():
        data = data.assign(**{cat: data[cat].astype(str).str.split(",")}).explode(cat)
        data[cat] = data[cat].str.strip()
    top = data[cat].value_counts().head(8).index
    return data[data[cat].isin(top)]

plot_data = prepare(df, cat_col, num_col)

def chart_A(data, cat, num):
    """Chart A – Bar chart (mean per category)"""
    order = data.groupby(cat)[num].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x=cat, y=num, order=order,
                palette="Blues_d", estimator="mean", errorbar=None, ax=ax)
    ax.set_title(f"Average {num} by {cat}", fontsize=14)
    ax.set_xlabel(cat); ax.set_ylabel(f"Mean {num}")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    return fig

def chart_B(data, cat, num):
    """Chart B – Box plot (distribution per category)"""
    order = data.groupby(cat)[num].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x=cat, y=num, order=order,
                palette="Set2", ax=ax)
    ax.set_title(f"{num} Distribution by {cat}", fontsize=14)
    ax.set_xlabel(cat); ax.set_ylabel(num)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    return fig

CHARTS      = {"A": chart_A,    "B": chart_B}
CHART_NAMES = {"A": "Bar Chart", "B": "Box Plot"}


defaults = {
    "current_chart": None,   # "A" or "B"
    "chart_shown":   False,
    "start_time":    None,
    "results":       [],     # [{chart, chart_name, elapsed}]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.divider()

if st.button("Show me a chart", use_container_width=False):
    st.session_state.current_chart = random.choice(["A", "B"])
    st.session_state.chart_shown   = True
    st.session_state.start_time    = time.time()

if st.session_state.chart_shown:
    key = st.session_state.current_chart
    fig = CHARTS[key](plot_data, cat_col, num_col)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    if st.button("Did I answer your question?"):
        elapsed = round(time.time() - st.session_state.start_time, 2)
        st.session_state.results.append({
            "chart":      key,
            "chart_name": CHART_NAMES[key],
            "elapsed":    elapsed,
        })
        st.success(f" You answered in **{elapsed} s**  |  Chart shown: **{key} – {CHART_NAMES[key]}**")
        # Reset for next round
        st.session_state.chart_shown   = False
        st.session_state.current_chart = None

if st.session_state.results:
    st.divider()
    st.subheader("A/B Test Results")

    results_df = pd.DataFrame(st.session_state.results)

    summary = (
        results_df
        .groupby(["chart", "chart_name"])["elapsed"]
        .agg(Trials="count", Avg_seconds="mean", Min_seconds="min")
        .reset_index()
        .assign(Avg_seconds=lambda d: d["Avg_seconds"].round(2),
                Min_seconds=lambda d: d["Min_seconds"].round(2))
    )
    st.dataframe(summary, use_container_width=True)

    # Declare winner only if both charts have been tested
    if len(summary) == 2:
        winner = summary.loc[summary["Avg_seconds"].idxmin()]
        st.success(
            f"Chart **{winner['chart']} ({winner['chart_name']})** "
            f"was answered faster on average ({winner['Avg_seconds']} s)"
        )

    if st.button("Reset results"):
        st.session_state.results = []
        st.rerun()
