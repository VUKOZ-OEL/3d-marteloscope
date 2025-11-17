import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# --- Konstanta: používáme jen tyto tři sloupce ---
COLUMNS = ["Page", "Author", "Comment"]
WORKSHEET = None  # None = první list (gid=0). Nebo zadej název listu.

# --- Připojení ke Google Sheets (přes secrets [connections.gsheets]) ---
@st.cache_resource
def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)

conn = get_conn()

@st.cache_data(ttl=5)
def load_df() -> pd.DataFrame:
    # Čteme bez usecols, abychom nevyvolali ValueError, když sloupce chybí / jsou jinak pojmenované
    df = conn.read(
        spreadsheet=st.secrets["connections"]["gsheets"]["spreadsheet"],
        worksheet=WORKSHEET,
        ttl=0
    )

    # Pokud list prázdný, připravíme prázdný DF s našimi třemi sloupci
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Normalizace: ponecháme jen námi očekávané sloupce + doplníme chybějící
    # – case-insensitive match pro jistotu (Page vs page apod.)
    cols_lower = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    for col in COLUMNS:
        src = cols_lower.get(col.lower())
        if src is not None:
            out[col] = df[src]
        else:
            out[col] = ""  # když sloupec neexistuje, založíme prázdný

    # Odstranění řádků, které jsou úplně prázdné (všechny 3 sloupce prázdné)
    mask_nonempty = out.apply(lambda r: any(str(r[c]).strip() for c in COLUMNS), axis=1)
    out = out[mask_nonempty].copy()

    return out

def append_row(page: str | None, author: str | None, comment: str):
    # Vložíme nový řádek NA ZAČÁTEK (nové nahoře)
    row = {
        "Page": (page or "").strip(),
        "Author": (author or "").strip(),
        "Comment": comment.strip(),
    }
    df_old = load_df()
    df_new = pd.concat([pd.DataFrame([row], columns=COLUMNS), df_old], ignore_index=True)

    # Přepíšeme celý list novým obsahem – nejjednodušší a spolehlivé
    conn.update(
        spreadsheet=st.secrets["connections"]["gsheets"]["spreadsheet"],
        worksheet=WORKSHEET,
        data=df_new
    )
    load_df.clear()  # vyčistit cache, aby se hned načetla nová data

st.markdown("#### 💬 Comments:")

# --- Formulář pro přidání komentáře ---
with st.form("new_comment"):
    col1, col2 = st.columns([1, 1])
    with col1:
        page = st.text_input("**Page:**", placeholder="optional")
    with col2:
        author = st.text_input("**Author:**", placeholder="optional")
    comment = st.text_area("**Comment:**", placeholder="Your comment", height=100)
    submitted = st.form_submit_button("➕ Add comment")
    if submitted:
        if not comment.strip():
            st.error("**Comment** cannot be empty.")
        else:
            try:
                append_row(page, author, comment)
                st.success("Comment saved sucessfuly.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save comment: {e}")

# --- Filtrování a zobrazení ---
df = load_df()
if df.empty:
    st.caption("Not commented yet.")
else:
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        uniq_pages = ["(all)"] + sorted([p for p in df["Page"].dropna().unique() if str(p).strip() != ""])
        f_page = st.selectbox("Filter by Page", options=uniq_pages, index=0)
    #with fcol2:
        #f_text = st.text_input("Fulltext (Author/Comment)")

    df_view = df.copy()
    if f_page != "(all)":
        df_view = df_view[df_view["Page"] == f_page]


    st.dataframe(
        df_view[COLUMNS],
        use_container_width=True,
        hide_index=True
    )