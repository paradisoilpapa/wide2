import streamlit as st
import pandas as pd
from itertools import combinations

st.set_page_config(page_title="三連複・二車複評価ツール", layout="wide")
st.title("🎯 買い目評価アプリ（7車立て対応）")

st.markdown("---")

# --- 入力欄 ---
st.markdown("### ◎ 本命とヒモを入力")
anchor = st.text_input("◎（本命、例：5）", max_chars=1)
sub = st.text_input("ヒモ（例：1234）", max_chars=7)

# --- ランク入力 ---
st.markdown("### 🎯 ランク入力（対応する買い目順に S/A/B など）")
rank_input = st.text_input("ランク入力（例：SABABB）")

# --- オッズ入力 ---
st.markdown("### 💰 オッズ入力（対応する買い目順に半角スペース区切り）")
odds_input = st.text_input("オッズ入力（例：12.5 5.3 7.8 22.1 3.2 6.6）")

# --- データ整形 ---
def sanitize(text):
    return [s for s in text.replace(" ", "").strip() if s.isdigit()]

def make_trios(anchor, subs):
    return list(combinations(sorted([int(anchor)] + subs), 3))

# --- 買い目構成 ---
buy_list = []
sub_digits = sanitize(sub)
if anchor and sub_digits and len(sub_digits) >= 2:
    subs = list(set(sub_digits))
    if anchor in subs:
        subs.remove(anchor)
    base_combis = make_trios(anchor, subs)
    buy_list = ["-".join(map(str, sorted(x))) for x in base_combis if str(anchor) in map(str, x)]

# --- 表の作成 ---
odds = odds_input.strip().split()
ranks = list(rank_input.strip().upper())

data = []
for i, b in enumerate(buy_list):
    odd = float(odds[i]) if i < len(odds) else None
    rank = ranks[i] if i < len(ranks) else ""
    data.append({"買い目": b, "オッズ": odd, "ランク": rank})

df = pd.DataFrame(data)

# --- トリガミ削除処理 ---
gami_removed = df[df["オッズ"] >= 3.0].copy()
total_odd = 1 / (1 / gami_removed["オッズ"]).sum() if not gami_removed.empty else 0

# --- 表示 ---
st.markdown("### 📝 買い目とオッズ一覧")
st.dataframe(df, use_container_width=True)

st.markdown("### 🔍 トリガミ削除後の評価")
if gami_removed.empty:
    st.error("3.0未満の買い目しか存在しないため、見送り対象です。")
elif len(gami_removed) < 4:
    st.warning("削除後、構成が3点以下のため見送り対象です。")
else:
    st.success(f"削除後の合成オッズ：{total_odd:.2f} 倍")

# --- Bランクで低オッズの削除候補表示 ---
st.markdown("### ⚠️ Bランク削除候補（オッズ5.0未満）")
b_candidates = df[(df["ランク"] == "B") & (df["オッズ"] < 5.0)]
if not b_candidates.empty:
    st.dataframe(b_candidates, use_container_width=True)
else:
    st.info("削除候補なし（Bランクかつ5倍未満は存在しません）")

# --- Sランク厚張り対象 ---
st.markdown("### 💸 Sランク厚張り対象")
s_targets = df[df["ランク"] == "S"]
if not s_targets.empty:
    min_row = s_targets.sort_values("オッズ").iloc[0]
    st.write(f"対象：{min_row['買い目']}（{min_row['オッズ']}倍）")
else:
    st.info("Sランクが存在しません")
