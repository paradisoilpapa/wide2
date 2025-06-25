import streamlit as st
import re
from itertools import combinations

st.set_page_config(page_title="三連複・二車複判定アプリ", layout="centered")
st.title("🎯 三連複・二車複 判定アプリ（7車立て専用）")

# --- 入力欄 ---
st.subheader("◎本命とヒモを入力")
anchor = st.text_input("◎（本命1車）", "5")
himo = st.text_input("ヒモ（例：1234 または 1 2 3 4）", "1 2 3 4")

odds_input = st.text_area("三連複オッズ入力（6点）", "5.2\n4.0\n6.1\n7.8\n3.3\n9.6")
odds_lines = [float(line) for line in odds_input.strip().replace(" ", "\n").split("\n") if line.strip()]

rank_input = st.text_input("ランク入力（例：SABBBB）", "SABBBB")

# --- 二車複欄（任意） ---
st.subheader("二車複オプション（任意）")
ni_odds_input = st.text_area("2車複オッズ（最大4点）", "2.1\n1.6\n1.9\n2.3")
ni_odds = [float(line) for line in ni_odds_input.strip().replace(" ", "\n").split("\n") if line.strip()]

# --- 正規化＆買い目作成 ---
def normalize_nums(txt):
    return re.findall(r"\d", txt)

anchor = anchor.strip()
himos = normalize_nums(himo)

sanrenpuku = [tuple(sorted([int(anchor), int(a), int(b)])) for a, b in combinations(himos, 2)]

if len(sanrenpuku) != len(odds_lines):
    st.error(f"⚠️ 買い目数 {len(sanrenpuku)} に対してオッズ数 {len(odds_lines)} が一致しません")
    st.stop()

# --- 判定ロジック ---
def combined_odds(odds_list):
    return round(1 / sum([1/o for o in odds_list]), 2)

# --- 三連複 合成オッズチェック ---
valid_indices = [i for i, odd in enumerate(odds_lines) if odd >= 3.0]
valid_ranks = [rank_input[i] for i in valid_indices]

reduced_odds = [odds_lines[i] for i in valid_indices]
reduced_baimoku = [sanrenpuku[i] for i in valid_indices]

# --- 合成計算（削減後 or フル） ---
base_odds = odds_lines if combined_odds(odds_lines) >= 3.0 else reduced_odds
base_set = sanrenpuku if combined_odds(odds_lines) >= 3.0 else reduced_baimoku

final_odds = combined_odds(base_odds)

# --- 表示 ---
st.subheader("三連複 結果")
st.markdown(f"**合成オッズ：{final_odds}倍（{len(base_set)}点）**")

if final_odds >= 3.0 and len(base_set) >= 4:
    st.success("✅ 購入OK")
    for i, (o, b) in enumerate(zip(base_odds, base_set)):
        st.write(f"{b}：{o}倍")
        
    # Sランク抽出（補足）
    s_odds = [odds_lines[i] for i in range(len(rank_input)) if rank_input[i] == "S"]
    if s_odds:
        s_min = min(s_odds)
        st.info(f"Sランク内最低オッズ：{s_min}倍 → 厚張り候補")
else:
    st.warning("⛔ 購入NG（点数 or 合成オッズ未達）")

# --- 削除候補 ---
st.subheader("削除候補（Bランク）")
b_indices = [i for i, r in enumerate(rank_input) if r == "B"]
b_candidates = [(i, sanrenpuku[i], odds_lines[i]) for i in b_indices if odds_lines[i] < 5.0]

for i, b, o in b_candidates:
    st.write(f"候補：{b} → {o}倍")

# --- 二車複 判定 ---
st.subheader("二車複 結果")
ni_valid = [o for o in ni_odds if o >= 1.5]

if len(ni_valid) >= 3:
    ni_combined = combined_odds(ni_valid)
    st.markdown(f"**合成オッズ：{ni_combined}倍（{len(ni_valid)}点）**")
    st.success("✅ 購入OK")
    s_odds_ni = min(ni_valid)
    st.info(f"最低オッズ（厚張り候補）：{s_odds_ni}倍")
else:
    st.warning("⛔ 二車複：購入NG（オッズ or 点数不足）")
