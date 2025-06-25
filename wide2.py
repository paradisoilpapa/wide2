import streamlit as st

st.set_page_config(page_title="三連複・二車複判断アプリ", layout="centered")

st.title("🎯 三連複・二車複 購入判断アプリ（7車立て専用）")

st.subheader("① 三連複 買い目入力")

triplet_combos = []
triplet_odds = []
triplet_confidences = []

for i in range(6):
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        combo = st.text_input(f"買い目{i+1}（例：1-2-3）", key=f"tri_combo_{i}")
    with col2:
        odds = st.number_input("オッズ", min_value=0.0, value=10.0, step=0.1, key=f"tri_odds_{i}")
    with col3:
        conf = st.selectbox("自信度", ["-", "S", "A", "B"], key=f"tri_conf_{i}")
    if combo and odds > 0 and conf != "-":
        triplet_combos.append(combo)
        triplet_odds.append(odds)
        triplet_confidences.append(conf)

st.subheader("② 二車複（補助） 買い目入力")

pair_combos = []
pair_odds = []

for i in range(4):
    col1, col2 = st.columns([3, 2])
    with col1:
        combo = st.text_input(f"買い目{i+1}（例：1-2）", key=f"pair_combo_{i}")
    with col2:
        odds = st.number_input("オッズ", min_value=0.0, value=3.0, step=0.1, key=f"pair_odds_{i}")
    if combo and odds > 0:
        pair_combos.append(combo)
        pair_odds.append(odds)

# --- 判定ロジック ---
def evaluate_combos(odds_list, confidences, combos):
    if any(o < 3.0 for o in odds_list):
        return "❌ 3連複に3倍未満のオッズが含まれています → ケン（見送り）", [], []

    inv_sum = sum(1 / o for o in odds_list)
    combined_odds = round(1 / inv_sum, 2) if inv_sum > 0 else 0

    if combined_odds < 3.0:
        to_cut = [(combos[i], odds_list[i]) for i in range(len(confidences)) if confidences[i] == "B"]
        remaining = [(combos[i], odds_list[i], confidences[i]) for i in range(len(confidences)) if confidences[i] != "B"]
        return f"⚠ 合成オッズが3倍未満です（{combined_odds}倍） → Bランクから削減候補を検討", to_cut, remaining

    return f"✅ 合成オッズ：{combined_odds}倍 → 購入OK", [], []

def evaluate_pairs(odds_list):
    valid_odds = [o for o in odds_list if o >= 1.5]
    if not valid_odds:
        return "❌ ガミ回避できる二車複がありません → 見送り"
    inv_sum = sum(1 / o for o in valid_odds)
    combined_odds = round(1 / inv_sum, 2) if inv_sum > 0 else 0
    if combined_odds < 1.5:
        return f"❌ 合成オッズが1.5倍未満です（{combined_odds}倍） → 見送り"
    return f"✅ 二車複 合成オッズ：{combined_odds}倍 → 購入OK"

def recommend_thick_bet(confidences, odds_list, combos):
    s_candidates = [(combos[i], odds_list[i]) for i in range(len(confidences)) if confidences[i] == "S"]
    if not s_candidates:
        return "厚張り対象：なし（Sランクが存在しないか未入力）"
    s_candidates.sort(key=lambda x: x[1])
    return f"厚張り対象：{s_candidates[0][0]}（オッズ {s_candidates[0][1]}倍）"

# --- 出力 ---
st.subheader("③ 判定結果")

if triplet_combos:
    st.markdown("### 三連複 判定")
    triplet_msg, to_cut, remaining = evaluate_combos(triplet_odds, triplet_confidences, triplet_combos)
    st.info(triplet_msg)
    if to_cut:
        st.warning("🔻 Bランク削減候補：")
        for c, o in to_cut:
            st.markdown(f"- {c}（{o}倍）")
    if remaining:
        st.success("✅ 削減後の構成候補：")
        for c, o, conf in remaining:
            st.markdown(f"- {c}（{o}倍）｜自信度：{conf}")
    st.markdown("### 厚張り判定")
    st.info(recommend_thick_bet(triplet_confidences, triplet_odds, triplet_combos))

if pair_combos:
    st.markdown("### 二車複 判定")
    st.info(evaluate_pairs(pair_odds))
