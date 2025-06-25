import streamlit as st
from itertools import combinations

# タイトルと説明
st.title("7車立て競輪レース　三連複・二車複オッズ計算アプリ")
st.write("◎（本命）とヒモを選択すると、自動的に三連複および二車複の買い目が生成されます。各買い目のオッズを入力すると、合成オッズが計算され、購入基準の判定が表示されます。")

# --- 戦略ルールチェック（ステップ1） ---
st.subheader("📘 レース対象条件チェック")
st.markdown("- ✅ **レースが7車立てか** → このアプリは7車専用です")
st.markdown("- ✅ **三連複構成が6点以内で組めるか** → 自動制限あり")
st.markdown("- ✅ **◎が明確かつ構成内にいるか** → 不在なら見送り")

# 入力欄
horses = list(range(1, 8))  # 1～7車
main = st.selectbox("◎ 本命", horses)
options = [h for h in horses if h != main]
himo = st.multiselect("ヒモ (最大4車)", options)

if len(himo) > 4:
    st.error("ヒモは最大4車まで選択してください。")
    st.stop()

himo_sorted = sorted(himo)
trifecta_combs = [tuple(sorted((main,)+comb)) for comb in combinations(himo_sorted, 2)] if len(himo_sorted) >= 2 else []
pair_combs = [tuple(sorted((main, h))) for h in himo_sorted] if len(himo_sorted) >= 1 else []

# --- 三連複 ---
st.subheader("🧠 三連複：買い目構成とオッズ評価")
if trifecta_combs:
    header_cols = st.columns([1, 1])
    header_cols[0].write("**買い目**")
    header_cols[1].write("**オッズ**")
    trif_odds = []
    for comb in trifecta_combs:
        comb_str = "".join(map(str, comb))
        cols = st.columns([1, 1])
        cols[0].write(comb_str)
        odd_value = cols[1].number_input(
            f"{comb_str} のオッズ", min_value=0.0, value=0.0, step=0.1,
            label_visibility="hidden", key=f"odds_trif_{comb_str}"
        )
        trif_odds.append((comb_str, odd_value))

    low_odds = [k for k, v in trif_odds if 0 < v < 3.0]
    if low_odds:
        st.warning(f"単独オッズ3倍未満あり（{', '.join(low_odds)}）→ ケン推奨")

    valid_odds = [1/o for _, o in trif_odds if o > 0]
    if valid_odds:
        combined = round(1.0 / sum(valid_odds), 2)
        st.write(f"合成オッズ: {combined}倍")
        if combined >= 3.0:
            st.success("購入基準クリア（3倍以上）")
        elif len(valid_odds) >= 4:
            st.warning("削減検討：高オッズ or 弱ヒモを削る")
        else:
            st.error("削減後4点未満 → ケン確定")
    else:
        st.write("合成オッズ: -")
else:
    st.info("三連複はヒモが2車以上必要です")

# --- 二車複 ---
st.subheader("🔁 二車複：補助判断とオッズ評価")
if pair_combs:
    header_cols = st.columns([1, 1])
    header_cols[0].write("**買い目**")
    header_cols[1].write("**オッズ**")
    pair_odds = []
    for comb in pair_combs:
        comb_str = "".join(map(str, comb))
        cols = st.columns([1, 1])
        cols[0].write(comb_str)
        odd_value = cols[1].number_input(
            f"{comb_str} のオッズ", min_value=0.0, value=0.0, step=0.1,
            label_visibility="hidden", key=f"odds_pair_{comb_str}"
        )
        pair_odds.append((comb_str, odd_value))

    valid_odds = [1/o for _, o in pair_odds if o > 0]
    gami = [k for k, v in pair_odds if 0 < v <= 1.4]
    if gami:
        st.warning(f"ガミオッズあり（{', '.join(gami)}）→ 除外推奨")
    if valid_odds:
        combined = round(1.0 / sum(valid_odds), 2)
        st.write(f"合成オッズ: {combined}倍")
        if combined >= 1.5:
            st.success("二車複：購入候補")
        else:
            st.warning("見送り（合成1.5倍未満）")
    else:
        st.write("合成オッズ: -")
else:
    st.info("二車複はヒモが1車以上必要です")

# --- ケン判断まとめ ---
st.markdown("---")
st.subheader("🛡 ケン基準まとめ")
st.markdown("- 🔸 三連複構成が組めない → 精度・的中率の土台崩壊")
st.markdown("- 🔸 合成オッズが3倍未満（単独低オッズ含む） → 低期待値")
st.markdown("- 🔸 削減後も買い目4点未満 → ケン確定")
st.markdown("- 🔸 二車複がガミオッズ構成のみ → 無益投資")
