import streamlit as st
import itertools
import pandas as pd

st.title("三連複・二車複 買い目評価ツール（7車立て対応）")

# --- 入力 ---
anchor = st.text_input("◎（本命）", placeholder="例：5")
himos = st.text_input("ヒモ（最大4車）", placeholder="例：1 2 3 4")

if anchor and himos:
    try:
        anchor = anchor.strip()
        himo_list = himos.strip().split()

        if anchor in himo_list:
            himo_list.remove(anchor)  # 重複回避

        # --- 三連複 買い目の生成 ---
        sanren_pats = list(itertools.combinations(himo_list, 2))
        sanren_kaime = ["".join(sorted([anchor, p1, p2])) for p1, p2 in sanren_pats]

        # --- 二車複 買い目の生成 ---
        nisha_kaime = ["".join(sorted([anchor, h])) for h in himo_list]

        # --- 三連複 表示 ---
        st.markdown("### ✅ 三連複：買い目とオッズ入力")
        sanren_data = []
        for k in sanren_kaime:
            st.markdown(f"**▶ {k}**")
            odd = st.number_input(f"三連複オッズ: {k}", key=f"sanren_{k}", min_value=1.0, step=0.1)
            sanren_data.append((k, odd))

        valid_odds = [1/o for _, o in sanren_data if o > 0]
        if valid_odds:
            inv_sum = sum(valid_odds)
            synth_odds = round(1 / inv_sum, 2)
            st.markdown(f"### 📊 三連複 合成オッズ：**{synth_odds}倍**")
            if synth_odds >= 3.0:
                st.success("三連複：購入基準クリア（合成3倍以上）")
            else:
                st.warning("三連複：購入見送り（合成オッズ3倍未満）")

        # --- 二車複 表示 ---
        st.markdown("---")
        st.markdown("### ✅ 二車複：買い目とオッズ入力")
        nisha_data = []
        for k in nisha_kaime:
            st.markdown(f"**▶ {k}**")
            odd = st.number_input(f"二車複オッズ: {k}", key=f"nisha_{k}", min_value=1.0, step=0.1)
            nisha_data.append((k, odd))

        valid_odds2 = [1/o for _, o in nisha_data if o > 0]
        if valid_odds2:
            inv_sum2 = sum(valid_odds2)
            synth_odds2 = round(1 / inv_sum2, 2)
            st.markdown(f"### 📊 二車複 合成オッズ：**{synth_odds2}倍**")
            if synth_odds2 >= 1.5:
                st.success("二車複：購入基準クリア（合成1.5倍以上）")
            else:
                st.warning("二車複：購入見送り（合成オッズ1.5倍未満）")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("◎とヒモを入力してください。例：◎=5, ヒモ=1 2 3 4")
