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

        # --- 入力確認の表示 ---
        st.markdown(f"### 🎯 入力内容")
        st.markdown(f"◎ 本命：**{anchor}**")
        st.markdown(f"ヒモ候補：**{' '.join(himo_list)}**")

        # --- 三連複 買い目の生成 ---
        sanren_pats = list(itertools.combinations(himo_list, 2))
        sanren_kaime = ["".join(sorted([anchor, p1, p2])) for p1, p2 in sanren_pats]

        # --- 三連複 表示 ---
        st.markdown("### ✅ 三連複：買い目とオッズ入力")
        sanren_data = []
        for k in sanren_kaime:
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown(f"▶ **{k}**")
            with cols[1]:
                odd = st.number_input("三連複オッズ", key=f"sanren_{k}", min_value=0.0, step=0.1)
            sanren_data.append((k, odd))

        valid_sanren_data = [(k, o) for k, o in sanren_data if o > 0]

        low_odds = [(k, o) for k, o in valid_sanren_data if o < 3.0]
        if low_odds:
            st.error("🚫 見送り：3倍未満の買い目が含まれているため購入不可")
        else:
            inv_sum = sum([1/o for _, o in valid_sanren_data])
            synth_odds = round(1 / inv_sum, 2)
            st.markdown(f"### 📊 三連複 合成オッズ：**{synth_odds}倍**")

            if synth_odds >= 3.0:
                st.success("✅ 購入可：6点構成で合成オッズ3倍以上クリア")
                st.markdown("💰 推奨：三連複各100円＋Sランクに追加張り")
            else:
                sorted_candidates = sorted(valid_sanren_data, key=lambda x: (-1 if x[1] >= 30.0 else 0, x[1]))
                reduced = sorted_candidates.copy()
                removed = []
                while len(reduced) >= 4:
                    inv_sum_new = sum([1/o for _, o in reduced])
                    synth_new = round(1 / inv_sum_new, 2)
                    if synth_new >= 3.0:
                        st.warning(f"💡 削減後 合成オッズ：{synth_new}倍 → {len(reduced)}点で購入可")
                        st.markdown(f"除外買い目：{', '.join([k for k, _ in removed])}")
                        break
                    removed.append(reduced.pop(0))
                else:
                    st.error("🚫 見送り：削減しても4点未満 or 合成3倍未満")

        st.markdown("---")
        st.markdown("### ✅ 二車複：買い目とオッズ入力")
        nisha_kaime = ["".join(sorted([anchor, h])) for h in himo_list]
        nisha_data = []
        for k in nisha_kaime:
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown(f"▶ **{k}**")
            with cols[1]:
                odd = st.number_input("二車複オッズ", key=f"nisha_{k}", min_value=0.0, step=0.1)
            nisha_data.append((k, odd))

        valid_nisha_data = [(k, o) for k, o in nisha_data if o > 1.4]
        if len(valid_nisha_data) > 4:
            valid_nisha_data = sorted(valid_nisha_data, key=lambda x: -x[1])[:4]

        valid_odds2 = [1/o for _, o in valid_nisha_data]
        if valid_odds2:
            inv_sum2 = sum(valid_odds2)
            synth_odds2 = round(1 / inv_sum2, 2)
            st.markdown(f"### 📊 二車複 合成オッズ：**{synth_odds2}倍**")
            if synth_odds2 >= 1.5:
                st.success("✅ 二車複：購入基準クリア（合成1.5倍以上、最大4点）")
                if len(valid_nisha_data) <= 2:
                    st.markdown("💰 推奨：1〜2点 → 各200〜400円張り")
                else:
                    st.markdown("💰 推奨：各100円（計400円以内）")
            else:
                st.error("🚫 二車複：合成オッズ1.5倍未満 → 見送り")
        else:
            st.info("二車複：有効な買い目が存在しない またはすべてガミ（1.4倍以下）")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("◎とヒモを入力してください。例：◎=5, ヒモ=1 2 3 4")

