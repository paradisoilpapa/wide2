import streamlit as st
import itertools
import pandas as pd

st.title("三連複・二車複 買い目評価ツール（7車立て対応）")

# --- 入力 ---
anchor_input = st.text_input("◎（本命）", placeholder="例：5")
himos_input = st.text_input("ヒモ（最大4車）", placeholder="例：1 2 3 4")

if anchor_input and himos_input:
    try:
        anchor_input = anchor_input.strip()
        himos_input = himos_input.strip()
        anchor_num = int(anchor_input)

        if " " in himos_input:
            himo_str_list = himos_input.split()
        else:
            himo_str_list = list(himos_input)

        himo_str_list = [h for h in himo_str_list if h != ""]
        himo_nums = []
        for h in himo_str_list:
            if h.isdigit():
                himo_nums.append(int(h))

        himo_set = set(himo_nums)
        if anchor_num in himo_set:
            himo_set.remove(anchor_num)
        himo_nums = sorted(himo_set)

        st.markdown(f"### 🎯 入力内容")
        st.markdown(f"◎ 本命：**{anchor_num}**")
        st.markdown(f"ヒモ候補：**{' '.join(map(str, himo_nums))}**")

        trifecta_combos = []
        if len(himo_nums) >= 2:
            for combo in itertools.combinations(himo_nums, 2):
                trifecta_combos.append((anchor_num, *combo))
        else:
            trifecta_combos = []

        st.markdown("### ✅ 三連複：買い目とオッズ入力")
        trifecta_data = []
        for idx, combo in enumerate(trifecta_combos):
            combo_str = "-".join(map(str, sorted(combo)))
            odd = st.number_input(
                f"{combo_str} のオッズ", min_value=0.0, value=0.0, step=0.1,
                key=f"odds_trifecta_{idx}"
            )
            trifecta_data.append((combo_str, odd))

        valid_trifecta_data = [(k, o) for k, o in trifecta_data if o > 0]
        low_odds = [(k, o) for k, o in valid_trifecta_data if o < 3.0]

        if low_odds:
            st.error("🚫 見送り：3倍未満の買い目が含まれているため購入不可")
        else:
            inv_sum = sum((1 / o) for _, o in valid_trifecta_data) if valid_trifecta_data else 0.0
            synth_odds = round(1 / inv_sum, 2) if inv_sum != 0 else 0.0
            st.markdown(f"### 📊 三連複 合成オッズ：**{synth_odds}倍**")

            if synth_odds >= 3.0:
                st.success("✅ 購入可：6点構成で合成オッズ3倍以上クリア")
            else:
                sorted_candidates = sorted(
                    valid_trifecta_data,
                    key=lambda x: (-1 if x[1] >= 30.0 else 0, x[1])
                )
                reduced = sorted_candidates.copy()
                removed = []
                while len(reduced) >= 4:
                    inv_sum_new = sum(1 / o for _, o in reduced)
                    synth_new = round(1 / inv_sum_new, 2) if inv_sum_new != 0 else 0.0
                    if synth_new >= 3.0:
                        st.warning(f"💡 削減後 合成オッズ：{synth_new}倍 → {len(reduced)}点で購入可")
                        st.markdown(f"除外買い目：{', '.join(k for k, _ in removed)}")
                        break
                    removed.append(reduced.pop(0))
                else:
                    st.error("🚫 見送り：削減しても4点未満 or 合成3倍未満")

        st.markdown("---")
        st.markdown("### ✅ 二車複：買い目とオッズ入力")
        nishafuku_combos = [(anchor_num, h) for h in himo_nums]
        nishafuku_data = []
        for idx, pair in enumerate(nishafuku_combos):
            pair_str = "-".join(map(str, sorted(pair)))
            odd = st.number_input(
                f"{pair_str} のオッズ", min_value=0.0, value=0.0, step=0.1,
                key=f"odds_pair_{idx}"
            )
            nishafuku_data.append((pair_str, odd))

        # --- ガミオッズ除外・4点制限・合成オッズ評価 ---
        valid_nishafuku_data = [(k, o) for k, o in nishafuku_data if o > 1.4]
        if not valid_nishafuku_data:
            st.error("🚫 見送り：二車複すべてガミオッズ（1.4倍以下）のため除外")
        else:
            if len(valid_nishafuku_data) > 4:
                valid_nishafuku_data = sorted(valid_nishafuku_data, key=lambda x: -x[1])[:4]

            inv_sum2 = sum(1 / o for _, o in valid_nishafuku_data)
            synth_odds2 = round(1 / inv_sum2, 2) if inv_sum2 != 0 else 0.0
            st.markdown(f"### 📊 二車複 合成オッズ：**{synth_odds2}倍**")

            if synth_odds2 >= 1.5:
                st.success("✅ 二車複：購入基準クリア（合成1.5倍以上、最大4点）")
            else:
                st.error("🚫 二車複：合成オッズ1.5倍未満 → 見送り")

        # --- 勝負レース選別ランク評価の仮表示（条件は今後追加可能） ---
        st.markdown("---")
        st.markdown("### 🏁 勝負レースランク評価（試験実装）")
        st.markdown("- **Sランク**：◎＋ライン＋ヒモ構成で\"ほぼ穴なし\" → 厚張り対象")
        st.markdown("- **Aランク**：◎の信頼度が高く、ヒモの一部は展開依存 → 通常買い")
        st.markdown("- **Bランク**：展開待ち・低確率のヒモが混じる → 削減対象／見送り候補")
        st.info("※ ランク自動評価は今後条件追加により実装予定")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("◎とヒモを入力してください。例：◎=5, ヒモ=1 2 3 4")

