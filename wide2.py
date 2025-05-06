import streamlit as st
import pandas as pd
from itertools import combinations

st.set_page_config(page_title="7車版期待値計算", layout="wide")

st.title("⭐2複＆ワイド 7車版期待値計算アプリ⭐")

# --- シンボルスコア定義 ---
symbol_scores = {
    "◎": 0.57,
    "〇": 0.49,
    "▲": 0.36,
    "△": 0.20,
    "×": 0.17
}
default_score = 0.08
anchor_symbols = ["◎", "〇", "▲"]

# --- シンボル入力 ---
st.subheader("▼ シンボル入力（最大5名）")
symbols_input = {}
cols = st.columns(len(symbol_scores))
for i, sym in enumerate(symbol_scores.keys()):
    with cols[i]:
        symbols_input[sym] = st.text_input(f"{sym}（{symbol_scores[sym]}）", key=f"symbol_{sym}")

# --- ボタンで組合せ生成 ---
if st.button("組合せ生成"):
    filled = [(sym, symbols_input[sym].strip()) for sym in symbol_scores if symbols_input[sym].strip()]
    used_numbers = set(num for _, num in filled)
    all_numbers = [str(i) for i in range(1, 8)]
    missing_numbers = [n for n in all_numbers if n not in used_numbers]
    combined = filled + [("無", n) for n in missing_numbers[:2]]
    combined_sorted = sorted(combined, key=lambda x: int(x[1]))
    number_map = {sym: num for sym, num in combined}

    # 無印を除いた対象組合せ作成
    odds_target = []
    for a, b in combinations(combined_sorted, 2):
        sym_a, num_a = a
        sym_b, num_b = b
        if "無" not in [sym_a, sym_b]:
            odds_target.append(((sym_a, sym_b), f"{num_a}-{num_b}"))

    st.session_state['odds_target'] = odds_target
    st.session_state['number_map'] = number_map
    st.success("組合せを生成しました。下にオッズ入力欄が出ます。")

# --- オッズ入力欄表示 ---
if 'odds_target' in st.session_state:
    st.subheader("▼ オッズ個別入力（絶対に消えない）")
    odds_inputs = {}
    for (sym_pair, num_pair) in st.session_state['odds_target']:
        label = f"{num_pair}（{sym_pair[0]}{sym_pair[1]}）"
        odds_inputs[num_pair] = st.number_input(label, min_value=0.0, step=0.1, format="%.1f", key=f"odds_{num_pair}")
    st.session_state['odds_inputs'] = odds_inputs

# --- 期待値計算ボタン ---
if st.button("期待値計算"):
    if 'odds_inputs' not in st.session_state:
        st.error("先に『組合せ生成』を押してください。")
    else:
        odds_inputs = st.session_state['odds_inputs']
        odds_target = st.session_state['odds_target']
        number_map = st.session_state['number_map']

        anchor_ev_totals = {sym: 0.0 for sym in anchor_symbols}
        anchor_ev_counts = {sym: 0 for sym in anchor_symbols}

        results = []
        ev_map_by_number = {}

        for (a, b), pair in odds_target:
            odds = odds_inputs.get(pair)
            if odds is None or odds <= 0:
                continue
            s1 = symbol_scores.get(a, default_score)
            s2 = symbol_scores.get(b, default_score)
            win_rate = 0.5 * s1 * s2
            ev = win_rate * odds
            if a in anchor_symbols:
                anchor_ev_totals[a] += ev
                anchor_ev_counts[a] += 1
            if b in anchor_symbols:
                anchor_ev_totals[b] += ev
                anchor_ev_counts[b] += 1
            results.append((pair, f"{a}{b}", round(win_rate,5), odds, round(ev,5), ""))

            for num in [int(number_map.get(a, 0)), int(number_map.get(b, 0))]:
                if num not in ev_map_by_number:
                    ev_map_by_number[num] = []
                ev_map_by_number[num].append(ev)

        anchor_avg_ev = {sym: (anchor_ev_totals[sym] / anchor_ev_counts[sym]) if anchor_ev_counts[sym] else 0 for sym in anchor_symbols}
        anchor_symbol, anchor_max_ev = max(anchor_avg_ev.items(), key=lambda x: x[1])

        final_results = []
        ev_by_number = {num: round(sum(vals)/len(vals), 3) for num, vals in ev_map_by_number.items()}

        for (pair, marks, win_rate, odds, ev, _) in results:
            is_anchor = anchor_symbol in marks and ev >= 1.0
            final_results.append((pair, marks, win_rate, odds, ev, "※" if is_anchor else "□ ケン"))

        df_result = pd.DataFrame(final_results, columns=["車番", "記号", "勝率", "オッズ", "期待値", "評価"])
        st.subheader("▼ 計算結果（期待値表）")
        st.dataframe(df_result)

        # --- シンボル別：対応車番と平均期待値 ---
        st.subheader("▼ シンボル別：対応車番と平均期待値")
        symbol_ev_summary = []
        for sym in symbol_scores.keys():
            car_num_str = symbols_input[sym].strip()
            if car_num_str and car_num_str.isdigit():
                car_num = int(car_num_str)
                avg_ev = ev_by_number.get(car_num, 0.0)
                symbol_ev_summary.append((sym, car_num, avg_ev))

        if symbol_ev_summary:
            df_symbol_ev = pd.DataFrame(symbol_ev_summary, columns=["記号", "車番", "平均期待値"])
            st.dataframe(df_symbol_ev)
        else:
            st.markdown("- 該当なし")

        # --- ワイド候補出力 ---
        st.subheader("▼ ワイド候補（期待値補完）")
        wide_pairs = []
        wide_candidate = [(k, v) for k, v in ev_by_number.items() if 0.7 <= v < 1.3]
        over1 = [(k, v) for k, v in ev_by_number.items() if v >= 1.0]

        if wide_candidate and over1:
            wide_candidate_sorted = sorted(wide_candidate, key=lambda x: abs(1.0 - x[1]))
            wide_base = wide_candidate_sorted[0][0]
            for num, ev2 in over1:
                if num == wide_base:
                    continue
                avg = (ev_by_number[wide_base] + ev2) / 2
                if avg >= 1.0:
                    wide_pairs.append((wide_base, num, round(avg, 3)))

        if wide_pairs:
            df_wide = pd.DataFrame(wide_pairs, columns=["ワイド軸", "相手軸", "平均期待値"])
            st.dataframe(df_wide)
        else:
            st.markdown("- 該当なし")

        st.subheader("▼ 無印との必要オッズ参考")
        for sym, score in symbol_scores.items():
            min_odds = round(25 / score, 2)
            st.markdown(f"- 無×{sym}：{min_odds}倍以上")
