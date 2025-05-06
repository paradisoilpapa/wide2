import streamlit as st
import pandas as pd
from itertools import combinations

st.set_page_config(page_title="7車版期待値計算", layout="wide")
st.title("⭐ 7車版期待値計算アプリ（完全移植＋ワイド対応版）⭐")

# --- シンボル定義 ---
symbol_scores = {
    "◎": 0.57,
    "〇": 0.49,
    "▲": 0.36,
    "△": 0.20,
    "×": 0.17
}
default_score = 0.08
anchor_symbols = ["◎", "〇", "▲"]

# --- シンボル入力欄 ---
st.subheader("▼ シンボル入力")
symbol_inputs = {}
cols = st.columns(len(symbol_scores))
for i, sym in enumerate(symbol_scores):
    with cols[i]:
        symbol_inputs[sym] = st.text_input(f"{sym}（{symbol_scores[sym]}）", key=f"symbol_{sym}")

# --- 組合せ生成 ---
if st.button("組合せ生成"):
    filled = [(sym, symbol_inputs[sym].strip()) for sym in symbol_scores if symbol_inputs[sym].strip()]
    used_numbers = set(num for _, num in filled)
    all_numbers = [str(i) for i in range(1, 8)]
    missing_numbers = [n for n in all_numbers if n not in used_numbers]
    combined = filled + [("無", n) for n in missing_numbers[:2]]
    combined_sorted = sorted(combined, key=lambda x: int(x[1]))

    odds_target = []
    number_map = {sym: num for sym, num in combined}
    for a, b in combinations(combined_sorted, 2):
        sym_a, num_a = a
        sym_b, num_b = b
        if "無" not in [sym_a, sym_b]:
            odds_target.append(((sym_a, sym_b), f"{num_a}-{num_b}"))

    st.session_state['odds_target'] = odds_target
    st.session_state['number_map'] = number_map
    st.success("組合せを生成しました。オッズを入力してください。")

# --- オッズ入力 ---
odds_inputs = {}
if 'odds_target' in st.session_state:
    st.subheader("▼ オッズ入力")
    for (sym_pair, num_pair) in st.session_state['odds_target']:
        label = f"{num_pair}（{sym_pair[0]}{sym_pair[1]}）"
        odds_inputs[num_pair] = st.number_input(label, min_value=0.0, step=0.1, format="%.1f", key=f"odds_{num_pair}")
    st.session_state['odds_inputs'] = odds_inputs

# --- 期待値計算 ---
if st.button("期待値計算"):
    odds_target = st.session_state['odds_target']
    number_map = st.session_state['number_map']
    odds_inputs = st.session_state['odds_inputs']

    anchor_ev_totals = {sym: 0.0 for sym in anchor_symbols}
    anchor_ev_counts = {sym: 0 for sym in anchor_symbols}
    symbol_ev_map = {sym: [] for sym in symbol_scores}
    symbol_combos = {sym: [] for sym in symbol_scores}

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
        results.append((pair, f"{a}{b}", win_rate, odds, ev))
        if a != "無":
            symbol_ev_map[a].append((ev, number_map.get(a, '?')))
            if ev >= 1.0:
                symbol_combos[a].append((pair, f"{a}{b}", ev, False))
        if b != "無":
            symbol_ev_map[b].append((ev, number_map.get(b, '?')))
            if ev >= 1.0:
                symbol_combos[b].append((pair, f"{a}{b}", ev, False))
        for num in [int(number_map.get(a, 0)), int(number_map.get(b, 0))]:
            if num not in ev_map_by_number:
                ev_map_by_number[num] = []
            ev_map_by_number[num].append(ev)

    anchor_avg_ev = {sym: anchor_ev_totals[sym]/anchor_ev_counts[sym] if anchor_ev_counts[sym] else 0 for sym in anchor_symbols}
    anchor_symbol, _ = max(anchor_avg_ev.items(), key=lambda x: x[1])

    final_results = []
    for (pair, marks, win_rate, odds, ev) in results:
        is_anchor = anchor_symbol in marks and ev >= 1.0
        final_results.append((pair, marks, round(win_rate,5), odds, round(ev,5), "※" if is_anchor else "□ ケン"))

    df_result = pd.DataFrame(final_results, columns=["車番", "記号", "勝率", "オッズ", "期待値", "評価"])
    st.subheader("▼ 計算結果（期待値表）")
    st.dataframe(df_result)

    # シンボル別まとめ
    symbol_summary = []
    for sym, ev_list in symbol_ev_map.items():
        if ev_list:
            avg_ev = sum(ev for ev, _ in ev_list) / len(ev_list)
            nums = sorted(set(num for _, num in ev_list))
            mark = "※" if sym == anchor_symbol else ""
            symbol_summary.append((sym, ','.join(nums), len(ev_list), round(avg_ev, 5), mark))

    df_summary = pd.DataFrame(symbol_summary, columns=["シンボル", "対応車番", "該当件数", "平均期待値", "評価"])
    st.subheader("▼ シンボル別まとめ")
    st.dataframe(df_summary)

    # 高期待値買い目出力
    st.subheader("▼ シンボル別：期待値1.0以上の買い目")
    for sym, combos in symbol_combos.items():
        if combos:
            anchor_mark = "※" if sym == anchor_symbol else ""
            st.markdown(f"**■ {sym}{anchor_mark}**")
            for pair, marks, ev, _ in sorted(combos, key=lambda x: -x[2]):
                star = "※" if anchor_symbol in marks and ev >= 1.0 else ""
                st.markdown(f"- {pair}（{marks}）: 期待値={ev:.2f}{star}")

    # ワイド候補出力
    st.subheader("▼ ワイド候補（期待値補完）")
    wide_pairs = []
    under1 = [(k, v) for k, v in {num: sum(vals)/len(vals) for num, vals in ev_map_by_number.items()}.items() if v < 1.0]
    over1 = [(k, v) for k, v in {num: sum(vals)/len(vals) for num, vals in ev_map_by_number.items()}.items() if v >= 1.0]

    if under1 and over1:
        under1_sorted = sorted(under1, key=lambda x: abs(1.0 - x[1]))
        wide_base = under1_sorted[0][0]
        for num, ev2 in over1:
            avg = ((sum(ev_map_by_number[wide_base])/len(ev_map_by_number[wide_base])) + ev2) / 2
            if avg >= 1.0:
                wide_pairs.append((wide_base, num, round(avg, 3)))

    if wide_pairs:
        df_wide = pd.DataFrame(wide_pairs, columns=["ワイド軸", "相手軸", "平均期待値"])
        st.dataframe(df_wide)
    else:
        st.markdown("- 該当なし")

    # 無印との参考出力
    st.subheader("▼ 無印との必要オッズ参考")
    for sym, score in symbol_scores.items():
        min_odds = round(25 / score, 2)
        st.markdown(f"- 無×{sym}：{min_odds}倍以上")
