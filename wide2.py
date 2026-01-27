# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple, Union

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜2着順位分布 → ランク別入賞 v1.2")

# -------- 基本設定 --------
MAX_FIELD = 7  # ★7車まで（8・9は不要）

RANK_SYMBOLS = {
    1: "carFR順位１位", 2: "carFR順位２位", 3: "carFR順位３位", 4: "carFR順位４位",
    5: "carFR順位５位", 6: "carFR順位６位",
    7: "carFR順位７～位",
}
def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, "carFR順位７～位")


def parse_rankline(s: str) -> List[str]:
    """
    V順位（例: '1432...'）をパース。
    - 許容文字: 1～7（※7車運用）
    - 4～MAX_FIELD 桁（4～7）
    - 重複なし
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    if not s.isdigit() or not (4 <= len(s) <= MAX_FIELD):
        return []
    if any(ch not in "1234567" for ch in s):  # ★1～7のみ
        return []
    if len(set(s)) != len(s):
        return []
    return list(s)

def parse_finish(s: str) -> List[str]:
    """
    着順（～3桁まで使用、余分は切り捨て）
    - 許容文字: 1～7（※7車運用）
    - 重複は先着優先で無視
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    s = "".join(ch for ch in s if ch in "1234567")  # ★1～7のみ
    out: List[str] = []
    for ch in s:
        if ch not in out:
            out.append(ch)
        if len(out) == 3:
            break
    return out


# =========================
# 条件付き2着分布（累積対応）
# =========================
WINNER_RANKS = (1, 2, 3, 4, 5)  # ★1着が評価1～5位のときだけ見る
RR_OUT = "圏外"                 # 2着車がV順位内にいない等（保険）

PairKey = Tuple[int, Union[int, str]]  # (winner_rank, runnerup_rank or "圏外")

def build_conditional_runnerup_tables(pair_counts: Dict[PairKey, int], max_field: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    pair_counts[(wr, rr)] = 回数
    出力：回数ピボット、割合%ピボット
    """
    cols = list(range(1, max_field + 1)) + [RR_OUT]
    count_rows = []
    pct_rows = []

    for wr in WINNER_RANKS:
        total = sum(int(pair_counts.get((wr, rr), 0)) for rr in cols)

        # 回数
        row_c = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            row_c[str(rr)] = int(pair_counts.get((wr, rr), 0))
        count_rows.append(row_c)

        # 割合（wr→wr は理屈上ありえないので空欄）
        row_p = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            if isinstance(rr, int) and rr == wr:
                row_p[str(rr)] = None
            else:
                v = int(pair_counts.get((wr, rr), 0))
                row_p[str(rr)] = round(100.0 * v / total, 1) if total > 0 else 0.0
        pct_rows.append(row_p)

    return pd.DataFrame(count_rows), pd.DataFrame(pct_rows)


# =========================
# 入力タブ
# =========================
tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

byrace_rows: List[Dict] = []

# 前日まで：ランク別（全体）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0})

# 前日まで：条件付き2着分布（全体）
pair_counts_manual: Dict[PairKey, int] = defaultdict(int)


# -------- A. 日次手入力 --------
with tabs[0]:
    st.subheader("日次手入力（全体・最大12R）")

    cols_hdr = st.columns([1, 1, 2, 1.5])
    cols_hdr[0].markdown("**R**")
    cols_hdr[1].markdown("**頭数**")
    cols_hdr[2].markdown("**V順位(例: 1432...)**")
    cols_hdr[3].markdown("**着順(～3桁)**")

    for i in range(1, 13):
        c1, c2, c3, c4 = st.columns([1, 1, 2, 1.5])
        rid = c1.text_input("", key=f"rid_{i}", value=str(i))

        # ★8・9車は排除（4～7のみ）
        field = c2.number_input("", min_value=4, max_value=MAX_FIELD, value=7, key=f"field_{i}")

        vline = c3.text_input("", key=f"vline_{i}", value="")
        fin = c4.text_input("", key=f"fin_{i}", value="")

        vorder = parse_rankline(vline)
        finish = parse_finish(fin)

        any_input = any([vorder, finish])
        if any_input:
            if vorder and len(vorder) <= field:
                byrace_rows.append({
                    "race": rid,
                    "field": field,
                    "vorder": vorder,
                    "finish": finish,
                })
            else:
                st.warning(f"R{rid}: 入力不整合（V順位/頭数を確認）。V順位は頭数桁以下、4～{MAX_FIELD}車のみ対象。")


# -------- B. 前日までの集計（累積） ※順番統一：2着分布→ランク別 --------
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")

    # ② 条件付き2着分布（全体）※先頭
    st.markdown("## 2着順位分布（累積・回数）")
    st.caption("復習用：1着が評価1〜5位のとき、2着の評価順位の回数を入力。1→1 / 2→2 / 3→3 / 4→4 / 5→5 は入力不要。")

    cols = list(range(1, MAX_FIELD + 1)) + [RR_OUT]

    h = st.columns([1] + [1]*len(cols))
    h[0].markdown("**条件：1着の評価順位**")
    for j, rr in enumerate(cols, start=1):
        h[j].markdown(f"**2着={rr}**")

    for wr in WINNER_RANKS:
        row_cols = st.columns([1] + [1]*len(cols))
        row_cols[0].write(f"評価{wr}位が1着")

        for j, rr in enumerate(cols, start=1):
            if isinstance(rr, int) and rr == wr:
                row_cols[j].write("-")
                continue
            v = row_cols[j].number_input(
                "",
                key=f"pair_prev_wr{wr}_rr{rr}",
                min_value=0,
                value=0
            )
            if v:
                pair_counts_manual[(wr, rr)] += int(v)

    st.divider()

    # ① ランク別入賞回数（全体）※後ろ
    st.markdown("## ランク別 入賞回数（累積）")

    MU_BIN_R = 7  # 7位以降まとめ

    def add_rank_rec(r: int, N: int, C1: int, C2: int, C3: int):
        rec = agg_rank_manual[r]
        rec["N"]  += int(N)
        rec["C1"] += int(C1)
        rec["C2"] += int(C2)
        rec["C3"] += int(C3)

    hdr = st.columns([1, 1, 1, 1])
    hdr[0].markdown("ランク（表示）")
    hdr[1].markdown("N_r")
    hdr[2].markdown("1着回数")
    hdr[3].markdown("2着回数／3着回数")

    for r in range(1, 7):
        c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
        c0.write(rank_symbol(r))
        N  = c1.number_input("", key=f"aggN_{r}",  min_value=0, value=0)
        C1 = c2.number_input("", key=f"aggC1_{r}", min_value=0, value=0)
        c3_cols = c3.columns(2)
        C2 = c3_cols[0].number_input("", key=f"aggC2_{r}", min_value=0, value=0)
        C3 = c3_cols[1].number_input("", key=f"aggC3_{r}", min_value=0, value=0)
        if any([N, C1, C2, C3]):
            add_rank_rec(r, N, C1, C2, C3)

    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    c0.write("carFR順位７～位")
    N_mu  = c1.number_input("", key=f"aggN_{MU_BIN_R}",  min_value=0, value=0)
    C1_mu = c2.number_input("", key=f"aggC1_{MU_BIN_R}", min_value=0, value=0)
    c3_cols = c3.columns(2)
    C2_mu = c3_cols[0].number_input("", key=f"aggC2_{MU_BIN_R}", min_value=0, value=0)
    C3_mu = c3_cols[1].number_input("", key=f"aggC3_{MU_BIN_R}", min_value=0, value=0)
    if any([N_mu, C1_mu, C2_mu, C3_mu]):
        add_rank_rec(MU_BIN_R, N_mu, C1_mu, C2_mu, C3_mu)


# =========================
# 集計：日次 + 前日まで累積 を合算
# =========================

# --- ランク別（日次） ---
rank_daily = {r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, MAX_FIELD + 1)}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if not vorder:
        continue

    car_by_rank = {i + 1: vorder[i] for i in range(len(vorder))}
    L = len(vorder)
    for r in range(1, min(L, MAX_FIELD) + 1):
        rank_daily[r]["N"] += 1
        car = car_by_rank[r]
        if len(finish) >= 1 and finish[0] == car:
            rank_daily[r]["C1"] += 1
        if len(finish) >= 2 and finish[1] == car:
            rank_daily[r]["C2"] += 1
        if len(finish) >= 3 and finish[2] == car:
            rank_daily[r]["C3"] += 1

# --- ランク別（合算） ---
rank_total = {r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, MAX_FIELD + 1)}
for r in range(1, MAX_FIELD + 1):
    for k in ("N", "C1", "C2", "C3"):
        rank_total[r][k] += rank_daily[r][k]

# 前日まで累積（1～6と7+）
for r, rec in agg_rank_manual.items():
    if r == 7:
        rank_total[7]["N"]  += rec["N"]
        rank_total[7]["C1"] += rec["C1"]
        rank_total[7]["C2"] += rec["C2"]
        rank_total[7]["C3"] += rec["C3"]
    else:
        rank_total[r]["N"]  += rec["N"]
        rank_total[r]["C1"] += rec["C1"]
        rank_total[r]["C2"] += rec["C2"]
        rank_total[r]["C3"] += rec["C3"]

# --- 条件付き2着分布（日次） ---
pair_counts_daily: Dict[PairKey, int] = defaultdict(int)

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if len(finish) < 2 or not vorder:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    win_car = finish[0]
    run_car = finish[1]

    win_rank = car_to_rank.get(win_car)
    if win_rank not in WINNER_RANKS:
        continue

    run_rank = car_to_rank.get(run_car, RR_OUT)
    pair_counts_daily[(win_rank, run_rank)] += 1

# --- 条件付き2着分布（合算） ---
pair_counts_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair_counts_daily.items():
    pair_counts_total[k] += int(v)
for k, v in pair_counts_manual.items():
    pair_counts_total[k] += int(v)


# =========================
# 出力：分析結果（順番統一：2着分布→ランク表）
# =========================
with tabs[2]:
    st.subheader("2着順位分布（全体累積）｜1着が評価1〜5位のとき")

    df_count, df_pct = build_conditional_runnerup_tables(pair_counts_total, MAX_FIELD)

    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df_count, use_container_width=True, hide_index=True)

    st.markdown("### 割合%（1→1 / 2→2 / 3→3 / 4→4 / 5→5 は空欄）")
    st.dataframe(df_pct, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("ランク別 入賞テーブル（全体累積）")

    def rate(x, n):
        return round(100.0 * x / n, 1) if n > 0 else None

    rows_out = []
    for r in range(1, 7):
        rec = rank_total.get(r, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
        N, C1, C2, C3 = rec["N"], rec["C1"], rec["C2"], rec["C3"]
        rows_out.append({
            "ランク": rank_symbol(r),
            "出走数N": N,
            "1着回数": C1,
            "2着回数": C2,
            "3着回数": C3,
            "1着率%": rate(C1, N),
            "連対率%": rate(C1 + C2, N),
            "3着内率%": rate(C1 + C2 + C3, N),
        })

    # 7+（7..MAX_FIELD をまとめ）※MAX_FIELD=7なので実質「7位のみ」だが形は維持
    N = C1 = C2 = C3 = 0
    for r in range(7, MAX_FIELD + 1):
        rec = rank_total.get(r, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
        N  += rec["N"]
        C1 += rec["C1"]
        C2 += rec["C2"]
        C3 += rec["C3"]

    rows_out.append({
    "ランク": "carFR順位７～位",
    "出走数N": N,
    "1着回数": C1,
    "2着回数": C2,
    "3着回数": C3,
    "1着率%": rate(C1, N),
    "連対率%": rate(C1 + C2, N),
    "3着内率%": rate(C1 + C2 + C3, N),
})

# ★6位＋7位 合算（3連複のヒモ参考用）
rec6 = rank_total.get(6, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
rec7 = rank_total.get(7, {"N": 0, "C1": 0, "C2": 0, "C3": 0})

N67  = rec6["N"]  + rec7["N"]
C167 = rec6["C1"] + rec7["C1"]
C267 = rec6["C2"] + rec7["C2"]
C367 = rec6["C3"] + rec7["C3"]

rows_out.append({
    "ランク": "carFR順位６～７位（合算）",
    "出走数N": N67,
    "1着回数": C167,
    "2着回数": C267,
    "3着回数": C367,
    "1着率%": rate(C167, N67),
    "連対率%": rate(C167 + C267, N67),
    "3着内率%": rate(C167 + C267 + C367, N67),
})

st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
