# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple, Union

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ分析（開催区分別）", layout="wide")
st.title("ヴェロビ 組み方分析（可変頭数・開催区分別／全体集計） v2.7+条件付き2着分布")

# -------- 基本設定 --------
MAX_FIELD = 9

DAY_OPTIONS = ["L", "F2", "F1", "G"]

DAY_LABELS = {
    "L": "ガールズ（L級）",
    "F2": "F2",
    "F1": "F1",
    "G": "G",
}

RANK_SYMBOLS = {
    1: "carFR順位１位", 2: "carFR順位２位", 3: "carFR順位３位", 4: "carFR順位４位", 5: "carFR順位５位", 6: "carFR順位６位",
    7: "carFR順位７～位", 8: "carFR順位７～位", 9: "carFR順位７～位",
}
def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, "carFR順位７～位")


def parse_rankline(s: str) -> List[str]:
    """
    V順位（例: '1432...'）をパース。
    - 許容文字: 1～9
    - 4～MAX_FIELD 桁
    - 重複なし
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    if not s.isdigit() or not (4 <= len(s) <= MAX_FIELD):
        return []
    if any(ch not in "123456789" for ch in s):
        return []
    if len(set(s)) != len(s):
        return []
    return list(s)

def parse_finish(s: str) -> List[str]:
    """
    着順（～3桁まで使用、余分は切り捨て）
    - 許容文字: 1～9
    - 重複は先着優先で無視
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    s = "".join(ch for ch in s if ch in "123456789")
    out: List[str] = []
    for ch in s:
        if ch not in out:
            out.append(ch)
        if len(out) == 3:
            break
    return out


# =========================
# 追加：条件付き2着分布（累積対応）
# =========================
WINNER_RANKS = (1, 2, 3)  # 1着が評価1～3位のときだけ見る
RR_OUT = "圏外"           # 2着車がV順位内にいない等

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
        # このwr条件の総数
        total = 0
        for rr in cols:
            total += int(pair_counts.get((wr, rr), 0))

        # 回数行
        row_c = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            row_c[str(rr)] = int(pair_counts.get((wr, rr), 0))
        count_rows.append(row_c)

        # ％行（wr自身は「ありえない」ので空欄にする：1→1, 2→2, 3→3）
        row_p = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            if isinstance(rr, int) and rr == wr:
                row_p[str(rr)] = None
            else:
                v = int(pair_counts.get((wr, rr), 0))
                row_p[str(rr)] = round(100.0 * v / total, 1) if total > 0 else 0.0
        pct_rows.append(row_p)

    df_count = pd.DataFrame(count_rows)
    df_pct = pd.DataFrame(pct_rows)
    return df_count, df_pct


# -------- 入力タブ --------
input_tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（ランク別回数）", "分析結果"])

byrace_rows: List[Dict] = []
agg_counts_manual: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0})

# ★追加：前日まで累積（条件付き2着分布）
pair_counts_manual: Dict[PairKey, int] = defaultdict(int)


# A. 日次手入力（開催区分は1回だけ指定→全行に適用）
with input_tabs[0]:
    st.subheader("日次手入力（開催区分別・最大12R）")

    day_global = st.selectbox(
        "開催区分（この選択を全レースに適用）",
        DAY_OPTIONS,
        key="global_day",
        format_func=lambda k: DAY_LABELS.get(k, k),
    )

    cols_hdr = st.columns([1,1,2,1.5])
    cols_hdr[0].markdown("**R**")
    cols_hdr[1].markdown("**頭数**")
    cols_hdr[2].markdown("**V順位(例: 1432...)**")
    cols_hdr[3].markdown("**着順(～3桁)**")

    for i in range(1, 13):
        c1, c2, c3, c4 = st.columns([1,1,2,1.5])
        rid = c1.text_input("", key=f"rid_{i}", value=str(i))
        field = c2.number_input("", min_value=4, max_value=MAX_FIELD, value=min(7, MAX_FIELD), key=f"field_{i}")
        vline = c3.text_input("", key=f"vline_{i}", value="")
        fin = c4.text_input("", key=f"fin_{i}", value="")

        vorder = parse_rankline(vline)
        finish = parse_finish(fin)

        any_input = any([vorder, finish])
        if any_input:
            if vorder and (4 <= field <= MAX_FIELD) and len(vorder) <= field:
                byrace_rows.append({
                    "day": day_global,
                    "race": rid,
                    "field": field,
                    "vorder": vorder,
                    "finish": finish,
                })
            else:
                st.warning(f"R{rid}: 入力不整合（V順位/頭数を確認）。V順位は頭数桁以下、4～{MAX_FIELD}車のみ対象。")


# B. 前日までの集計（手入力）
with input_tabs[1]:
    st.subheader("前日までの集計（開催区分 × ランク（◎〜無）の入賞回数）")

    MU_BIN_R = 7  # 無 まとめ先のランク番号（内部は7に集約）
    def add_rec(day: str, r: int, N: int, C1: int, C2: int, C3: int):
        rec = agg_counts_manual[(day, r)]
        rec["N"]  += int(N)
        rec["C1"] += int(C1)
        rec["C2"] += int(C2)
        rec["C3"] += int(C3)

    for day in DAY_OPTIONS:
        st.markdown(f"**{DAY_LABELS[day]}**")
        ch = st.columns([1,1,1,1])
        ch[0].markdown("ランク（表示）")
        ch[1].markdown("N_r")
        ch[2].markdown("1着回数")
        ch[3].markdown("2着回数／3着回数")

        for r in range(1, 7):
            c0, c1, c2, c3 = st.columns([1,1,1,1])
            c0.write(rank_symbol(r))
            N  = c1.number_input("", key=f"agg_{day}_N_{r}",  min_value=0, value=0)
            C1 = c2.number_input("", key=f"agg_{day}_C1_{r}", min_value=0, value=0)
            c3_cols = c3.columns(2)
            C2 = c3_cols[0].number_input("", key=f"agg_{day}_C2_{r}", min_value=0, value=0)
            C3 = c3_cols[1].number_input("", key=f"agg_{day}_C3_{r}", min_value=0, value=0)
            if any([N, C1, C2, C3]):
                add_rec(day, r, N, C1, C2, C3)

        c0, c1, c2, c3 = st.columns([1,1,1,1])
        c0.write("carFR順位７～位")
        N_mu  = c1.number_input("", key=f"agg_{day}_N_{MU_BIN_R}",  min_value=0, value=0)
        C1_mu = c2.number_input("", key=f"agg_{day}_C1_{MU_BIN_R}", min_value=0, value=0)
        c3_cols = c3.columns(2)
        C2_mu = c3_cols[0].number_input("", key=f"agg_{day}_C2_{MU_BIN_R}", min_value=0, value=0)
        C3_mu = c3_cols[1].number_input("", key=f"agg_{day}_C3_{MU_BIN_R}", min_value=0, value=0)
        if any([N_mu, C1_mu, C2_mu, C3_mu]):
            add_rec(day, MU_BIN_R, N_mu, C1_mu, C2_mu, C3_mu)

    # ★追加：前日までの累積（条件付き2着分布）
    st.divider()
    st.subheader("前日までの集計（復習用）：1着が評価1〜3位のとき、2着の評価順位（累積回数）")
    st.caption("ここは開催区分なし（全体累積）。『回数』だけ入力。未入力は0のままでOK。")

    cols = list(range(1, MAX_FIELD + 1)) + [RR_OUT]

    # ヘッダ
    h = st.columns([1] + [1]*len(cols))
    h[0].markdown("**条件：1着の評価順位**")
    for j, rr in enumerate(cols, start=1):
        h[j].markdown(f"**2着={rr}**")

    # 行（wr=1..3）
    for wr in WINNER_RANKS:
        row_cols = st.columns([1] + [1]*len(cols))
        row_cols[0].write(f"評価{wr}位が1着")

        for j, rr in enumerate(cols, start=1):
            # 1→1, 2→2, 3→3 は理屈上ありえないので入力欄を出さない（固定0）
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


# -------- 集計構築（開催区分別 + 全体） --------
rank_counts_daily: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"N":0, "C1":0, "C2":0, "C3":0})

for row in byrace_rows:
    day = row["day"]
    vorder = row["vorder"]
    finish = row["finish"]

    car_by_rank = {i+1: vorder[i] for i in range(len(vorder))}

    L = len(vorder)
    for i in range(1, min(L, MAX_FIELD) + 1):
        rank_counts_daily[(day, i)]["N"] += 1
        car = car_by_rank[i]
        if len(finish) >= 1 and finish[0] == car:
            rank_counts_daily[(day, i)]["C1"] += 1
        if len(finish) >= 2 and finish[1] == car:
            rank_counts_daily[(day, i)]["C2"] += 1
        if len(finish) >= 3 and finish[2] == car:
            rank_counts_daily[(day, i)]["C3"] += 1

for (day, r), rec in agg_counts_manual.items():
    rank_counts_daily[(day, r)]["N"]  += rec["N"]
    rank_counts_daily[(day, r)]["C1"] += rec["C1"]
    rank_counts_daily[(day, r)]["C2"] += rec["C2"]
    rank_counts_daily[(day, r)]["C3"] += rec["C3"]

rank_counts_total: Dict[int, Dict[str, int]] = {r: {"N":0, "C1":0, "C2":0, "C3":0} for r in range(1, MAX_FIELD + 1)}
for (day, r), rec in rank_counts_daily.items():
    for k in ("N","C1","C2","C3"):
        rank_counts_total[r][k] += rec[k]


# ★追加：日次手入力分から「条件付き2着」回数を作る
pair_counts_daily: Dict[PairKey, int] = defaultdict(int)

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])

    if len(finish) < 2 or len(vorder) < 1:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}

    win_car = finish[0]
    run_car = finish[1]

    win_rank = car_to_rank.get(win_car)
    if win_rank not in WINNER_RANKS:
        continue

    run_rank = car_to_rank.get(run_car, RR_OUT)
    pair_counts_daily[(win_rank, run_rank)] += 1

# 合算（累積）
pair_counts_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair_counts_daily.items():
    pair_counts_total[k] += int(v)
for k, v in pair_counts_manual.items():
    pair_counts_total[k] += int(v)


# -------- 出力タブ --------
with input_tabs[2]:
    # ★目的の分布：1-2.. / 2-1.. / 3-1..
    st.subheader("復習：1着が評価1〜3位のとき、2着の評価順位分布（累積）")

    df_count, df_pct = build_conditional_runnerup_tables(pair_counts_total, MAX_FIELD)

    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df_count, use_container_width=True, hide_index=True)

    st.markdown("### 割合%（1→1 / 2→2 / 3→3 は空欄）")
    st.dataframe(df_pct, use_container_width=True, hide_index=True)

    st.divider()

    # 既存出力（そのまま）
    st.subheader("開催区分別：ランク別 入賞テーブル（◎〜carFR順位７～位）")
    for day in DAY_OPTIONS:
        rows_out = []
        for r in range(1, 7):
            rec = rank_counts_daily.get((day, r), {"N":0,"C1":0,"C2":0,"C3":0})
            N, C1, C2, C3 = rec["N"], rec["C1"], rec["C2"], rec["C3"]
            def rate(x, n): return round(100*x/n, 1) if n>0 else None
            rows_out.append({
                "ランク": rank_symbol(r),
                "出走数N": N,
                "1着回数": C1,
                "2着回数": C2,
                "3着回数": C3,
                "1着率%": rate(C1,N),
                "連対率%": rate(C1+C2,N),
                "3着内率%": rate(C1+C2+C3,N),
            })

        N=C1=C2=C3=0
        for r in range(7, MAX_FIELD+1):
            rec = rank_counts_daily.get((day, r), {"N":0,"C1":0,"C2":0,"C3":0})
            N  += rec["N"]
            C1 += rec["C1"]
            C2 += rec["C2"]
            C3 += rec["C3"]
        def rate(x, n): return round(100*x/n, 1) if n>0 else None
        rows_out.append({
            "ランク": "carFR順位７～位",
            "出走数N": N,
            "1着回数": C1,
            "2着回数": C2,
            "3着回数": C3,
            "1着率%": rate(C1,N),
            "連対率%": rate(C1+C2,N),
            "3着内率%": rate(C1+C2+C3,N),
        })

        df_day = pd.DataFrame(rows_out)
        st.markdown(f"### {DAY_LABELS[day]}")
        st.dataframe(df_day, use_container_width=True, hide_index=True)

    rows_total = []
    for r in range(1, 7):
        rec = rank_counts_total.get(r, {"N":0,"C1":0,"C2":0,"C3":0})
        N, C1, C2, C3 = rec["N"], rec["C1"], rec["C2"], rec["C3"]
        def rate(x, n): return round(100*x/n, 1) if n>0 else None
        rows_total.append({
            "ランク": rank_symbol(r),
            "出走数N": N,
            "1着回数": C1,
            "2着回数": C2,
            "3着回数": C3,
            "1着率%": rate(C1,N),
            "連対率%": rate(C1+C2,N),
            "3着内率%": rate(C1+C2+C3,N),
        })

    N=C1=C2=C3=0
    for r in range(7, MAX_FIELD+1):
        rec = rank_counts_total.get(r, {"N":0,"C1":0,"C2":0,"C3":0})
        N  += rec["N"]
        C1 += rec["C1"]
        C2 += rec["C2"]
        C3 += rec["C3"]
    def rate(x, n): return round(100*x/n, 1) if n>0 else None
    rows_total.append({
        "ランク": "carFR順位７～位",
        "出走数N": N,
        "1着回数": C1,
        "2着回数": C2,
        "3着回数": C3,
        "1着率%": rate(C1,N),
        "連対率%": rate(C1+C2,N),
        "3着内率%": rate(C1+C2+C3,N),
    })

    st.markdown("### 全体")
    st.dataframe(pd.DataFrame(rows_total), use_container_width=True, hide_index=True)
