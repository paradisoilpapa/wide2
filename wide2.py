# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple, Union

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜1→2順位分布 ＋ ランク別入賞 v1.4")

# =========================
# 基本設定（7車レース）
# =========================
FIELD_SIZE = 7
RR_OUT = "圏外"  # 5車入力モード時のみ使用

# V順位入力モード（5車 or 7車）
v_mode = st.radio(
    "V順位入力モード",
    options=["上位5車入力", "7車フル入力"],
    horizontal=True,
    index=0,
)

PRED_RANKS = 5 if v_mode == "上位5車入力" else 7
WINNER_RANKS = tuple(range(1, PRED_RANKS + 1))  # 1着条件も入力モードに合わせる

RANK_SYMBOLS = {
    1: "carFR順位１位",
    2: "carFR順位２位",
    3: "carFR順位３位",
    4: "carFR順位４位",
    5: "carFR順位５位",
    6: "carFR順位６位",
    7: "carFR順位７位",
}


def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, f"carFR順位{r}位")


PairKey = Tuple[int, Union[int, str]]  # (winner_rank, other_rank or "圏外")


def parse_rankline(s: str, pred_ranks: int) -> List[str]:
    """
    V順位（例: 5車='14325', 7車='1432567'）をパース
    - 許容文字: 1～7
    - 桁数: pred_ranks 固定
    - 重複なし
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    if not s.isdigit() or len(s) != pred_ranks:
        return []
    if any(ch not in "1234567" for ch in s):
        return []
    if len(set(s)) != len(s):
        return []
    return list(s)


def parse_finish(s: str) -> List[str]:
    """
    着順（～3桁まで使用）
    - 許容文字: 1～7
    - 重複は先着優先で無視
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    s = "".join(ch for ch in s if ch in "1234567")
    out: List[str] = []
    for ch in s:
        if ch not in out:
            out.append(ch)
        if len(out) == 3:
            break
    return out


def build_conditional_tables(
    pair_counts: Dict[PairKey, int],
    pred_ranks: int,
    field_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    pair_counts[(wr, rr)] = 回数
    出力：回数ピボット、割合%ピボット
    ※ wr==rr のセルは回数/割合ともに None（空欄）
    ※ N は wr==rr を除外した合計
    ※ 5車入力モード: 列 = 1..5 + 圏外
       7車入力モード: 列 = 1..7
    """
    if pred_ranks == field_size:
        cols = list(range(1, field_size + 1))  # 1..7
    else:
        cols = list(range(1, pred_ranks + 1)) + [RR_OUT]  # 1..5 + 圏外

    count_rows = []
    pct_rows = []

    for wr in WINNER_RANKS:
        total = 0
        for rr in cols:
            if isinstance(rr, int) and rr == wr:
                continue
            total += int(pair_counts.get((wr, rr), 0))

        row_c = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            if isinstance(rr, int) and rr == wr:
                row_c[str(rr)] = None
            else:
                row_c[str(rr)] = int(pair_counts.get((wr, rr), 0))
        count_rows.append(row_c)

        row_p = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            if isinstance(rr, int) and rr == wr:
                row_p[str(rr)] = None
            else:
                v = int(pair_counts.get((wr, rr), 0))
                row_p[str(rr)] = round(100.0 * v / total, 1) if total > 0 else 0.0
        pct_rows.append(row_p)

    return pd.DataFrame(count_rows), pd.DataFrame(pct_rows)


def rate(x: int, n: int):
    return round(100.0 * x / n, 1) if n > 0 else None


# =========================
# Tabs
# =========================
tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

# 日次の入力行
byrace_rows: List[Dict] = []

# 前日まで：ランク別（1～PRED_RANKS）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(
    lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0}
)

# 前日まで：1→2（評価順位）
pair12_manual: Dict[PairKey, int] = defaultdict(int)

# =========================
# A. 日次手入力
# =========================
with tabs[0]:
    st.subheader("日次手入力（7車固定・最大12R）")

    example_v = "14325" if PRED_RANKS == 5 else "1432567"
    st.caption(f"V順位は「評価順」を{PRED_RANKS}桁で入力（例：{example_v}）。着順は～3桁。")

    cols_hdr = st.columns([1, 1, 2, 1.5])
    cols_hdr[0].markdown("**R**")
    cols_hdr[1].markdown("**頭数**")
    cols_hdr[2].markdown(f"**V順位({PRED_RANKS}桁・例:{example_v})**")
    cols_hdr[3].markdown("**着順(～3桁)**")

    for i in range(1, 13):
        c1, c2, c3, c4 = st.columns([1, 1, 2, 1.5])

        rid = c1.text_input("", key=f"rid_{i}", value=str(i))
        c2.write(str(FIELD_SIZE))
        vline = c3.text_input("", key=f"vline_{i}", value="")
        fin = c4.text_input("", key=f"fin_{i}", value="")

        vorder = parse_rankline(vline, PRED_RANKS)
        finish = parse_finish(fin)

        any_input = any([vline.strip(), fin.strip()])
        if any_input:
            if vorder:
                byrace_rows.append(
                    {
                        "race": rid,
                        "vorder": vorder,  # 上位5車 or 7車フル
                        "finish": finish,  # ～3着
                    }
                )
            else:
                st.warning(f"R{rid}: V順位は{PRED_RANKS}桁で入力してください（例：{example_v}）。")

# =========================
# B. 前日までの集計（累積）
# =========================
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")

    # 5車入力モードでは「圏外」を残す、7車フル入力では1～7のみ
    if PRED_RANKS == FIELD_SIZE:
        cols_12 = list(range(1, FIELD_SIZE + 1))  # 1..7
    else:
        cols_12 = list(range(1, PRED_RANKS + 1)) + [RR_OUT]  # 1..5 + 圏外

    # ---- 1→2（累積入力）
    st.markdown("## 1→2 着順位分布（累積・回数）")
    if PRED_RANKS == FIELD_SIZE:
        st.caption("1着が評価1〜7位のとき、2着の評価順位の回数を入力。1→1 / 2→2 / ... / 7→7 は空欄（入力不可）。")
    else:
        st.caption("1着が評価1〜5位のとき、2着の評価順位の回数を入力。1→1 / 2→2 / ... / 5→5 は空欄（入力不可）。5車入力外は『圏外』へ。")

    h = st.columns([1.5] + [1] * len(cols_12))
    h[0].markdown("**条件：1着の評価順位**")
    for j, rr in enumerate(cols_12, start=1):
        h[j].markdown(f"**2着={rr}**")

    for wr in WINNER_RANKS:
        row_cols = st.columns([1.5] + [1] * len(cols_12))
        row_cols[0].write(f"評価{wr}位が1着")
        for j, rr in enumerate(cols_12, start=1):
            if isinstance(rr, int) and rr == wr:
                row_cols[j].write("")  # 同順位セルは空欄
                continue

            v = row_cols[j].number_input(
                "",
                key=f"pair12_prev_wr{wr}_rr{rr}",
                min_value=0,
                value=0,
            )
            if v:
                pair12_manual[(wr, rr)] += int(v)

    st.divider()

    # ---- ランク別 入賞回数（累積入力）
    st.markdown("## ランク別 入賞回数（累積）")
    st.caption(f"V順位入力モードが {PRED_RANKS}車 なので、1～{PRED_RANKS}位まで入力。Nは各順位が存在したレース数（通常は同じ値）です。")

    hdr = st.columns([1.5, 1, 1, 1.5])
    hdr[0].markdown("**ランク**")
    hdr[1].markdown("**出走数N**")
    hdr[2].markdown("**1着回数**")
    hdr[3].markdown("**2着回数 / 3着回数**")

    def add_rank_rec(r: int, N: int, C1: int, C2: int, C3: int):
        rec = agg_rank_manual[r]
        rec["N"] += int(N)
        rec["C1"] += int(C1)
        rec["C2"] += int(C2)
        rec["C3"] += int(C3)

    for r in range(1, PRED_RANKS + 1):
        c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1.5])
        c0.write(rank_symbol(r))
        N = c1.number_input("", key=f"aggN_{r}", min_value=0, value=0)
        C1 = c2.number_input("", key=f"aggC1_{r}", min_value=0, value=0)
        c3_cols = c3.columns(2)
        C2 = c3_cols[0].number_input("", key=f"aggC2_{r}", min_value=0, value=0)
        C3 = c3_cols[1].number_input("", key=f"aggC3_{r}", min_value=0, value=0)

        if any([N, C1, C2, C3]):
            add_rank_rec(r, N, C1, C2, C3)

# =========================
# 集計：日次 + 前日まで累積 を合算
# =========================

# --- ランク別（日次） ---
rank_daily: Dict[int, Dict[str, int]] = {
    r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, PRED_RANKS + 1)
}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if not vorder:
        continue

    car_by_rank = {i + 1: vorder[i] for i in range(len(vorder))}  # 1..PRED_RANKS
    for r in range(1, PRED_RANKS + 1):
        rank_daily[r]["N"] += 1
        car = car_by_rank.get(r)
        if car is None:
            continue

        if len(finish) >= 1 and finish[0] == car:
            rank_daily[r]["C1"] += 1
        if len(finish) >= 2 and finish[1] == car:
            rank_daily[r]["C2"] += 1
        if len(finish) >= 3 and finish[2] == car:
            rank_daily[r]["C3"] += 1

# --- ランク別（合算） ---
rank_total: Dict[int, Dict[str, int]] = {
    r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, PRED_RANKS + 1)
}

for r in range(1, PRED_RANKS + 1):
    for k in ("N", "C1", "C2", "C3"):
        rank_total[r][k] += rank_daily[r][k]

for r, rec in agg_rank_manual.items():
    if r in rank_total:
        rank_total[r]["N"] += rec["N"]
        rank_total[r]["C1"] += rec["C1"]
        rank_total[r]["C2"] += rec["C2"]
        rank_total[r]["C3"] += rec["C3"]

# --- 1→2（日次） ---
pair12_daily: Dict[PairKey, int] = defaultdict(int)

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if len(finish) < 2 or not vorder:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}  # 入力された順位範囲
    win_car = finish[0]
    run_car = finish[1]

    win_rank = car_to_rank.get(win_car)
    if win_rank not in WINNER_RANKS:
        continue

    # 5車入力時は6,7位を「圏外」に集約。7車入力時は6/7も実順位で入る
    run_rank = car_to_rank.get(run_car, RR_OUT)
    pair12_daily[(win_rank, run_rank)] += 1

# --- 1→2（合算） ---
pair12_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair12_daily.items():
    pair12_total[k] += int(v)
for k, v in pair12_manual.items():
    pair12_total[k] += int(v)

# =========================
# 出力：分析結果
# =========================
with tabs[2]:
    if PRED_RANKS == FIELD_SIZE:
        st.subheader("1→2 着順位分布（全体累積）｜1着が評価1〜7位のとき")
    else:
        st.subheader("1→2 着順位分布（全体累積）｜1着が評価1〜5位のとき")

    df12_count, df12_pct = build_conditional_tables(pair12_total, PRED_RANKS, FIELD_SIZE)

    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df12_count, use_container_width=True, hide_index=True)

    st.markdown("### 割合%（同順位セルは空欄）")
    st.dataframe(df12_pct, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("ランク別 入賞テーブル（全体累積）")

    rows_out = []

    # 実測（1～PRED_RANKS）
    for r in range(1, PRED_RANKS + 1):
        rec = rank_total.get(r, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
        N, C1, C2, C3 = rec["N"], rec["C1"], rec["C2"], rec["C3"]
        rows_out.append(
            {
                "ランク": rank_symbol(r),
                "出走数N": N,
                "1着回数": C1,
                "2着回数": C2,
                "3着回数": C3,
                "1着率%": rate(C1, N),
                "連対率%": rate(C1 + C2, N),
                "3着内率%": rate(C1 + C2 + C3, N),
            }
        )

    # 5車入力モードのときだけ、6～7位推定合算を表示
    if PRED_RANKS < FIELD_SIZE:
        N_base = rank_total.get(1, {"N": 0})["N"]  # 通常は総レース数

        sum_c1 = sum(rank_total.get(r, {"C1": 0})["C1"] for r in range(1, PRED_RANKS + 1))
        sum_c2 = sum(rank_total.get(r, {"C2": 0})["C2"] for r in range(1, PRED_RANKS + 1))
        sum_c3 = sum(rank_total.get(r, {"C3": 0})["C3"] for r in range(1, PRED_RANKS + 1))

        label = f"carFR順位{PRED_RANKS+1}～{FIELD_SIZE}位（推定合算）"

        C1_rest = max(0, N_base - sum_c1)
        C2_rest = max(0, N_base - sum_c2)
        C3_rest = max(0, N_base - sum_c3)

        rows_out.append(
            {
                "ランク": label,
                "出走数N": N_base,
                "1着回数": C1_rest,
                "2着回数": C2_rest,
                "3着回数": C3_rest,
                "1着率%": rate(C1_rest, N_base),
                "連対率%": rate(C1_rest + C2_rest, N_base),
                "3着内率%": rate(C1_rest + C2_rest + C3_rest, N_base),
            }
        )

    st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
