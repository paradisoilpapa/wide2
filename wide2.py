# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜2着分布 → 3連複分布 → ランク別入賞 v1.6（7車固定）")

# -------- 基本設定 --------
MAX_FIELD = 7              # ★ 7車固定
WINNER_RANKS = (1, 2, 3)   # 1着が評価1～3位を対象

RANK_SYMBOLS = {
    1: "carFR順位１位",
    2: "carFR順位２位",
    3: "carFR順位３位",
    4: "carFR順位４位",
    5: "carFR順位５位",
    6: "carFR順位６位",
    7: "carFR順位７～位",
}
def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, "carFR順位７～位")


def parse_rankline(s: str) -> List[str]:
    """V順位（例: '1432...'）をパース（7車固定）。"""
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
    """着順（～3桁まで使用、余分は切り捨て）"""
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
# 2着分布（条件付き）
# =========================
PairKey = Tuple[int, int]  # (1着の評価順位wr, 2着の評価順位rr)

def build_runnerup_tables(pair_counts: Dict[PairKey, int], max_field: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(range(1, max_field + 1))
    rows_c, rows_p = [], []

    for wr in WINNER_RANKS:
        total = sum(int(pair_counts.get((wr, rr), 0)) for rr in cols)

        row_c = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            row_c[str(rr)] = int(pair_counts.get((wr, rr), 0))
        rows_c.append(row_c)

        row_p = {"1着の評価順位": wr, "N": total}
        for rr in cols:
            if rr == wr:
                row_p[str(rr)] = None
            else:
                v = int(pair_counts.get((wr, rr), 0))
                row_p[str(rr)] = round(100.0 * v / total, 1) if total > 0 else 0.0
        rows_p.append(row_p)

    return pd.DataFrame(rows_c), pd.DataFrame(rows_p)


# =========================
# 3連複分布（順番なし：wr と 残り2枠の組）
#   key = (wr, a, b) where a < b, a/b は評価順位（2着と3着を区別しない）
# =========================
TrifukuKey = Tuple[int, int, int]  # (wr, a, b) ただし a<b

def trifuku_matrix_counts(tri: Dict[TrifukuKey, int], wr: int, max_field: int) -> pd.DataFrame:
    ranks = list(range(1, max_field + 1))
    cols = [str(j) for j in ranks]
    idx  = [str(i) for i in ranks]

    data = []
    for i_s in idx:
        i = int(i_s)
        row = []
        for j_s in cols:
            j = int(j_s)

            # 無効セル：
            # - i==j（同じ順位の重複はない）
            # - i==wr or j==wr（wrは別枠なので相手2枠に出ない）
            # - i>j は下三角は使わない（a<b の片側だけに集約）
            invalid = (i == j) or (i == wr) or (j == wr) or (i > j)

            if invalid:
                row.append(None)
            else:
                a, b = i, j  # i<j のみ採用
                row.append(int(tri.get((wr, a, b), 0)))
        data.append(row)

    return pd.DataFrame(data, index=idx, columns=cols, dtype="object")

def trifuku_matrix_pct(tri: Dict[TrifukuKey, int], wr: int, max_field: int) -> pd.DataFrame:
    df_c = trifuku_matrix_counts(tri, wr, max_field)
    df_p = df_c.copy()

    # wrごとの総数（有効セルのみ）
    total = 0
    for i in df_c.index:
        for c in df_c.columns:
            v = df_c.loc[i, c]
            if v is None:
                continue
            total += int(v)

    if total <= 0:
        # 0埋め（Noneはそのまま）
        for i in df_p.index:
            for c in df_p.columns:
                if df_p.loc[i, c] is None:
                    continue
                df_p.loc[i, c] = 0.0
        return df_p

    for i in df_p.index:
        for c in df_p.columns:
            v = df_c.loc[i, c]
            if v is None:
                df_p.loc[i, c] = None
            else:
                df_p.loc[i, c] = round(100.0 * int(v) / total, 1)
    return df_p


# =========================
# Tabs
# =========================
tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

byrace_rows: List[Dict] = []

# 前日まで：ランク別（全体）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0})

# 前日まで：2着分布（全体）
pair_counts_manual: Dict[PairKey, int] = defaultdict(int)

# 前日まで：3連複（順番なし）分布（全体）
trifuku_counts_manual: Dict[TrifukuKey, int] = defaultdict(int)


# -------- 日次手入力 --------
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
                st.warning(f"R{rid}: 入力不整合（V順位/頭数を確認）。")


# -------- 前日まで累積入力（順番統一：2着分布→3連複→ランク別） --------
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")

    # 1) 2着分布（累積・回数）
    st.markdown("## 2着順位分布（累積・回数）")
    cols = list(range(1, MAX_FIELD + 1))
    h = st.columns([1] + [1] * len(cols))
    h[0].markdown("**条件：1着の評価順位**")
    for j, rr in enumerate(cols, start=1):
        h[j].markdown(f"**2着={rr}**")

    for wr in WINNER_RANKS:
        row_cols = st.columns([1] + [1] * len(cols))
        row_cols[0].write(f"評価{wr}位が1着")
        for j, rr in enumerate(cols, start=1):
            if rr == wr:
                row_cols[j].write("-")
                continue
            v = row_cols[j].number_input("", key=f"pair_prev_wr{wr}_rr{rr}", min_value=0, value=0)
            if v:
                pair_counts_manual[(wr, rr)] += int(v)

    st.divider()

    # 2) 3連複（累積・回数）＝ 1着がwrのとき、残り2枠(a,b)（順番なし）
    st.markdown("## 3連複 分布（累積・回数）｜1着=wr のとき『相手2枠（順番なし）』")
    st.caption("※三連単ではありません。2着と3着の順は無視して a-b（a<b）で入力します。")

    ranks = list(range(1, MAX_FIELD + 1))

    for wr in WINNER_RANKS:
        with st.expander(f"1着=評価{wr}位 のとき（相手2枠 a-b の回数表）", expanded=(wr == 1)):
            cols_j = [str(j) for j in ranks]
            rows_i = [str(i) for i in ranks]

            hh = st.columns([1] + [1] * len(cols_j))
            hh[0].markdown("**a\\b**")
            for j, b_s in enumerate(cols_j, start=1):
                hh[j].markdown(f"**{b_s}**")

            for i_s in rows_i:
                i = int(i_s)
                row = st.columns([1] + [1] * len(cols_j))
                row[0].write(i_s)

                for j, b_s in enumerate(cols_j, start=1):
                    b = int(b_s)

                    # 無効：同順位 / wrを含む / 下三角(i>=b)は使わない
                    invalid = (i == b) or (i == wr) or (b == wr) or (i > b)
                    if invalid:
                        row[j].write("-")
                        continue

                    v = row[j].number_input(
                        "",
                        key=f"tri_prev_wr{wr}_a{i}_b{b}",
                        min_value=0,
                        value=0
                    )
                    if v:
                        a = i
                        trifuku_counts_manual[(wr, a, b)] += int(v)

    st.divider()

    # 3) ランク別入賞回数（累積）
    st.markdown("## ランク別 入賞回数（累積）")
    MU_BIN_R = 7  # 7車なので実質7位

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
# 集計（日次 + 前日まで累積 を合算）
# =========================
pair_counts_daily: Dict[PairKey, int] = defaultdict(int)
trifuku_counts_daily: Dict[TrifukuKey, int] = defaultdict(int)
rank_daily = {r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, MAX_FIELD + 1)}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if not vorder:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}

    # ランク別（日次）
    for r in range(1, min(len(vorder), MAX_FIELD) + 1):
        rank_daily[r]["N"] += 1
        car = vorder[r - 1]
        if len(finish) >= 1 and finish[0] == car:
            rank_daily[r]["C1"] += 1
        if len(finish) >= 2 and finish[1] == car:
            rank_daily[r]["C2"] += 1
        if len(finish) >= 3 and finish[2] == car:
            rank_daily[r]["C3"] += 1

    # 2着分布（日次）
    if len(finish) >= 2:
        wr = car_to_rank.get(finish[0])
        rr = car_to_rank.get(finish[1])
        if wr in WINNER_RANKS and rr is not None:
            pair_counts_daily[(wr, rr)] += 1

    # 3連複（日次）＝ wr と 残り2枠（順番なし）
    if len(finish) >= 3:
        wr = car_to_rank.get(finish[0])
        r2 = car_to_rank.get(finish[1])
        r3 = car_to_rank.get(finish[2])
        if wr in WINNER_RANKS and r2 is not None and r3 is not None:
            # 相手2枠は wr と重複しない＆互いに重複しない
            if (r2 != wr) and (r3 != wr) and (r2 != r3):
                a, b = sorted((r2, r3))
                trifuku_counts_daily[(wr, a, b)] += 1

# 合算：2着分布
pair_counts_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair_counts_daily.items():
    pair_counts_total[k] += int(v)
for k, v in pair_counts_manual.items():
    pair_counts_total[k] += int(v)

# 合算：3連複（順番なし）
trifuku_counts_total: Dict[TrifukuKey, int] = defaultdict(int)
for k, v in trifuku_counts_daily.items():
    trifuku_counts_total[k] += int(v)
for k, v in trifuku_counts_manual.items():
    trifuku_counts_total[k] += int(v)

# 合算：ランク別
rank_total = {r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, MAX_FIELD + 1)}
for r in range(1, MAX_FIELD + 1):
    for k in ("N", "C1", "C2", "C3"):
        rank_total[r][k] += rank_daily[r][k]
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


# =========================
# 出力：分析結果（順番統一）
# =========================
with tabs[2]:
    st.subheader("2着順位分布（全体累積）｜1着が評価1〜3位のとき")
    df_count, df_pct = build_runnerup_tables(pair_counts_total, MAX_FIELD)
    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df_count, use_container_width=True, hide_index=True)
    st.markdown("### 割合%（1→1 / 2→2 / 3→3 は空欄）")
    st.dataframe(df_pct, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("3連複 分布（全体累積）｜1着=wr のとき『相手2枠（順番なし）』")
    for wr in WINNER_RANKS:
        st.markdown(f"### 1着=評価{wr}位 のとき")
        st.markdown("**回数**（上三角だけ使用：a<b）")
        st.dataframe(trifuku_matrix_counts(trifuku_counts_total, wr, MAX_FIELD), use_container_width=True)
        st.markdown("**割合%**（wr内での構成比）")
        st.dataframe(trifuku_matrix_pct(trifuku_counts_total, wr, MAX_FIELD), use_container_width=True)
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

    rec7 = rank_total.get(7, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
    N, C1, C2, C3 = rec7["N"], rec7["C1"], rec7["C2"], rec7["C3"]
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

    st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
