# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜2車単→2車複→ワイド→ランク別 v1.7（7車固定）")

# -------- 基本設定 --------
MAX_FIELD = 7              # ★ 7車固定
FOCUS_RANKS = (1, 2, 3)    # 復習対象（評価1〜3位）

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
# 汎用：相手順位分布テーブル（回数・割合）
# counts[(t, p)] = tが絡むときの相手p回数（順序なし・対称カウント）
# =========================
TPKey = Tuple[int, int]  # (target_rank, partner_rank)

def partner_tables(counts: Dict[TPKey, int], title_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(range(1, MAX_FIELD + 1))
    rows_c, rows_p = [], []

    for t in FOCUS_RANKS:
        total = sum(int(counts.get((t, p), 0)) for p in cols if p != t)

        row_c = {title_col: t, "N": total}
        row_p = {title_col: t, "N": total}

        for p in cols:
            if p == t:
                row_c[str(p)] = None
                row_p[str(p)] = None
            else:
                v = int(counts.get((t, p), 0))
                row_c[str(p)] = v
                row_p[str(p)] = round(100.0 * v / total, 1) if total > 0 else 0.0

        rows_c.append(row_c)
        rows_p.append(row_p)

    return pd.DataFrame(rows_c), pd.DataFrame(rows_p)


# =========================
# Tabs
# =========================
tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

byrace_rows: List[Dict] = []

# 前日まで：ランク別（全体）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0})

# 前日まで：2車単（1着→2着）分布
exacta_manual: Dict[TPKey, int] = defaultdict(int)   # (1着評価, 2着評価)

# 前日まで：2車複（連対したときの相手）分布
quinella_manual: Dict[TPKey, int] = defaultdict(int) # (対象評価, 相手評価)

# 前日まで：ワイド（3着内のときの相手）分布
wide_manual: Dict[TPKey, int] = defaultdict(int)     # (対象評価, 相手評価)


# -------- 日次手入力 --------
with tabs[0]:
    st.subheader("日次手入力（全体・最大12R）")

    cols_hdr = st.columns([1, 2, 2])
    cols_hdr[0].markdown("**R**")
    cols_hdr[1].markdown("**V順位(例: 1432765)**")
    cols_hdr[2].markdown("**着順(～3桁)**")

    for i in range(1, 13):
        c1, c2, c3 = st.columns([1, 2, 2])
        rid = c1.text_input("", key=f"rid_{i}", value=str(i))
        vline = c2.text_input("", key=f"vline_{i}", value="")
        fin = c3.text_input("", key=f"fin_{i}", value="")

        vorder = parse_rankline(vline)
        finish = parse_finish(fin)

        any_input = any([vorder, finish])
        if any_input:
            if vorder:
                byrace_rows.append({
                    "race": rid,
                    "vorder": vorder,
                    "finish": finish,
                })
            else:
                st.warning(f"R{rid}: 入力不整合（V順位を確認）。")


# -------- 前日まで累積入力（順番統一：2車単→2車複→ワイド→ランク別） --------
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")

    cols = list(range(1, MAX_FIELD + 1))

    # 1) 2車単（1着→2着）
    st.markdown("## 2車単（復習）｜1着が評価1〜3位のとき、2着評価分布（回数）")
    h = st.columns([1] + [1] * len(cols))
    h[0].markdown("**1着評価\\2着評価**")
    for j, p in enumerate(cols, start=1):
        h[j].markdown(f"**{p}**")

    for w in FOCUS_RANKS:
        row = st.columns([1] + [1] * len(cols))
        row[0].write(f"{w}")
        for j, p in enumerate(cols, start=1):
            if p == w:
                row[j].write("-")
                continue
            v = row[j].number_input("", key=f"exa_prev_w{w}_p{p}", min_value=0, value=0)
            if v:
                exacta_manual[(w, p)] += int(v)

    st.divider()

    # 2) 2車複（対象が連対した時の相手）
    st.markdown("## 2車複（復習）｜評価1〜3位が連対（1-2着）したとき、相手評価分布（回数）")
    h = st.columns([1] + [1] * len(cols))
    h[0].markdown("**対象評価\\相手評価**")
    for j, p in enumerate(cols, start=1):
        h[j].markdown(f"**{p}**")

    for t in FOCUS_RANKS:
        row = st.columns([1] + [1] * len(cols))
        row[0].write(f"{t}")
        for j, p in enumerate(cols, start=1):
            if p == t:
                row[j].write("-")
                continue
            v = row[j].number_input("", key=f"qui_prev_t{t}_p{p}", min_value=0, value=0)
            if v:
                quinella_manual[(t, p)] += int(v)

    st.divider()

    # 3) ワイド（対象が3着内の時の相手）
    st.markdown("## ワイド（復習）｜評価1〜3位が3着内のとき、相手評価分布（回数）")
    st.caption("※1レースで対象評価が3着内にいると、相手は最大2つ（他の2車）カウントされます。")

    h = st.columns([1] + [1] * len(cols))
    h[0].markdown("**対象評価\\相手評価**")
    for j, p in enumerate(cols, start=1):
        h[j].markdown(f"**{p}**")

    for t in FOCUS_RANKS:
        row = st.columns([1] + [1] * len(cols))
        row[0].write(f"{t}")
        for j, p in enumerate(cols, start=1):
            if p == t:
                row[j].write("-")
                continue
            v = row[j].number_input("", key=f"wid_prev_t{t}_p{p}", min_value=0, value=0)
            if v:
                wide_manual[(t, p)] += int(v)

    st.divider()

    # 4) ランク別入賞回数（累積）
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
# 日次から分布を作る
# =========================
exacta_daily: Dict[TPKey, int] = defaultdict(int)
quinella_daily: Dict[TPKey, int] = defaultdict(int)
wide_daily: Dict[TPKey, int] = defaultdict(int)

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

    # 2車単（1着→2着）
    if len(finish) >= 2:
        w = car_to_rank.get(finish[0])
        p = car_to_rank.get(finish[1])
        if w in FOCUS_RANKS and p is not None and p != w:
            exacta_daily[(w, p)] += 1

    # 2車複（連対：1-2着の組）…対象が1〜3位なら相手分布を両向きで加算
    if len(finish) >= 2:
        r1 = car_to_rank.get(finish[0])
        r2 = car_to_rank.get(finish[1])
        if r1 is not None and r2 is not None and r1 != r2:
            if r1 in FOCUS_RANKS:
                quinella_daily[(r1, r2)] += 1
            if r2 in FOCUS_RANKS:
                quinella_daily[(r2, r1)] += 1

    # ワイド（3着内：上位3車のペア）…対象が1〜3位なら相手（他2車）を加算
    if len(finish) >= 3:
        r1 = car_to_rank.get(finish[0])
        r2 = car_to_rank.get(finish[1])
        r3 = car_to_rank.get(finish[2])
        if None not in (r1, r2, r3) and len({r1, r2, r3}) == 3:
            top3 = [r1, r2, r3]
            for t in FOCUS_RANKS:
                if t in top3:
                    others = [x for x in top3 if x != t]  # 2つ
                    for p in others:
                        wide_daily[(t, p)] += 1


# =========================
# 合算（前日まで + 日次）
# =========================
def merge_counts(a: Dict[TPKey, int], b: Dict[TPKey, int]) -> Dict[TPKey, int]:
    out = defaultdict(int)
    for k, v in a.items():
        out[k] += int(v)
    for k, v in b.items():
        out[k] += int(v)
    return out

exacta_total = merge_counts(exacta_daily, exacta_manual)
quinella_total = merge_counts(quinella_daily, quinella_manual)
wide_total = merge_counts(wide_daily, wide_manual)

# ランク別合算
rank_total = {r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, MAX_FIELD + 1)}
for r in range(1, MAX_FIELD + 1):
    for k in ("N", "C1", "C2", "C3"):
        rank_total[r][k] += rank_daily[r][k]
for r, rec in agg_rank_manual.items():
    rr = 7 if r == 7 else r
    for k in ("N", "C1", "C2", "C3"):
        rank_total[rr][k] += int(rec[k])


# =========================
# 出力：分析結果（順番統一）
# =========================
with tabs[2]:
    st.subheader("2車単（復習）｜1着が評価1〜3位のとき、2着評価分布")
    df_c, df_p = partner_tables(exacta_total, "1着の評価順位")
    st.markdown("### 回数（N=条件付き総数）")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    st.markdown("### 割合%（同じ順位列は空欄）")
    st.dataframe(df_p, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("2車複（復習）｜評価1〜3位が連対（1-2着）したとき、相手評価分布")
    df_c, df_p = partner_tables(quinella_total, "対象評価")
    st.markdown("### 回数（N=対象評価が連対した総数）")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    st.markdown("### 割合%（同じ順位列は空欄）")
    st.dataframe(df_p, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("ワイド（復習）｜評価1〜3位が3着内のとき、相手評価分布")
    df_c, df_p = partner_tables(wide_total, "対象評価")
    st.markdown("### 回数（N=対象評価が3着内に来た時の“相手カウント総数”）")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    st.markdown("### 割合%（同じ順位列は空欄）")
    st.dataframe(df_p, use_container_width=True, hide_index=True)

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
