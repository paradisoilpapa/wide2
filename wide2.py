# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ分析（開催区分別）", layout="wide")
st.title("ヴェロビ 組み方分析（可変頭数・開催区分別／全体集計） v2.7+復習分布累積")

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


# ===== 復習分布（累積用）設定 =====
WINNER_RANKS = (1, 2, 3)      # 1着が評価1～3位の条件だけ見る
RUNNERUP_MAX = 6             # 2着は評価1～6位を個別
RUNNERUP_BIN = 7             # 7位以上はまとめて「7+」


def build_runnerup_dist_from_counts(pair_counts: Dict[Tuple[int, int], int]) -> pd.DataFrame:
    """
    pair_counts[(winner_rank, runnerup_rank_bin)] = 回数
    runnerup_rank_bin: 1..6, 7(=7+)
    """
    rows = []
    for wr in WINNER_RANKS:
        N = 0
        for rr in list(range(1, RUNNERUP_MAX + 1)) + [RUNNERUP_BIN]:
            N += int(pair_counts.get((wr, rr), 0))

        if N == 0:
            rows.append({
                "条件(1着の評価順位)": wr,
                "対象レース数N": 0,
                "2着の評価順位": "-",
                "回数": 0,
                "割合%": 0.0,
            })
            continue

        for rr in list(range(1, RUNNERUP_MAX + 1)) + [RUNNERUP_BIN]:
            c = int(pair_counts.get((wr, rr), 0))
            if c == 0:
                continue
            label = rr if rr != RUNNERUP_BIN else "7+"
            rows.append({
                "条件(1着の評価順位)": wr,
                "対象レース数N": N,
                "2着の評価順位": label,
                "回数": c,
                "割合%": round(100.0 * c / N, 1),
            })
    return pd.DataFrame(rows)


# -------- 入力タブ --------
input_tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

byrace_rows: List[Dict] = []

# 既存：ランク別回数の累積（開催区分別）
agg_counts_manual: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0})

# ★追加：復習分布の累積（開催区分なし・全体のみ）
# pair_counts_manual[(winner_rank, runnerup_rank_bin)] = 回数
pair_counts_manual: Dict[Tuple[int, int], int] = defaultdict(int)


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
            # V順位は頭数以下の桁であること（例外: 未入力は許容）
            if vorder and (4 <= field <= MAX_FIELD) and len(vorder) <= field:
                byrace_rows.append({
                    "day": day_global,   # 内部キー（"L","F2","F1","G"）
                    "race": rid,
                    "field": field,
                    "vorder": vorder,
                    "finish": finish,
                })
            else:
                st.warning(f"R{rid}: 入力不整合（V順位/頭数を確認）。V順位は頭数桁以下、4～{MAX_FIELD}車のみ対象。")


# B. 前日までの集計（手入力）
with input_tabs[1]:
    st.subheader("前日までの集計（累積）")

    # ---- 既存：開催区分×ランク別（1着/2着/3着） ----
    st.markdown("## ① ランク別 入賞回数（開催区分別・既存）")

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

        # 1～6を個別入力
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

        # 無（7位以降）
        c0, c1, c2, c3 = st.columns([1,1,1,1])
        c0.write("carFR順位７～位")
        N_mu  = c1.number_input("", key=f"agg_{day}_N_{MU_BIN_R}",  min_value=0, value=0)
        C1_mu = c2.number_input("", key=f"agg_{day}_C1_{MU_BIN_R}", min_value=0, value=0)
        c3_cols = c3.columns(2)
        C2_mu = c3_cols[0].number_input("", key=f"agg_{day}_C2_{MU_BIN_R}", min_value=0, value=0)
        C3_mu = c3_cols[1].number_input("", key=f"agg_{day}_C3_{MU_BIN_R}", min_value=0, value=0)
        if any([N_mu, C1_mu, C2_mu, C3_mu]):
            add_rec(day, MU_BIN_R, N_mu, C1_mu, C2_mu, C3_mu)

    st.divider()

    # ---- ★追加：復習用（1着評価X→2着評価Yの回数）累積 ----
    st.markdown("## ② 復習用：『1着が評価1〜3位』のとき、2着の評価順位（累積）")
    st.caption("ここは開催区分なし（全体累積）です。2着は 1〜6位 + 7位以上(7+) にまとめます。")

    # ヘッダ
    hdr = st.columns([1] + [1]*RUNNERUP_MAX + [1])
    hdr[0].markdown("**条件：1着の評価順位**")
    for rr in range(1, RUNNERUP_MAX + 1):
        hdr[rr].markdown(f"**2着=評価{rr}位**")
    hdr[-1].markdown("**2着=7+**")

    for wr in WINNER_RANKS:
        cols = st.columns([1] + [1]*RUNNERUP_MAX + [1])
        cols[0].write(f"評価{wr}位が1着")
        for rr in range(1, RUNNERUP_MAX + 1):
            v = cols[rr].number_input(
                "",
                key=f"pair_wr{wr}_rr{rr}",
                min_value=0,
                value=0
            )
            if v:
                pair_counts_manual[(wr, rr)] += int(v)

        v7 = cols[-1].number_input(
            "",
            key=f"pair_wr{wr}_rr7p",
            min_value=0,
            value=0
        )
        if v7:
            pair_counts_manual[(wr, RUNNERUP_BIN)] += int(v7)


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

# 手入力の前日まで集計を合算
for (day, r), rec in agg_counts_manual.items():
    rank_counts_daily[(day, r)]["N"]  += rec["N"]
    rank_counts_daily[(day, r)]["C1"] += rec["C1"]
    rank_counts_daily[(day, r)]["C2"] += rec["C2"]
    rank_counts_daily[(day, r)]["C3"] += rec["C3"]

# 全体集計の構築
rank_counts_total: Dict[int, Dict[str, int]] = {r: {"N":0, "C1":0, "C2":0, "C3":0} for r in range(1, MAX_FIELD + 1)}
for (day, r), rec in rank_counts_daily.items():
    for k in ("N","C1","C2","C3"):
        rank_counts_total[r][k] += rec[k]

# ===== ★追加：復習分布（レース入力ぶん）を作る =====
pair_counts_daily: Dict[Tuple[int, int], int] = defaultdict(int)

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])

    if len(finish) < 2:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    win_car = finish[0]
    run_car = finish[1]

    win_rank = car_to_rank.get(win_car)
    if win_rank not in WINNER_RANKS:
        continue

    r2 = car_to_rank.get(run_car)
    if r2 is None or r2 >= RUNNERUP_BIN:
        rr_bin = RUNNERUP_BIN
    else:
        rr_bin = r2

    pair_counts_daily[(win_rank, rr_bin)] += 1

# 日次 + 前日累積 を合算
pair_counts_total: Dict[Tuple[int, int], int] = defaultdict(int)
for k, v in pair_counts_daily.items():
    pair_counts_total[k] += int(v)
for k, v in pair_counts_manual.items():
    pair_counts_total[k] += int(v)


# -------- 出力タブ --------
with input_tabs[2]:
    # ★復習分布（累積）
    st.subheader("復習：『1着が評価1〜3位』のとき、2着は評価何位が多いか（累積分布）")
    st.dataframe(build_runnerup_dist_from_counts(pair_counts_total), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("開催区分別：ランク別 入賞テーブル（◎〜carFR順位７～位）")
    for day in DAY_OPTIONS:
        rows_out = []
        # 1～6位は個別
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

        # 7位以降を「無」にまとめる
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

    # 全体集計も同じ処理
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
