# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜1→2順位分布 ＋ ランク別入賞 ＋ 回収 v2.0（7車固定・欠車対応）")

# =========================
# 基本設定（7車ベース）
# =========================
FIELD_SIZE = 7  # 基本は7車立て（欠車で日次頭数は5/6/7）
WINNER_RANKS = tuple(range(1, 8))  # 1着条件は評価1〜7位（存在しない順位はN=0のまま）

RANK_SYMBOLS = {
    1: "順流順位１位",
    2: "順流順位２位",
    3: "順流順位３位",
    4: "順流順位４位",
    5: "順流順位５位",
    6: "順流順位６位",
    7: "順流順位７位",
}


def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, f"順流順位{r}位")


PairKey = Tuple[int, int]  # (winner_rank, second_rank)


def parse_rankline(s: str, expected_len: int) -> List[str]:
    """
    V順位をパース（欠車対応）
    - expected_len 桁（5/6/7）
    - 許容文字: 1～7
    - 重複なし
    """
    if not s:
        return []
    s = s.replace("-", "").replace(" ", "").replace("/", "").replace(",", "")
    if not s.isdigit() or len(s) != expected_len:
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


def build_conditional_tables(pair_counts: Dict[PairKey, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    pair_counts[(wr, rr)] = 回数
    出力：回数ピボット、割合%ピボット
    ※ wr==rr のセルは回数/割合ともに None（空欄）
    ※ N は wr==rr を除外した合計
    """
    cols = list(range(1, FIELD_SIZE + 1))  # 1..7
    count_rows = []
    pct_rows = []

    for wr in WINNER_RANKS:
        total = 0
        for rr in cols:
            if rr == wr:
                continue
            total += int(pair_counts.get((wr, rr), 0))

        row_c = {"1着の順流順位": wr, "N": total}
        for rr in cols:
            if rr == wr:
                row_c[str(rr)] = None
            else:
                row_c[str(rr)] = int(pair_counts.get((wr, rr), 0))
        count_rows.append(row_c)

        row_p = {"1着の順流順位": wr, "N": total}
        for rr in cols:
            if rr == wr:
                row_p[str(rr)] = None
            else:
                v = int(pair_counts.get((wr, rr), 0))
                row_p[str(rr)] = round(100.0 * v / total, 1) if total > 0 else 0.0
        pct_rows.append(row_p)

    return pd.DataFrame(count_rows), pd.DataFrame(pct_rows)


def rate(x: int, n: int):
    return round(100.0 * x / n, 1) if n > 0 else None


def comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def points_per_race_2f_145(field_n: int) -> int:
    """
    2車複「1-2345」点数
    評価1位固定、相手は評価2～5位まで
    5/6/7車立てでは常に4点
    """
    partner_n = min(max(field_n - 1, 0), 4)
    return partner_n


def points_per_race_3f_145(field_n: int) -> int:
    """
    3連複「1-2345-2345」点数
    評価1位固定、評価2～5位から2頭選ぶ
    5/6/7車立てでは常に C(4,2)=6 点
    """
    partner_n = min(max(field_n - 1, 0), 4)
    return comb2(partner_n)


def new_payout_rec():
    # N: 対象レース数
    # KSUM: 総点数
    # H: 的中（配当あり）
    # U: 的中（配当未入力）
    # SUM: 払戻合計（100円あたり円）
    return {"N": 0, "KSUM": 0, "H": 0, "U": 0, "SUM": 0}


# =========================
# Tabs
# =========================
tabs = st.tabs(["日次手入力（最大12R）", "前日までの集計（累積）", "分析結果"])

# 日次の入力行
byrace_rows: List[Dict] = []

# 前日まで：ランク別（1～7）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(
    lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0}
)

# 前日まで：1→2（順流順位）
pair12_manual: Dict[PairKey, int] = defaultdict(int)

# 前日まで：2車複 1-2345 回収
agg_payout_2f145_manual = new_payout_rec()

# 前日まで：3連複 1-2345-2345 回収
agg_payout_3f145_manual = new_payout_rec()


# =========================
# A. 日次手入力（欠車対応）
# =========================
with tabs[0]:
    st.subheader("日次手入力（7車ベース・欠車対応・最大12R）")
    st.caption(
        "各Rごとに頭数（5/6/7）を選択してください。"
        "V順位はその頭数ぶんの桁数で入力（例：7車=1432567 / 6車=143256）。着順は～3桁。"
        "配当は100円あたりの払戻金（円）を入力。未入力は0のままでOK（集計上は“配当未入力”としてカウント）。"
    )

    cols_hdr = st.columns([1, 1.1, 2.6, 1.2, 1.2, 1.2])
    cols_hdr[0].markdown("**R**")
    cols_hdr[1].markdown("**頭数**")
    cols_hdr[2].markdown("**V順位（頭数ぶんの桁数）**")
    cols_hdr[3].markdown("**着順(～3桁)**")
    cols_hdr[4].markdown("**2車複配当**")
    cols_hdr[5].markdown("**3連複配当**")

    for i in range(1, 13):
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1.1, 2.6, 1.2, 1.2, 1.2])

        rid = c1.text_input("", key=f"rid_{i}", value=str(i))
        field_n = c2.selectbox("", options=[7, 6, 5], index=0, key=f"field_n_{i}")
        vline = c3.text_input("", key=f"vline_{i}", value="")
        fin = c4.text_input("", key=f"fin_{i}", value="")

        pay_2f = c5.number_input("", key=f"pay2f_{i}", min_value=0, value=0, step=10)
        pay_3f = c6.number_input("", key=f"pay3f_{i}", min_value=0, value=0, step=10)

        vorder = parse_rankline(vline, field_n)
        finish = parse_finish(fin)

        any_input = any([vline.strip(), fin.strip(), pay_2f > 0, pay_3f > 0])
        if any_input:
            if not vorder:
                st.warning(f"R{rid}: 頭数{field_n}なので、V順位は{field_n}桁で入力してください。")
                continue

            vset = set(vorder)
            invalid_finish = [x for x in finish if x not in vset]
            if invalid_finish:
                st.warning(
                    f"R{rid}: 着順 {''.join(invalid_finish)} がV順位（出走車）に含まれていません。"
                    " 欠車/入力ミスの可能性があります。"
                )

            byrace_rows.append(
                {
                    "race": rid,
                    "field_n": field_n,
                    "vorder": vorder,
                    "finish": finish,
                    "pay_2f": int(pay_2f),
                    "pay_3f": int(pay_3f),
                }
            )


# =========================
# B. 前日までの集計（累積）
# =========================
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")

    cols_12 = list(range(1, FIELD_SIZE + 1))  # 1..7

    # ---- 1→2（累積入力）
    st.markdown("## 1→2 着順位分布（累積・回数）")
    st.caption("1着が順流順位1〜7位のとき、2着の順流順位の回数を入力。1→1 / 2→2 / ... / 7→7 は空欄（入力不可）。")

    h = st.columns([1.8] + [1] * len(cols_12))
    h[0].markdown("**条件：1着の順流順位**")
    for j, rr in enumerate(cols_12, start=1):
        h[j].markdown(f"**2着={rr}**")

    for wr in WINNER_RANKS:
        row_cols = st.columns([1.8] + [1] * len(cols_12))
        row_cols[0].write(f"順流順位{wr}位が1着")
        for j, rr in enumerate(cols_12, start=1):
            if rr == wr:
                row_cols[j].write("")
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
    st.caption("順流順位1～7位まで入力。Nは各順位が存在したレース数（欠車があると下位順位のNは減ります）。")

    hdr = st.columns([1.8, 1, 1, 1.8])
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

    for r in range(1, 8):
        c0, c1, c2, c3 = st.columns([1.8, 1, 1, 1.8])
        c0.write(rank_symbol(r))
        N = c1.number_input("", key=f"aggN_{r}", min_value=0, value=0)
        C1 = c2.number_input("", key=f"aggC1_{r}", min_value=0, value=0)
        c3_cols = c3.columns(2)
        C2 = c3_cols[0].number_input("", key=f"aggC2_{r}", min_value=0, value=0)
        C3 = c3_cols[1].number_input("", key=f"aggC3_{r}", min_value=0, value=0)

        if any([N, C1, C2, C3]):
            add_rank_rec(r, N, C1, C2, C3)

    st.divider()

    # ---- 2車複 1-2345 回収（累積入力）
    st.markdown("## 2車複 回収（累積）｜評価1位－評価2〜5位")
    st.caption(
        "2車複は全流しではなく、固定で『1-2345』のみ集計します。"
        "前日までの累積を入力してください。"
    )

    h2 = st.columns([2.2, 1, 1, 1, 1, 1.2])
    h2[0].markdown("**買い目形**")
    h2[1].markdown("**対象N**")
    h2[2].markdown("**KSUM**")
    h2[3].markdown("**SUM**")
    h2[4].markdown("**H**")
    h2[5].markdown("**U**")

    c0, c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1, 1.2])
    c0.write("1-2345")

    N = c1.number_input("", key="pN_2f145", min_value=0, value=0)
    KSUM = c2.number_input("", key="pKSUM_2f145", min_value=0, value=0)
    SUM = c3.number_input("", key="pSUM_2f145", min_value=0, value=0, step=10)
    H = c4.number_input("", key="pH_2f145", min_value=0, value=0)
    U = c5.number_input("", key="pU_2f145", min_value=0, value=0)

    if any([N, KSUM, SUM, H, U]):
        agg_payout_2f145_manual["N"] += int(N)
        agg_payout_2f145_manual["KSUM"] += int(KSUM)
        agg_payout_2f145_manual["SUM"] += int(SUM)
        agg_payout_2f145_manual["H"] += int(H)
        agg_payout_2f145_manual["U"] += int(U)

    st.divider()

    # ---- 3連複 1-2345-2345 回収（累積入力）
    st.markdown("## 3連複 回収（累積）｜評価1位－評価2〜5位－評価2〜5位")
    st.caption(
        "3連複は全流しではなく、固定で『1-2345-2345』のみ集計します。"
        "前日までの累積を入力してください。"
    )

    h3 = st.columns([2.2, 1, 1, 1, 1, 1.2])
    h3[0].markdown("**買い目形**")
    h3[1].markdown("**対象N**")
    h3[2].markdown("**KSUM**")
    h3[3].markdown("**SUM**")
    h3[4].markdown("**H**")
    h3[5].markdown("**U**")

    c0, c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1, 1.2])
    c0.write("1-2345-2345")

    N = c1.number_input("", key="pN_3f145", min_value=0, value=0)
    KSUM = c2.number_input("", key="pKSUM_3f145", min_value=0, value=0)
    SUM = c3.number_input("", key="pSUM_3f145", min_value=0, value=0, step=10)
    H = c4.number_input("", key="pH_3f145", min_value=0, value=0)
    U = c5.number_input("", key="pU_3f145", min_value=0, value=0)

    if any([N, KSUM, SUM, H, U]):
        agg_payout_3f145_manual["N"] += int(N)
        agg_payout_3f145_manual["KSUM"] += int(KSUM)
        agg_payout_3f145_manual["SUM"] += int(SUM)
        agg_payout_3f145_manual["H"] += int(H)
        agg_payout_3f145_manual["U"] += int(U)


# =========================
# 集計：日次 + 前日まで累積 を合算
# =========================

# --- ランク別（日次） ---
rank_daily: Dict[int, Dict[str, int]] = {
    r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, 8)
}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if not vorder:
        continue

    car_by_rank = {i + 1: vorder[i] for i in range(len(vorder))}

    for r in range(1, len(vorder) + 1):
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
    r: {"N": 0, "C1": 0, "C2": 0, "C3": 0} for r in range(1, 8)
}

for r in range(1, 8):
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

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    win_car = finish[0]
    sec_car = finish[1]

    win_rank = car_to_rank.get(win_car)
    sec_rank = car_to_rank.get(sec_car)

    if win_rank is None or sec_rank is None:
        continue

    pair12_daily[(win_rank, sec_rank)] += 1

# --- 1→2（合算） ---
pair12_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair12_daily.items():
    pair12_total[k] += int(v)
for k, v in pair12_manual.items():
    pair12_total[k] += int(v)

# --- 2車複「1-2345」回収（日次） ---
payout_2f145_daily = new_payout_rec()

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    field_n = int(row.get("field_n", len(vorder) or 0))
    if not vorder or field_n <= 0 or len(finish) < 2:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    finish_ranks = [car_to_rank.get(car) for car in finish[:2]]

    if any(r is None for r in finish_ranks):
        continue

    payout_2f145_daily["N"] += 1
    payout_2f145_daily["KSUM"] += points_per_race_2f_145(field_n)

    finish_rank_set = set(int(r) for r in finish_ranks)
    hit = (1 in finish_rank_set) and all(r in {1, 2, 3, 4, 5} for r in finish_rank_set)

    if hit:
        pay = int(row.get("pay_2f", 0))
        if pay > 0:
            payout_2f145_daily["H"] += 1
            payout_2f145_daily["SUM"] += pay
        else:
            payout_2f145_daily["U"] += 1

# --- 3連複「1-2345-2345」回収（日次） ---
payout_3f145_daily = new_payout_rec()

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    field_n = int(row.get("field_n", len(vorder) or 0))
    if not vorder or field_n <= 0 or len(finish) < 3:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    finish_ranks = [car_to_rank.get(car) for car in finish[:3]]

    if any(r is None for r in finish_ranks):
        continue

    payout_3f145_daily["N"] += 1
    payout_3f145_daily["KSUM"] += points_per_race_3f_145(field_n)

    finish_rank_set = set(int(r) for r in finish_ranks)
    hit = (
        1 in finish_rank_set
        and len(finish_rank_set) == 3
        and all(r in {1, 2, 3, 4, 5} for r in finish_rank_set)
    )

    if hit:
        pay = int(row.get("pay_3f", 0))
        if pay > 0:
            payout_3f145_daily["H"] += 1
            payout_3f145_daily["SUM"] += pay
        else:
            payout_3f145_daily["U"] += 1

# --- 2車複「1-2345」回収（合算：日次 + 前日まで） ---
payout_2f145_total = new_payout_rec()
for k in ("N", "KSUM", "H", "U", "SUM"):
    payout_2f145_total[k] = payout_2f145_daily[k] + agg_payout_2f145_manual[k]

# --- 3連複「1-2345-2345」回収（合算：日次 + 前日まで） ---
payout_3f145_total = new_payout_rec()
for k in ("N", "KSUM", "H", "U", "SUM"):
    payout_3f145_total[k] = payout_3f145_daily[k] + agg_payout_3f145_manual[k]


# =========================
# 出力：分析結果
# =========================
with tabs[2]:
    st.subheader("1→2 着順位分布（全体累積）｜1着が順流順位1〜7位のとき（欠車対応）")
    st.caption("欠車レースでは存在しない下位順位（例：6車レースの評価7位）はNに含まれません。")

    df12_count, df12_pct = build_conditional_tables(pair12_total)

    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df12_count, use_container_width=True, hide_index=True)

    st.markdown("### 割合%（同順位セルは空欄）")
    st.dataframe(df12_pct, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("ランク別 入賞テーブル（全体累積）｜欠車対応")
    rows_out = []
    for r in range(1, 8):
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
    st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("2車複 回収期待値%｜評価1位－評価2〜5位")
    st.caption("回収期待値% = 払戻合計SUM ÷ 総点数KSUM。75%以上を候補として残す。")

    THRESH = 75.0

    rec = payout_2f145_total
    N = rec["N"]
    KSUM = rec["KSUM"]
    H = rec["H"]
    U = rec["U"]
    SUM = rec["SUM"]

    roi = round(SUM / KSUM, 1) if KSUM > 0 else None
    avg_pay = round(SUM / H, 1) if H > 0 else None
    hit_rate = round(100.0 * H / N, 1) if N > 0 else None

    df_2f = pd.DataFrame(
        [
            {
                "買い目形": "1-2345",
                "対象N": N,
                "総点数KSUM": KSUM,
                "払戻合計SUM": SUM,
                "的中H（配当あり）": H,
                "的中U（配当未入力）": U,
                "的中率%（配当あり分）": hit_rate,
                "平均配当（配当あり分）": avg_pay,
                "回収期待値%": roi,
                "判定": "◎" if (roi is not None and roi >= THRESH) else "",
            }
        ]
    )
    st.dataframe(df_2f, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("3連複 回収期待値%｜評価1位－評価2〜5位－評価2〜5位")

    rec = payout_3f145_total
    N = rec["N"]
    KSUM = rec["KSUM"]
    H = rec["H"]
    U = rec["U"]
    SUM = rec["SUM"]

    roi = round(SUM / KSUM, 1) if KSUM > 0 else None
    avg_pay = round(SUM / H, 1) if H > 0 else None
    hit_rate = round(100.0 * H / N, 1) if N > 0 else None

    df_3f = pd.DataFrame(
        [
            {
                "買い目形": "1-2345-2345",
                "対象N": N,
                "総点数KSUM": KSUM,
                "払戻合計SUM": SUM,
                "的中H（配当あり）": H,
                "的中U（配当未入力）": U,
                "的中率%（配当あり分）": hit_rate,
                "平均配当（配当あり分）": avg_pay,
                "回収期待値%": roi,
                "判定": "◎" if (roi is not None and roi >= THRESH) else "",
            }
        ]
    )
    st.dataframe(df_3f, use_container_width=True, hide_index=True)
