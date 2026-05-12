# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜1→2評価分布 ＋ 評価別入賞 ＋ 新回収率 v3.6（7車固定・欠車対応）")

# =========================
# 基本設定（7車ベース）
# =========================
FIELD_SIZE = 7
WINNER_RANKS = tuple(range(1, 8))
PATTERN_AXES = (1, 2)
AXIS1_TARGETS = (2, 3, 4, 5)
AXIS2_TARGETS = (1, 3)

RANK_SYMBOLS = {
    1: "評価１",
    2: "評価２",
    3: "評価３",
    4: "評価４",
    5: "評価５",
    6: "評価６",
    7: "評価７",
}


def rank_symbol(r: int) -> str:
    return RANK_SYMBOLS.get(r, f"評価{r}")


PairKey = Tuple[int, int]  # (winner_eval, second_eval)


def parse_rankline(s: str, expected_len: int) -> List[str]:
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
    cols = list(range(1, FIELD_SIZE + 1))
    count_rows = []
    pct_rows = []

    for wr in WINNER_RANKS:
        total = 0
        for rr in cols:
            if rr == wr:
                continue
            total += int(pair_counts.get((wr, rr), 0))

        row_c = {"1着の評価": wr, "N": total}
        for rr in cols:
            row_c[str(rr)] = None if rr == wr else int(pair_counts.get((wr, rr), 0))
        count_rows.append(row_c)

        row_p = {"1着の評価": wr, "N": total}
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


def new_payout_rec() -> Dict[str, int]:
    return {"N": 0, "KSUM": 0, "H": 0, "U": 0, "SUM": 0}


def add_rec(dst: Dict[str, int], src: Dict[str, int]):
    for k in ("N", "KSUM", "H", "U", "SUM"):
        dst[k] += int(src.get(k, 0))


def combine_recs(recs: List[Dict[str, int]]) -> Dict[str, int]:
    """
    同じレース群に対する複数買い目の合算。
    Nは足さず、最大Nを使う。
    KSUM/H/U/SUMは合算する。
    """
    out = new_payout_rec()
    out["N"] = max((int(r.get("N", 0)) for r in recs), default=0)
    for rec in recs:
        out["KSUM"] += int(rec.get("KSUM", 0))
        out["H"] += int(rec.get("H", 0))
        out["U"] += int(rec.get("U", 0))
        out["SUM"] += int(rec.get("SUM", 0))
    return out


def targets_for_pattern(axis: int, field_n: int) -> List[int]:
    """
    2車単固定型の相手評価。
    評価1→2345
    評価2→13
    欠車対応：存在する評価だけ返す。
    """
    if axis == 1:
        base = AXIS1_TARGETS
    elif axis == 2:
        base = AXIS2_TARGETS
    else:
        base = ()

    return [r for r in base if r <= field_n]


def ksum_2t_pattern(axis: int, field_n: int) -> int:
    """2車単固定型の点数。"""
    if axis > field_n:
        return 0
    return len(targets_for_pattern(axis, field_n))


def hit_2t_pattern(axis: int, win_rank: int, sec_rank: int, field_n: int) -> bool:
    """2車単固定型の的中判定。"""
    if axis > field_n:
        return False
    return win_rank == axis and sec_rank in targets_for_pattern(axis, field_n)


def pattern_label(axis: int) -> str:
    if axis == 1:
        return "2車単 1→2345"
    if axis == 2:
        return "2車単 2→13"
    return f"2車単 評価{axis}"


def pair_target_label(axis: int, target: int) -> str:
    return f"2車単 {axis}→{target}"


def ksum_axis_to_target(axis: int, target: int, field_n: int) -> int:
    """2車単：axis→target の点数。存在する評価だけ1点として扱う。"""
    if field_n < 2:
        return 0
    if axis > field_n or target > field_n:
        return 0
    if axis == target:
        return 0
    return 1


def hit_axis_to_target(axis: int, target: int, win_rank: int, sec_rank: int, field_n: int) -> bool:
    """2車単：axis→target の的中判定。"""
    if ksum_axis_to_target(axis, target, field_n) <= 0:
        return False
    return win_rank == axis and sec_rank == target


def payout_row(label: str, rec: Dict[str, int]) -> Dict:
    N = int(rec["N"])
    KSUM = int(rec["KSUM"])
    H = int(rec["H"])
    U = int(rec["U"])
    SUM = int(rec["SUM"])

    invest = KSUM * 100
    roi = round(100.0 * SUM / invest, 1) if invest > 0 else None
    avg_pay = round(SUM / H, 1) if H > 0 else None
    hit_rate = round(100.0 * H / N, 1) if N > 0 else None

    return {
        "型": label,
        "対象N": N,
        "総点数KSUM": KSUM,
        "投資額換算": invest,
        "払戻合計SUM": SUM,
        "的中H（配当あり）": H,
        "的中U（配当未入力）": U,
        "的中率%": hit_rate,
        "平均配当": avg_pay,
        "回収率%": roi,
        "判定": "◎" if (roi is not None and roi >= 100.0) else "",
    }


# =========================
# Tabs
# =========================
tabs = st.tabs(["日次手入力（最大36R）", "前日までの集計（累積）", "分析結果"])

# 日次の入力行
byrace_rows: List[Dict] = []

# 前日まで：評価別（1～7）
agg_rank_manual: Dict[int, Dict[str, int]] = defaultdict(
    lambda: {"N": 0, "C1": 0, "C2": 0, "C3": 0}
)

# 前日まで：1→2（評価）
pair12_manual: Dict[PairKey, int] = defaultdict(int)

# 前日まで：新回収率
# 2車単：1→2345 / 2→13
agg_payout_2t_pattern_manual: Dict[int, Dict[str, int]] = {
    axis: new_payout_rec() for axis in PATTERN_AXES
}

# 前日まで：個別回収（任意入力）
# 1→2 / 1→3 / 1→4 / 1→5 / 2→1 / 2→3
agg_payout_axis_target_manual: Dict[Tuple[int, int], Dict[str, int]] = {
    (1, target): new_payout_rec() for target in AXIS1_TARGETS
}
agg_payout_axis_target_manual.update({
    (2, target): new_payout_rec() for target in AXIS2_TARGETS
})


# =========================
# A. 日次手入力（欠車対応）
# =========================
with tabs[0]:
    st.subheader("日次手入力（7車ベース・欠車対応・最大60R）")
    st.caption(
        "入力中の白化を抑えるため、フォーム送信式です。"
        "V評価は頭数ぶんの桁数で入力（例：7車=1432567 / 6車=143256）。"
        "着順は～3桁。2車単配当は100円あたりの払戻金（円）を入力。"
    )

    with st.form("daily_input_form"):
        cols_hdr = st.columns([1, 1.1, 2.6, 1.2, 1.2])
        cols_hdr[0].markdown("**R**")
        cols_hdr[1].markdown("**頭数**")
        cols_hdr[2].markdown("**V評価（頭数ぶんの桁数）**")
        cols_hdr[3].markdown("**着順(～3桁)**")
        cols_hdr[4].markdown("**2車単配当**")

        daily_inputs = []

        for i in range(1, 61):
            c1, c2, c3, c4, c5 = st.columns([1, 1.1, 2.6, 1.2, 1.2])

            rid = c1.text_input("", key=f"rid_{i}", value=str(i))
            field_n = c2.selectbox("", options=[7, 6, 5], index=0, key=f"field_n_{i}")
            vline = c3.text_input("", key=f"vline_{i}", value="")
            fin = c4.text_input("", key=f"fin_{i}", value="")
            pay_2t = c5.number_input("", key=f"pay2t_{i}", min_value=0, value=0, step=10)

            daily_inputs.append(
                {
                    "rid": rid,
                    "field_n": field_n,
                    "vline": vline,
                    "fin": fin,
                    "pay_2t": pay_2t,
                }
            )

        st.form_submit_button("日次入力を反映")

    for item in daily_inputs:
        rid = item["rid"]
        field_n = int(item["field_n"])
        vline = item["vline"]
        fin = item["fin"]
        pay_2t = int(item["pay_2t"])

        vorder = parse_rankline(vline, field_n)
        finish = parse_finish(fin)

        any_input = any([vline.strip(), fin.strip(), pay_2t > 0])
        if any_input:
            if not vorder:
                st.warning(f"R{rid}: 頭数{field_n}なので、V評価は{field_n}桁で入力してください。")
                continue

            vset = set(vorder)
            invalid_finish = [x for x in finish if x not in vset]
            if invalid_finish:
                st.warning(
                    f"R{rid}: 着順 {''.join(invalid_finish)} がV評価（出走車）に含まれていません。"
                    " 欠車/入力ミスの可能性があります。"
                )

            byrace_rows.append(
                {
                    "race": rid,
                    "field_n": field_n,
                    "vorder": vorder,
                    "finish": finish,
                    "pay_2t": pay_2t,
                }
            )


# =========================
# B. 前日までの集計（累積）
# =========================
with tabs[1]:
    st.subheader("前日までの集計（累積・全体）")
    st.caption("入力中の白化を抑えるため、フォーム送信式です。入力後に下のボタンを押してください。")

    with st.form("prev_aggregate_form"):
        cols_12 = list(range(1, FIELD_SIZE + 1))

        st.markdown("## 1→2 着評価分布（累積・回数）")
        st.caption("1着が評価1〜7のとき、2着の評価の回数を入力。")

        h = st.columns([1.8] + [1] * len(cols_12))
        h[0].markdown("**条件：1着の評価**")
        for j, rr in enumerate(cols_12, start=1):
            h[j].markdown(f"**2着={rr}**")

        pair_inputs = []
        for wr in WINNER_RANKS:
            row_cols = st.columns([1.8] + [1] * len(cols_12))
            row_cols[0].write(f"評価{wr}が1着")
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
                pair_inputs.append((wr, rr, int(v)))

        st.divider()

        st.markdown("## 評価別 入賞回数（累積）")
        st.caption("評価1～7まで入力。Nは各評価が存在したレース数。")

        hdr = st.columns([1.8, 1, 1, 1.8])
        hdr[0].markdown("**評価**")
        hdr[1].markdown("**出走数N**")
        hdr[2].markdown("**1着回数**")
        hdr[3].markdown("**2着回数 / 3着回数**")

        rank_inputs = []
        for r in range(1, 8):
            c0, c1, c2, c3 = st.columns([1.8, 1, 1, 1.8])
            c0.write(rank_symbol(r))
            N = c1.number_input("", key=f"aggN_{r}", min_value=0, value=0)
            C1 = c2.number_input("", key=f"aggC1_{r}", min_value=0, value=0)
            c3_cols = c3.columns(2)
            C2 = c3_cols[0].number_input("", key=f"aggC2_{r}", min_value=0, value=0)
            C3 = c3_cols[1].number_input("", key=f"aggC3_{r}", min_value=0, value=0)
            rank_inputs.append((r, int(N), int(C1), int(C2), int(C3)))

        st.divider()

        st.markdown("## 新回収率（累積）")
        st.caption("2車単固定型だけ入力します。1→2345 / 2→13。")

        h4 = st.columns([2.4, 1, 1, 1, 1, 1.2])
        h4[0].markdown("**型**")
        h4[1].markdown("**対象N**")
        h4[2].markdown("**KSUM**")
        h4[3].markdown("**SUM**")
        h4[4].markdown("**H**")
        h4[5].markdown("**U**")

        payout_inputs = []
        for axis in PATTERN_AXES:
            c0, c1, c2, c3, c4, c5 = st.columns([2.4, 1, 1, 1, 1, 1.2])
            c0.write(pattern_label(axis))
            N = c1.number_input("", key=f"prev_2t_pat_{axis}_N", min_value=0, value=0)
            KSUM = c2.number_input("", key=f"prev_2t_pat_{axis}_KSUM", min_value=0, value=0)
            SUM = c3.number_input("", key=f"prev_2t_pat_{axis}_SUM", min_value=0, value=0, step=10)
            H = c4.number_input("", key=f"prev_2t_pat_{axis}_H", min_value=0, value=0)
            U = c5.number_input("", key=f"prev_2t_pat_{axis}_U", min_value=0, value=0)
            payout_inputs.append((axis, int(N), int(KSUM), int(SUM), int(H), int(U)))

        st.divider()

        st.markdown("## 個別回収（累積・任意）")
        st.caption("1→2345と2→13の内訳確認用です。1→2 / 1→3 / 1→4 / 1→5 / 2→1 / 2→3を個別に入力できます。不要なら0のままでOK。")

        h5 = st.columns([2.4, 1, 1, 1, 1, 1.2])
        h5[0].markdown("**型**")
        h5[1].markdown("**対象N**")
        h5[2].markdown("**KSUM**")
        h5[3].markdown("**SUM**")
        h5[4].markdown("**H**")
        h5[5].markdown("**U**")

        axis_target_inputs = []
        individual_pairs = [(1, target) for target in AXIS1_TARGETS] + [(2, target) for target in AXIS2_TARGETS]
        for axis, target in individual_pairs:
            c0, c1, c2, c3, c4, c5 = st.columns([2.4, 1, 1, 1, 1, 1.2])
            c0.write(pair_target_label(axis, target))
            N = c1.number_input("", key=f"prev_{axis}to_{target}_N", min_value=0, value=0)
            KSUM = c2.number_input("", key=f"prev_{axis}to_{target}_KSUM", min_value=0, value=0)
            SUM = c3.number_input("", key=f"prev_{axis}to_{target}_SUM", min_value=0, value=0, step=10)
            H = c4.number_input("", key=f"prev_{axis}to_{target}_H", min_value=0, value=0)
            U = c5.number_input("", key=f"prev_{axis}to_{target}_U", min_value=0, value=0)
            axis_target_inputs.append((axis, target, int(N), int(KSUM), int(SUM), int(H), int(U)))

        st.form_submit_button("前日までの集計を反映")

    for wr, rr, v in pair_inputs:
        if v:
            pair12_manual[(wr, rr)] += int(v)

    for r, N, C1, C2, C3 in rank_inputs:
        if any([N, C1, C2, C3]):
            rec = agg_rank_manual[r]
            rec["N"] += int(N)
            rec["C1"] += int(C1)
            rec["C2"] += int(C2)
            rec["C3"] += int(C3)

    for axis, N, KSUM, SUM, H, U in payout_inputs:
        if any([N, KSUM, SUM, H, U]):
            rec = agg_payout_2t_pattern_manual[axis]
            rec["N"] += int(N)
            rec["KSUM"] += int(KSUM)
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)
            rec["U"] += int(U)

    for axis, target, N, KSUM, SUM, H, U in axis_target_inputs:
        if any([N, KSUM, SUM, H, U]):
            rec = agg_payout_axis_target_manual[(axis, target)]
            rec["N"] += int(N)
            rec["KSUM"] += int(KSUM)
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)
            rec["U"] += int(U)


# =========================
# 集計：日次 + 前日まで累積
# =========================
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

pair12_daily: Dict[PairKey, int] = defaultdict(int)

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    if len(finish) < 2 or not vorder:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    win_rank = car_to_rank.get(finish[0])
    sec_rank = car_to_rank.get(finish[1])

    if win_rank is None or sec_rank is None:
        continue

    pair12_daily[(int(win_rank), int(sec_rank))] += 1

pair12_total: Dict[PairKey, int] = defaultdict(int)
for k, v in pair12_daily.items():
    pair12_total[k] += int(v)
for k, v in pair12_manual.items():
    pair12_total[k] += int(v)

# --- 新回収率（日次） ---
# 2車単：1→2345 / 2→13
payout_2t_pattern_daily: Dict[int, Dict[str, int]] = {
    axis: new_payout_rec() for axis in PATTERN_AXES
}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    field_n = int(row.get("field_n", len(vorder) or 0))

    if not vorder or field_n <= 0 or len(finish) < 2:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}

    win_rank = car_to_rank.get(finish[0])
    sec_rank = car_to_rank.get(finish[1])

    if win_rank is None or sec_rank is None:
        continue

    win_rank = int(win_rank)
    sec_rank = int(sec_rank)
    pay_2t = int(row.get("pay_2t", 0))

    for axis in PATTERN_AXES:
        ksum = ksum_2t_pattern(axis, field_n)
        if ksum <= 0:
            continue

        rec = payout_2t_pattern_daily[axis]
        rec["N"] += 1
        rec["KSUM"] += ksum

        if hit_2t_pattern(axis, win_rank, sec_rank, field_n):
            if pay_2t > 0:
                rec["H"] += 1
                rec["SUM"] += pay_2t
            else:
                rec["U"] += 1

# --- 個別（日次） ---
# 2車単：1→2 / 1→3 / 1→4 / 1→5 / 2→1 / 2→3
INDIVIDUAL_PAIRS = [(1, target) for target in AXIS1_TARGETS] + [(2, target) for target in AXIS2_TARGETS]
payout_axis_target_daily: Dict[Tuple[int, int], Dict[str, int]] = {
    pair: new_payout_rec() for pair in INDIVIDUAL_PAIRS
}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    field_n = int(row.get("field_n", len(vorder) or 0))

    if not vorder or field_n <= 0 or len(finish) < 2:
        continue

    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}

    win_rank = car_to_rank.get(finish[0])
    sec_rank = car_to_rank.get(finish[1])

    if win_rank is None or sec_rank is None:
        continue

    win_rank = int(win_rank)
    sec_rank = int(sec_rank)
    pay_2t = int(row.get("pay_2t", 0))

    for axis, target in INDIVIDUAL_PAIRS:
        ksum = ksum_axis_to_target(axis, target, field_n)
        if ksum <= 0:
            continue

        rec = payout_axis_target_daily[(axis, target)]
        rec["N"] += 1
        rec["KSUM"] += ksum

        if hit_axis_to_target(axis, target, win_rank, sec_rank, field_n):
            if pay_2t > 0:
                rec["H"] += 1
                rec["SUM"] += pay_2t
            else:
                rec["U"] += 1


payout_2t_pattern_total: Dict[int, Dict[str, int]] = {
    axis: new_payout_rec() for axis in PATTERN_AXES
}

for axis in PATTERN_AXES:
    add_rec(payout_2t_pattern_total[axis], payout_2t_pattern_daily[axis])
    add_rec(payout_2t_pattern_total[axis], agg_payout_2t_pattern_manual[axis])

combo_12 = combine_recs([payout_2t_pattern_total[1], payout_2t_pattern_total[2]])

payout_axis_target_total: Dict[Tuple[int, int], Dict[str, int]] = {
    pair: new_payout_rec() for pair in INDIVIDUAL_PAIRS
}

for pair in INDIVIDUAL_PAIRS:
    add_rec(payout_axis_target_total[pair], payout_axis_target_daily[pair])
    add_rec(payout_axis_target_total[pair], agg_payout_axis_target_manual[pair])


# =========================
# 出力：分析結果
# =========================
with tabs[2]:
    st.subheader("1→2 着評価分布（全体累積）｜1着が評価1〜7のとき（欠車対応）")
    st.caption("欠車レースでは存在しない下位評価はNに含まれません。")

    df12_count, df12_pct = build_conditional_tables(pair12_total)

    st.markdown("### 回数（Nは条件付き総数）")
    st.dataframe(df12_count, use_container_width=True, hide_index=True)

    st.markdown("### 割合%（同評価セルは空欄）")
    st.dataframe(df12_pct, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("評価別 入賞テーブル（全体累積）｜欠車対応")
    rows_out = []
    for r in range(1, 8):
        rec = rank_total.get(r, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
        N, C1, C2, C3 = rec["N"], rec["C1"], rec["C2"], rec["C3"]
        rows_out.append(
            {
                "評価": rank_symbol(r),
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

    st.subheader("新回収率｜2車単 1→2345 / 2→13")
    st.caption("三連複と3→46は使いません。2車単の固定型だけを集計します。合算行の対象Nは加算せず、同一レース数として表示します。")

    rows_new = []
    for axis in PATTERN_AXES:
        rows_new.append(
            payout_row(pattern_label(axis), payout_2t_pattern_total[axis])
        )

    rows_new.append(payout_row("合算：1→2345 ＋ 2→13", combo_12))

    st.dataframe(pd.DataFrame(rows_new), use_container_width=True, hide_index=True)

    st.markdown("### 個別回収｜1→2 / 1→3 / 1→4 / 1→5 / 2→1 / 2→3")
    st.caption("1→2345と2→13の内訳確認用です。どの目が回収を支えているかを見るための表です。")

    rows_individual = []
    for axis, target in INDIVIDUAL_PAIRS:
        rows_individual.append(
            payout_row(pair_target_label(axis, target), payout_axis_target_total[(axis, target)])
        )

    st.dataframe(pd.DataFrame(rows_individual), use_container_width=True, hide_index=True)

    st.markdown("### 買い目確認")
    st.write("2車単 1→2345：1→2 / 1→3 / 1→4 / 1→5")
    st.write("2車単 2→13：2→1 / 2→3")
