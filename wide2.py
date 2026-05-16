# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜軸1・2限定 標準棚/穴棚 v6.2｜配当係数つき｜7車固定・欠車対応")

# =========================
# 基本設定（7車ベース）
# =========================
FIELD_SIZE = 7
WINNER_RANKS = tuple(range(1, 8))
PATTERN_AXES = (1,)
AXIS1_TARGETS = (2, 3)
INDIVIDUAL_AXIS1_TARGETS = (2, 3)
AXIS2_TARGETS = ()
AXIS3_TARGETS = ()

# 2車複：軸1・2限定の標準棚 + 穴棚
# 標準棚：1-234 / 2-134
# 穴棚：  1-567 / 2-567
# 評価3は軸ではなく相手候補として扱います。
NISHAFUKU_PAIRS = [
    # 軸1に必要なペア
    (1, 2), (1, 3), (1, 4),
    (1, 5), (1, 6), (1, 7),

    # 軸2に必要なペア
    (2, 3), (2, 4),
    (2, 5), (2, 6), (2, 7),
]
NISHAFUKU_EXTRA_PAIRS = []


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
    return {"N": 0, "KSUM": 0, "H": 0, "SUM": 0}


def add_rec(dst: Dict[str, int], src: Dict[str, int]):
    for k in ("N", "KSUM", "H", "SUM"):
        dst[k] += int(src.get(k, 0))


def combine_recs(recs: List[Dict[str, int]]) -> Dict[str, int]:
    """
    同じレース群に対する複数買い目の合算。
    Nは足さず、最大Nを使う。
    KSUM/H/SUMは合算する。
    """
    out = new_payout_rec()
    out["N"] = max((int(r.get("N", 0)) for r in recs), default=0)
    for rec in recs:
        out["KSUM"] += int(rec.get("KSUM", 0))
        out["H"] += int(rec.get("H", 0))
        out["SUM"] += int(rec.get("SUM", 0))
    return out


def targets_for_pattern(axis: int, field_n: int) -> List[int]:
    """
    2車単固定型の相手評価。
    評価1→23
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
        return "2車単 1→23"
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


def nishafuku_label(a: int, b: int) -> str:
    return f"2車複 {a}-{b}"


NISHAFUKU_SET_DEFS = {
    # 標準棚：軸1・2のみ
    "標準 1-234": [nishafuku_label(1, 2), nishafuku_label(1, 3), nishafuku_label(1, 4)],
    "標準 2-134": [nishafuku_label(1, 2), nishafuku_label(2, 3), nishafuku_label(2, 4)],

    # 穴棚：軸1・2のみ
    "穴 1-567": [nishafuku_label(1, 5), nishafuku_label(1, 6), nishafuku_label(1, 7)],
    "穴 2-567": [nishafuku_label(2, 5), nishafuku_label(2, 6), nishafuku_label(2, 7)],
}

agg_payout_nishafuku_set_manual: Dict[str, Dict[str, int]] = {
    set_label: new_payout_rec() for set_label in NISHAFUKU_SET_DEFS
}


def ksum_nishafuku_pair(a: int, b: int, field_n: int) -> int:
    """2車複：評価a-b の点数。存在する評価だけ1点として扱う。"""
    if field_n < 2:
        return 0
    if a > field_n or b > field_n:
        return 0
    if a == b:
        return 0
    return 1


def hit_nishafuku_pair(a: int, b: int, win_rank: int, sec_rank: int, field_n: int) -> bool:
    """2車複：実際の1着・2着評価順位がa-bなら的中。順不同。"""
    if ksum_nishafuku_pair(a, b, field_n) <= 0:
        return False
    return {win_rank, sec_rank} == {a, b}



def payout_row(label: str, rec: Dict[str, int]) -> Dict:
    N = int(rec["N"])
    KSUM = int(rec["KSUM"])
    H = int(rec["H"])
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
        "的中率%": hit_rate,
        "平均配当": avg_pay,
        "回収率%": roi,
        "判定": "",
    }


def rec_for_labels(source: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, int]:
    """指定ラベル群の合算レコード。Nは最大N、KSUM/H/SUMは合算。"""
    return combine_recs([source[label] for label in labels if label in source])


def expected_set_hit_rate_from_pair12(labels: List[str], pair12_counts: Dict[PairKey, int]) -> float | None:
    """
    1→2着評価分布から、2車複セットの想定的中率を出す。
    2車複なので順不同で集計する。

    例：
      1-234 = 1-2 / 1-3 / 1-4
      → (1,2)+(2,1)+(1,3)+(3,1)+(1,4)+(4,1)
    """
    total = sum(int(v) for v in pair12_counts.values())
    if total <= 0:
        return None

    hit = 0
    for label in labels:
        # label例: "2車複 1-4"
        try:
            pair_part = label.replace("2車複", "").strip()
            a_str, b_str = pair_part.split("-")
            a, b = int(a_str), int(b_str)
        except Exception:
            continue

        hit += int(pair12_counts.get((a, b), 0))
        hit += int(pair12_counts.get((b, a), 0))

    return round(100.0 * hit / total, 1)


def payout_row_with_expected_set_hit(label: str, rec: Dict[str, int], labels: List[str], pair12_counts: Dict[PairKey, int]) -> Dict:
    """想定セット的中率と、実的中率との差を併記した行。"""
    row = payout_row(label, rec)

    expected_hit = expected_set_hit_rate_from_pair12(labels, pair12_counts)
    row["想定セット的中率%"] = expected_hit

    if row["的中率%"] is not None and expected_hit is not None:
        row["想定との差"] = round(row["的中率%"] - expected_hit, 1)
    else:
        row["想定との差"] = None

    return row


def expected_pair_hit_rate_from_pair12(a: int, b: int, pair12_counts: Dict[PairKey, int]) -> float | None:
    """1→2着評価分布から、評価a-bの2車複想定的中率を出す。順不同。"""
    total = sum(int(v) for v in pair12_counts.values())
    if total <= 0:
        return None
    hit = int(pair12_counts.get((a, b), 0)) + int(pair12_counts.get((b, a), 0))
    return round(100.0 * hit / total, 1)


def diff_status(diff, expected=None) -> str:
    """想定との差の状態をざっくり表示。想定0%は候補対象外。"""
    if expected is not None and expected == 0:
        return "対象外"
    if diff is None:
        return ""
    if diff >= 10:
        return "当たりすぎ"
    if diff <= -10:
        return "当たらなすぎ"
    return "中庸"



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
# 2車単：1→23
agg_payout_2t_pattern_manual: Dict[int, Dict[str, int]] = {
    axis: new_payout_rec() for axis in PATTERN_AXES
}

# 前日まで：個別回収（任意入力）
# 1→2 / 1→3
agg_payout_axis_target_manual: Dict[Tuple[int, int], Dict[str, int]] = {
    (1, target): new_payout_rec() for target in INDIVIDUAL_AXIS1_TARGETS
}

# 前日まで：2車複 1-23 + 5-12
agg_payout_nishafuku_manual: Dict[str, Dict[str, int]] = {
    nishafuku_label(a, b): new_payout_rec() for a, b in NISHAFUKU_PAIRS
}
for a, b in NISHAFUKU_EXTRA_PAIRS:
    agg_payout_nishafuku_manual[nishafuku_label(a, b)] = new_payout_rec()




# =========================
# A. 日次手入力（欠車対応）
# =========================
with tabs[0]:
    st.subheader("日次手入力（7車ベース・欠車対応・最大36R）")
    st.caption(
        "入力中の白化を抑えるため、フォーム送信式です。"
        "V評価は頭数ぶんの桁数で入力（例：7車=1432567 / 6車=143256）。"
        "着順は～3桁。2車複配当を入力。配当は100円あたりの払戻金（円）です。"
    )

    with st.form("daily_input_form"):
        cols_hdr = st.columns([1, 1.1, 2.9, 1.2, 1.2])
        cols_hdr[0].markdown("**R**")
        cols_hdr[1].markdown("**頭数**")
        cols_hdr[2].markdown("**V評価（頭数ぶんの桁数）**")
        cols_hdr[3].markdown("**着順(～3桁)**")
        cols_hdr[4].markdown("**2車複配当**")

        daily_inputs = []

        for i in range(1, 37):
            c1, c2, c3, c4, c5 = st.columns([1, 1.1, 2.9, 1.2, 1.2])

            rid = c1.text_input("", key=f"rid_{i}", value=str(i))
            field_n = c2.selectbox("", options=[7, 6, 5], index=0, key=f"field_n_{i}")
            vline = c3.text_input("", key=f"vline_{i}", value="")
            fin = c4.text_input("", key=f"fin_{i}", value="")
            pay_2f = c5.number_input("", key=f"pay2f_{i}", min_value=0, value=0, step=10)
            pay_2t = 0

            daily_inputs.append(
                {
                    "rid": rid,
                    "field_n": field_n,
                    "vline": vline,
                    "fin": fin,
                    "pay_2t": pay_2t,
                    "pay_2f": pay_2f,
                }
            )

        st.form_submit_button("日次入力を反映")

    for item in daily_inputs:
        rid = item["rid"]
        field_n = int(item["field_n"])
        vline = item["vline"]
        fin = item["fin"]
        pay_2t = int(item["pay_2t"])
        pay_2f = int(item["pay_2f"])

        vorder = parse_rankline(vline, field_n)
        finish = parse_finish(fin)

        any_input = any([vline.strip(), fin.strip(), pay_2f > 0])
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
                    "pay_2f": pay_2f,
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

        # 固定型（1→23 / 2→13）の累積入力は削除。
        # 必要な確認は下の「個別回収」で行う。
        payout_inputs = []


        st.divider()

        st.markdown("## 2車複シミュレーター用 前日集計（累積・任意）")
        st.caption("出力シミュレーターと同じセットを入力します。標準棚：1-234 / 2-134、穴棚：1-567 / 2-567。不要なら0のままでOK。")

        h6 = st.columns([2.4, 1, 1, 1, 1])
        h6[0].markdown("**型**")
        h6[1].markdown("**対象N**")
        h6[2].markdown("**KSUM**")
        h6[3].markdown("**SUM**")
        h6[4].markdown("**H**")

        nishafuku_set_inputs = []
        for set_label in NISHAFUKU_SET_DEFS.keys():
            safe_key = set_label.replace(" ", "_").replace("-", "_")
            c0, c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1, 1])
            c0.write(set_label)
            N = c1.number_input("", key=f"prev_2fset_{safe_key}_N", min_value=0, value=0)
            KSUM = c2.number_input("", key=f"prev_2fset_{safe_key}_KSUM", min_value=0, value=0)
            SUM = c3.number_input("", key=f"prev_2fset_{safe_key}_SUM", min_value=0, value=0, step=10)
            H = c4.number_input("", key=f"prev_2fset_{safe_key}_H", min_value=0, value=0)
            nishafuku_set_inputs.append((set_label, int(N), int(KSUM), int(SUM), int(H)))

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

    for axis, N, KSUM, SUM, H in payout_inputs:
        if any([N, KSUM, SUM, H]):
            rec = agg_payout_2t_pattern_manual[axis]
            rec["N"] += int(N)
            rec["KSUM"] += int(KSUM)
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)

    for set_label, N, KSUM, SUM, H in nishafuku_set_inputs:
        if any([N, KSUM, SUM, H]):
            rec = agg_payout_nishafuku_set_manual[set_label]
            rec["N"] += int(N)
            rec["KSUM"] += int(KSUM)
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)


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
# 2車単：1→23
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

# --- 個別（日次） ---
# 2車単：1→2 / 1→3
INDIVIDUAL_PAIRS = [(1, target) for target in INDIVIDUAL_AXIS1_TARGETS]
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


# --- 2車複シミュレーション（日次） ---
payout_nishafuku_daily: Dict[str, Dict[str, int]] = {
    nishafuku_label(a, b): new_payout_rec() for a, b in NISHAFUKU_PAIRS
}
for a, b in NISHAFUKU_EXTRA_PAIRS:
    payout_nishafuku_daily[nishafuku_label(a, b)] = new_payout_rec()

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
    pay_2f = int(row.get("pay_2f", 0))


    for a, b in NISHAFUKU_PAIRS:
        label = nishafuku_label(a, b)
        ksum = ksum_nishafuku_pair(a, b, field_n)
        if ksum <= 0:
            continue

        rec = payout_nishafuku_daily[label]
        rec["N"] += 1
        rec["KSUM"] += ksum

        if hit_nishafuku_pair(a, b, win_rank, sec_rank, field_n):
            if pay_2f > 0:
                rec["H"] += 1
                rec["SUM"] += pay_2f


    for a, b in NISHAFUKU_EXTRA_PAIRS:
        label = nishafuku_label(a, b)
        ksum = ksum_nishafuku_pair(a, b, field_n)
        if ksum <= 0:
            continue

        rec = payout_nishafuku_daily[label]
        rec["N"] += 1
        rec["KSUM"] += ksum

        if hit_nishafuku_pair(a, b, win_rank, sec_rank, field_n):
            if pay_2f > 0:
                rec["H"] += 1
                rec["SUM"] += pay_2f






payout_2t_pattern_total: Dict[int, Dict[str, int]] = {
    axis: new_payout_rec() for axis in PATTERN_AXES
}

for axis in PATTERN_AXES:
    add_rec(payout_2t_pattern_total[axis], payout_2t_pattern_daily[axis])
    add_rec(payout_2t_pattern_total[axis], agg_payout_2t_pattern_manual[axis])


payout_axis_target_total: Dict[Tuple[int, int], Dict[str, int]] = {
    pair: new_payout_rec() for pair in INDIVIDUAL_PAIRS
}

for pair in INDIVIDUAL_PAIRS:
    add_rec(payout_axis_target_total[pair], payout_axis_target_daily[pair])
    add_rec(payout_axis_target_total[pair], agg_payout_axis_target_manual[pair])

payout_nishafuku_total: Dict[str, Dict[str, int]] = {
    nishafuku_label(a, b): new_payout_rec() for a, b in NISHAFUKU_PAIRS
}
for a, b in NISHAFUKU_EXTRA_PAIRS:
    payout_nishafuku_total[nishafuku_label(a, b)] = new_payout_rec()

for label in payout_nishafuku_total.keys():
    add_rec(payout_nishafuku_total[label], payout_nishafuku_daily[label])
    add_rec(payout_nishafuku_total[label], agg_payout_nishafuku_manual[label])





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
    st.markdown("### 2車複シミュレーター｜軸1・2限定 標準棚 / 穴棚")
    st.caption("前日までのセット入力と、今日入力した個別2車複を合算し、1→2着評価分布から算出した想定セット的中率との差を表示します。軸は評価1・評価2のみ。判定☆は、標準棚・穴棚それぞれで想定との差が最も0に近い候補です。")

    sim_sets_2f = NISHAFUKU_SET_DEFS

    rows_2f_sim = []
    for set_label, labels in sim_sets_2f.items():
        today_set_rec = rec_for_labels(payout_nishafuku_total, labels)
        total_set_rec = combine_recs([agg_payout_nishafuku_set_manual[set_label], today_set_rec])
        row = payout_row_with_expected_set_hit(
            set_label,
            total_set_rec,
            labels,
            pair12_total,
        )
        row["棚"] = "穴" if set_label.startswith("穴") else "標準"
        rows_2f_sim.append(row)

    df_sim = pd.DataFrame(rows_2f_sim)

    # 候補判定：
    # 回収率100%以上ではなく、想定との差が0に一番近い行を☆候補にする。
    # 標準棚と穴棚は別グループとして扱い、それぞれ1つずつ☆を出す。
    # サンプルがない行、想定との差がNoneの行は対象外。
    if not df_sim.empty and "想定との差" in df_sim.columns:
        df_sim["判定"] = ""
        if "棚" in df_sim.columns:
            for shelf_name in ["標準", "穴"]:
                shelf_mask = (df_sim["棚"] == shelf_name) & df_sim["想定との差"].notna()
                if shelf_mask.any():
                    idx_star = df_sim.loc[shelf_mask, "想定との差"].abs().idxmin()
                    df_sim.loc[idx_star, "判定"] = "☆"
        else:
            valid_mask = df_sim["想定との差"].notna()
            if valid_mask.any():
                idx_star = df_sim.loc[valid_mask, "想定との差"].abs().idxmin()
                df_sim.loc[idx_star, "判定"] = "☆"

    # 見やすい列順に整理
    preferred_cols = [
        "棚",
        "型",
        "対象N",
        "総点数KSUM",
        "投資額換算",
        "払戻合計SUM",
        "的中H（配当あり）",
        "的中率%",
        "想定セット的中率%",
        "想定との差",
        "平均配当",
        "回収率%",
        "判定",
    ]
    df_sim = df_sim[[c for c in preferred_cols if c in df_sim.columns]]
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

    st.markdown("### 最終軸候補｜軸1・2限定 標準棚＋穴棚 合成")
    st.caption(
        "評価1・評価2のみを軸候補にします。"
        "通常は直近3回の波判定を優先。履歴未入力時は従来の軸別安定度で判定します。"
    )

    # 軸ごとに標準棚・穴棚をまとめる
    axis_defs = {
        1: ("標準 1-234", "穴 1-567"),
        2: ("標準 2-134", "穴 2-567"),
    }

    def _num_or_none(v):
        """None/NaNを安全に処理してfloatへ。"""
        try:
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    def _avg_diff(std_diff, hole_diff):
        """標準差と穴差の合成値。両方ある時だけ平均する。"""
        s = _num_or_none(std_diff)
        h = _num_or_none(hole_diff)
        if s is None or h is None:
            return None
        return round((s + h) / 2.0, 2)

    def _median(values):
        vals = sorted([float(v) for v in values if v is not None and not pd.isna(v)])
        if not vals:
            return None
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def _mean(values):
        vals = [float(v) for v in values if v is not None and not pd.isna(v)]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _std_pop(values):
        vals = [float(v) for v in values if v is not None and not pd.isna(v)]
        if len(vals) <= 1:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5

    def _deviation_stats(value, all_values):
        """候補群内での平均との差・中央値差・偏差値を返す。"""
        v = _num_or_none(value)
        vals = [float(x) for x in all_values if x is not None and not pd.isna(x)]
        if v is None or not vals:
            return {
                "平均との差": None,
                "中央値差": None,
                "偏差値": None,
                "基準位置": "",
            }
        m = _mean(vals)
        med = _median(vals)
        sd = _std_pop(vals)
        if sd and sd > 0:
            z = round(50 + 10 * ((v - m) / sd), 1)
        else:
            z = 50.0
        mean_delta = round(v - m, 1) if m is not None else None
        median_delta = round(v - med, 1) if med is not None else None

        if median_delta is not None and median_delta < 0 and z < 50:
            pos = "基準下"
        elif median_delta is not None and median_delta > 0 and z > 50:
            pos = "基準上"
        else:
            pos = "中央値付近"

        return {
            "平均との差": mean_delta,
            "中央値差": median_delta,
            "偏差値": z,
            "基準位置": pos,
        }

    def _wave_judge(values, baseline_median_delta=None, baseline_z=None):
        """
        直近3回の合成差から波判定。
        ここでは単なるプラス/マイナスではなく、候補群の中央値・偏差値より下にいるかも見る。
        優先順位：
        1) 基準下 + マイナス域 + 3回連続上昇
        2) 基準下 + 3回連続上昇
        3) 基準下 + 上昇傾向
        4) マイナス域 + 上昇傾向
        5) 基準下
        6) その他
        """
        vals = [float(v) for v in values if v is not None and not pd.isna(v)]
        if len(vals) < 2:
            return {
                "波判定": "履歴不足",
                "波優先度": None,
                "直近3回傾き": None,
                "現在合成差": vals[-1] if vals else None,
            }

        # 直近3回だけ使う
        vals = vals[-3:]
        current = vals[-1]
        slope = round(vals[-1] - vals[0], 2)

        if len(vals) >= 3:
            rising_strict = vals[0] < vals[1] < vals[2]
            rising_soft = vals[2] > vals[1] and vals[2] > vals[0]
        else:
            rising_strict = vals[1] > vals[0]
            rising_soft = rising_strict

        below_baseline = False
        if baseline_median_delta is not None and not pd.isna(baseline_median_delta):
            below_baseline = float(baseline_median_delta) < 0
        if baseline_z is not None and not pd.isna(baseline_z):
            below_baseline = below_baseline or float(baseline_z) < 50

        if below_baseline and current < 0 and rising_strict:
            label = "◎ 基準下＋マイナス域＋連続上昇"
            group = 1
        elif below_baseline and rising_strict:
            label = "○ 基準下＋連続上昇"
            group = 2
        elif below_baseline and rising_soft:
            label = "○ 基準下＋上昇"
            group = 3
        elif current < 0 and rising_soft:
            label = "△ マイナス域＋上昇"
            group = 4
        elif below_baseline:
            label = "△ 基準下"
            group = 5
        elif rising_soft:
            label = "▲ 上昇中（基準上）"
            group = 6
        else:
            label = "見送り"
            group = 7

        # 小さいほど優先。group優先、次に傾きが大きい軸、最後に現在値が0へ近い軸。
        priority = round(group * 10000 - slope * 100 + abs(current), 2)
        return {
            "波判定": label,
            "波優先度": priority,
            "直近3回傾き": slope,
            "現在合成差": current,
        }

    # ---- 波履歴入力：過去2回 + 現在値 = 直近3回 ----
    st.markdown("#### 波履歴入力｜過去2回分")
    st.caption(
        "各軸の過去2回分の標準差・穴差を入れると、現在値と合わせて直近3回の波を判定します。"
        "未入力なら従来の安定度判定に戻します。"
    )

    wave_history = {}
    with st.expander("波履歴を入力する（任意）", expanded=False):
        st.write("入力例：前々回=-14.2、前回=-7.8、今回=現在自動計算。")
        for axis in [1, 2]:
            st.markdown(f"**評価{axis}**")
            c1, c2, c3, c4 = st.columns(4)
            prev2_std = c1.number_input(
                "前々回 標準差",
                key=f"wave_axis{axis}_prev2_std",
                value=0.0,
                step=0.1,
                format="%.1f",
            )
            prev2_hole = c2.number_input(
                "前々回 穴差",
                key=f"wave_axis{axis}_prev2_hole",
                value=0.0,
                step=0.1,
                format="%.1f",
            )
            prev1_std = c3.number_input(
                "前回 標準差",
                key=f"wave_axis{axis}_prev1_std",
                value=0.0,
                step=0.1,
                format="%.1f",
            )
            prev1_hole = c4.number_input(
                "前回 穴差",
                key=f"wave_axis{axis}_prev1_hole",
                value=0.0,
                step=0.1,
                format="%.1f",
            )

            # 0.0を未入力扱いにすると本当に0.0の履歴を使えないので、チェックで使用可否を決める
            use_hist = st.checkbox(f"評価{axis}の波履歴を使う", key=f"wave_axis{axis}_use", value=False)
            wave_history[axis] = {
                "use": use_hist,
                "prev2_comp": _avg_diff(prev2_std, prev2_hole),
                "prev1_comp": _avg_diff(prev1_std, prev1_hole),
            }

    # 軸候補の現在合成差を先に作り、候補群内の平均・中央値・偏差値を判定に使う。
    axis_current_comps = {}
    for axis, (std_label, hole_label) in axis_defs.items():
        std_row = df_sim[df_sim["型"] == std_label]
        hole_row = df_sim[df_sim["型"] == hole_label]
        std_diff_tmp = std_row.iloc[0].get("想定との差") if not std_row.empty else None
        hole_diff_tmp = hole_row.iloc[0].get("想定との差") if not hole_row.empty else None
        axis_current_comps[axis] = _avg_diff(std_diff_tmp, hole_diff_tmp)
    axis_comp_values = list(axis_current_comps.values())

    axis_rows = []
    wave_chart_rows = []

    for axis, (std_label, hole_label) in axis_defs.items():
        std_row = df_sim[df_sim["型"] == std_label]
        hole_row = df_sim[df_sim["型"] == hole_label]

        std_diff = None
        hole_diff = None
        std_roi = None
        hole_roi = None

        if not std_row.empty:
            std_diff = std_row.iloc[0].get("想定との差")
            std_roi = std_row.iloc[0].get("回収率%")
        if not hole_row.empty:
            hole_diff = hole_row.iloc[0].get("想定との差")
            hole_roi = hole_row.iloc[0].get("回収率%")

        if pd.notna(std_diff) and pd.notna(hole_diff):
            stability = round(abs(float(std_diff)) + abs(float(hole_diff)), 1)
        else:
            stability = None

        current_comp = _avg_diff(std_diff, hole_diff)
        hist = wave_history.get(axis, {"use": False})

        if hist.get("use") and current_comp is not None:
            wave_values = [hist.get("prev2_comp"), hist.get("prev1_comp"), current_comp]
        else:
            wave_values = [current_comp]

        axis_dev = _deviation_stats(current_comp, axis_comp_values)
        wave_info = _wave_judge(
            wave_values,
            baseline_median_delta=axis_dev.get("中央値差"),
            baseline_z=axis_dev.get("偏差値"),
        )

        # グラフ用
        if hist.get("use") and current_comp is not None:
            wave_chart_rows.extend(
                [
                    {"回": "前々回", "軸": f"評価{axis}", "合成差": hist.get("prev2_comp")},
                    {"回": "前回", "軸": f"評価{axis}", "合成差": hist.get("prev1_comp")},
                    {"回": "現在", "軸": f"評価{axis}", "合成差": current_comp},
                ]
            )

        axis_rows.append(
            {
                "軸": f"評価{axis}",
                "標準型": std_label,
                "標準差": std_diff,
                "標準回収率%": std_roi,
                "穴型": hole_label,
                "穴差": hole_diff,
                "穴回収率%": hole_roi,
                "現在合成差": current_comp,
                "平均との差": axis_dev.get("平均との差"),
                "中央値差": axis_dev.get("中央値差"),
                "偏差値": axis_dev.get("偏差値"),
                "基準位置": axis_dev.get("基準位置"),
                "直近3回傾き": wave_info["直近3回傾き"],
                "波判定": wave_info["波判定"],
                "波優先度": wave_info["波優先度"],
                "軸別安定度": stability,
                "最終軸候補": "",
            }
        )

    df_axis = pd.DataFrame(axis_rows)

    if wave_chart_rows:
        df_wave_chart = pd.DataFrame(wave_chart_rows)
        df_wave_pivot = df_wave_chart.pivot(index="回", columns="軸", values="合成差").reindex(["前々回", "前回", "現在"])
        st.markdown("#### 波グラフ｜標準差・穴差の合成値")
        st.line_chart(df_wave_pivot)

    final_axis = None

    # 波履歴が使える軸がある場合は波優先度で選ぶ。
    # なければ従来どおり軸別安定度で選ぶ。
    if not df_axis.empty and df_axis["波優先度"].notna().any():
        idx_final = df_axis["波優先度"].idxmin()
        df_axis.loc[idx_final, "最終軸候補"] = "☆"
        final_axis = int(str(df_axis.loc[idx_final, "軸"]).replace("評価", ""))
        st.success(f"波判定での最終軸候補：評価{final_axis}")
    elif not df_axis.empty and df_axis["軸別安定度"].notna().any():
        idx_final = df_axis["軸別安定度"].idxmin()
        df_axis.loc[idx_final, "最終軸候補"] = "☆"
        final_axis = int(str(df_axis.loc[idx_final, "軸"]).replace("評価", ""))
        st.info(f"履歴未入力のため、従来の安定度で判定：評価{final_axis}")

    preferred_axis_cols = [
        "最終軸候補",
        "軸",
        "標準型",
        "標準差",
        "標準回収率%",
        "穴型",
        "穴差",
        "穴回収率%",
        "現在合成差",
        "平均との差",
        "中央値差",
        "偏差値",
        "基準位置",
        "直近3回傾き",
        "波判定",
        "軸別安定度",
    ]
    df_axis = df_axis[[c for c in preferred_axis_cols if c in df_axis.columns]]
    st.dataframe(df_axis, use_container_width=True, hide_index=True)

    st.markdown("#### 配当収束シミュレーション設定")
    st.caption("的中率のズレだけでなく、平均配当が基準より安すぎる/高すぎるかも候補判定に加えます。")
    c_pay1, c_pay2, c_pay3 = st.columns([1, 1, 2])
    BASE_AVG_PAY = c_pay1.number_input(
        "基準平均配当",
        key="base_avg_pay",
        min_value=100,
        value=1200,
        step=50,
    )
    OVERHEAT_Z = c_pay2.number_input(
        "過熱偏差値ライン",
        key="overheat_z_line",
        min_value=50.0,
        max_value=80.0,
        value=60.0,
        step=1.0,
        format="%.1f",
    )
    c_pay3.write("基準例：平塚A級7車の平均配当1397円を参考に、実戦では1200円前後を初期値にしています。")

    if final_axis is not None:
        st.markdown("### 最終相手候補｜軸別ペア比較")
        st.caption("最終軸と各評価の2車複を比較します。的中率偏差値に加え、平均配当が基準配当へ戻る余地も見ます。")

        pair_rows = []
        for opp in range(1, 8):
            if opp == final_axis:
                continue

            a, b = sorted((final_axis, opp))
            label = nishafuku_label(a, b)
            rec = payout_nishafuku_total.get(label, new_payout_rec())
            row = payout_row(label, rec)

            expected_pair = expected_pair_hit_rate_from_pair12(final_axis, opp, pair12_total)
            row["軸"] = f"評価{final_axis}"
            row["相手"] = opp
            row["想定ペア的中率%"] = expected_pair

            if row["的中率%"] is not None and expected_pair is not None:
                diff = round(row["的中率%"] - expected_pair, 1)
            else:
                diff = None

            row["想定との差"] = diff
            row["状態"] = diff_status(diff, expected_pair)

            avg_pay = row.get("平均配当")
            if avg_pay is not None and pd.notna(avg_pay) and BASE_AVG_PAY > 0:
                row["配当係数"] = round(float(avg_pay) / float(BASE_AVG_PAY), 2)
                row["配当差"] = round(float(avg_pay) - float(BASE_AVG_PAY), 1)
            else:
                row["配当係数"] = None
                row["配当差"] = None

            row["配当位置"] = ""
            row["総合候補理由"] = ""
            row["相手候補"] = ""
            pair_rows.append(row)

        df_pairs = pd.DataFrame(pair_rows)

        # 相手候補：
        # 旧ロジックの「想定との差が0に近い順」は、当たりすぎ/安定側を拾いやすい。
        # ここでは候補群内の平均との差・中央値差・偏差値を出し、
        # 「基準下＝下振れ圏」から戻り候補を優先する。
        if not df_pairs.empty and df_pairs["想定との差"].notna().any():
            candidate_mask = (
                df_pairs["想定との差"].notna()
                & df_pairs["想定ペア的中率%"].notna()
                & (df_pairs["想定ペア的中率%"] > 0)
            )

            diff_values = df_pairs.loc[candidate_mask, "想定との差"].tolist() if candidate_mask.any() else []
            pay_values = df_pairs.loc[candidate_mask & df_pairs["平均配当"].notna(), "平均配当"].tolist() if candidate_mask.any() else []
            for idx in df_pairs.index:
                stats = _deviation_stats(df_pairs.loc[idx, "想定との差"], diff_values)
                df_pairs.loc[idx, "平均との差"] = stats.get("平均との差")
                df_pairs.loc[idx, "中央値差"] = stats.get("中央値差")
                df_pairs.loc[idx, "偏差値"] = stats.get("偏差値")
                df_pairs.loc[idx, "基準位置"] = stats.get("基準位置")

                pay_stats = _deviation_stats(df_pairs.loc[idx, "平均配当"], pay_values)
                df_pairs.loc[idx, "配当偏差値"] = pay_stats.get("偏差値")

                coef = df_pairs.loc[idx, "配当係数"]
                if pd.isna(coef):
                    df_pairs.loc[idx, "配当位置"] = "未回収"
                elif float(coef) < 0.80:
                    df_pairs.loc[idx, "配当位置"] = "安すぎ"
                elif float(coef) <= 1.30:
                    df_pairs.loc[idx, "配当位置"] = "基準付近"
                else:
                    df_pairs.loc[idx, "配当位置"] = "高すぎ"

            if candidate_mask.any():
                # 基準下かつ想定的中率がある候補を優先。
                # さらに平均配当が基準より安い/基準付近なら「配当戻り余地あり」として加点する。
                expected_median = _median(df_pairs.loc[candidate_mask, "想定ペア的中率%"].tolist())
                df_pairs["_expected_ok"] = df_pairs["想定ペア的中率%"].apply(
                    lambda x: bool(pd.notna(x) and expected_median is not None and float(x) >= float(expected_median))
                )
                df_pairs["_below_base"] = (
                    (df_pairs["中央値差"].notna() & (df_pairs["中央値差"] < 0))
                    | (df_pairs["偏差値"].notna() & (df_pairs["偏差値"] < 50))
                )
                df_pairs["_overhit"] = df_pairs["偏差値"].notna() & (df_pairs["偏差値"] >= float(OVERHEAT_Z))
                df_pairs["_cold"] = df_pairs["偏差値"].notna() & (df_pairs["偏差値"] <= 40)
                df_pairs["_pay_low"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] < 0.80)
                df_pairs["_pay_near"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] >= 0.80) & (df_pairs["配当係数"] <= 1.30)
                df_pairs["_pay_high"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] > 1.30)

                # 優先順：
                # 1) 基準下＋想定率中央値以上＋配当安すぎ/基準付近（的中率も配当も戻り余地）
                # 2) 基準下＋想定率中央値以上
                # 3) 中央値付近＋配当安すぎ（1-2のような配当戻り候補）
                # 4) 基準下
                # 5) 中央値付近で過熱していないもの
                # 偏差値60以上は過熱扱いで強く減点。
                df_pairs["_候補優先"] = 9
                df_pairs.loc[
                    candidate_mask & ~df_pairs["_overhit"] & df_pairs["_below_base"] & df_pairs["_expected_ok"] & (df_pairs["_pay_low"] | df_pairs["_pay_near"]),
                    "_候補優先",
                ] = 1
                df_pairs.loc[
                    candidate_mask & ~df_pairs["_overhit"] & df_pairs["_below_base"] & df_pairs["_expected_ok"],
                    "_候補優先",
                ] = 2
                df_pairs.loc[
                    candidate_mask & ~df_pairs["_overhit"] & ~df_pairs["_below_base"] & df_pairs["_pay_low"],
                    "_候補優先",
                ] = 3
                df_pairs.loc[
                    candidate_mask & ~df_pairs["_overhit"] & df_pairs["_below_base"],
                    "_候補優先",
                ] = 4
                df_pairs.loc[
                    candidate_mask
                    & ~df_pairs["_overhit"]
                    & ~df_pairs["_below_base"]
                    & df_pairs["中央値差"].notna()
                    & (df_pairs["中央値差"].abs() <= 3.0),
                    "_候補優先",
                ] = 5

                df_pairs["配当戻り余地"] = ""
                df_pairs.loc[df_pairs["_pay_low"], "配当戻り余地"] = "あり"
                df_pairs.loc[df_pairs["_pay_near"], "配当戻り余地"] = "中庸"
                df_pairs.loc[df_pairs["_pay_high"], "配当戻り余地"] = "上振れ警戒"
                df_pairs.loc[df_pairs["_overhit"], "総合候補理由"] = "的中率過熱"
                df_pairs.loc[df_pairs["_below_base"] & df_pairs["_pay_low"], "総合候補理由"] = "下振れ＋配当安"
                df_pairs.loc[df_pairs["_below_base"] & df_pairs["_pay_near"], "総合候補理由"] = "下振れ＋配当中庸"
                df_pairs.loc[(~df_pairs["_below_base"]) & df_pairs["_pay_low"], "総合候補理由"] = "配当戻り狙い"

                # 並び順では、優先度→配当係数の低さ→偏差値の低さ→想定的中率を使う。
                candidate_df = df_pairs.loc[candidate_mask].sort_values(
                    ["_候補優先", "配当係数", "偏差値", "想定ペア的中率%", "相手"],
                    ascending=[True, True, True, False, True],
                )
                candidate_idx = candidate_df.head(3).index
                df_pairs.loc[candidate_idx, "相手候補"] = "☆"
                recommended_opps = sorted([int(x) for x in df_pairs.loc[candidate_idx, "相手"].tolist()])
                recommended_text = "".join(str(x) for x in recommended_opps)
                st.success(f"本日の推奨セット候補：評価{final_axis}-{recommended_text}")

                df_pairs = df_pairs.drop(columns=["_expected_ok", "_below_base", "_overhit", "_cold", "_pay_low", "_pay_near", "_pay_high", "_候補優先"])
            else:
                st.info("候補対象となる相手がありません。想定ペア的中率0%の組み合わせは除外しています。")
        else:
            st.info("相手候補を出すには、1→2着評価分布と日次2車複データが必要です。")

        preferred_pair_cols = [
            "相手候補",
            "軸",
            "相手",
            "型",
            "対象N",
            "的中H（配当あり）",
            "的中率%",
            "想定ペア的中率%",
            "想定との差",
            "平均との差",
            "中央値差",
            "偏差値",
            "基準位置",
            "状態",
            "平均配当",
            "配当係数",
            "配当差",
            "配当偏差値",
            "配当位置",
            "配当戻り余地",
            "総合候補理由",
            "回収率%",
        ]
        df_pairs = df_pairs[[c for c in preferred_pair_cols if c in df_pairs.columns]]
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)

    st.markdown("### 買い目確認")
    st.write("今日入力の個別2車複：標準棚＋穴棚に必要なペアを自動集計")
    st.write("前日集計入力・出力シミュレーター：標準 1-234 / 2-134、穴 1-567 / 2-567")
