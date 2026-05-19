# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ヴェロビ復習（全体累積）", layout="wide")
st.title("ヴェロビ 復習（全体累積）｜軸1・2限定 個別2車複 v10.0｜想定回収率・回収差判定｜固定想定ペア的%｜ペア別基準配当｜引継ぎ表つき｜7車固定・欠車対応")

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

# 小倉ミッドナイトA級7車・直近2年ベースのペア別平均配当（100円あたり）。
# 必要に応じて画面上で上書きできます。
PAIR_BASE_AVG_PAY_DEFAULTS = {
    "1-2": 271,
    "1-3": 436,
    "1-4": 654,
    "1-5": 1059,
    "1-6": 1754,
    "1-7": 1519,
    "2-3": 881,
    "2-4": 1333,
    "2-5": 1869,
    "2-6": 1657,
    "2-7": 4092,
}

# 小倉ミッドナイトA級7車・直近2年ベースの固定想定的中率（%）。
# 母数738R：1-2=167回、1-3=105回、1-4=78回、1-5=38回、1-6=26回、1-7=35回、
# 2-3=46回、2-4=42回、2-5=27回、2-6=15回、2-7=22回。
# 現在入力中の的中率とは独立した「想定ペア的%」として使用します。
PAIR_BASE_HIT_RATE_DEFAULTS = {
    "1-2": 22.6,
    "1-3": 14.2,
    "1-4": 10.6,
    "1-5": 5.1,
    "1-6": 3.5,
    "1-7": 4.7,
    "2-3": 6.2,
    "2-4": 5.7,
    "2-5": 3.7,
    "2-6": 2.0,
    "2-7": 3.0,
}

# 3連複 1-2-全の固定想定値。
# 小倉ミッドナイトA級7車・直近2年の3連複全集計より、
# 1-2-3=101回/平均378円、1-2-4=88回/537円、
# 1-2-5=57回/1053円、1-2-6=44回/1136円、1-2-7=30回/1545円。
TRIO_12_ALL_BASE_COUNTS = {
    "1-2-3": 101,
    "1-2-4": 88,
    "1-2-5": 57,
    "1-2-6": 44,
    "1-2-7": 30,
}
TRIO_12_ALL_BASE_AVG_PAYS = {
    "1-2-3": 378,
    "1-2-4": 537,
    "1-2-5": 1053,
    "1-2-6": 1136,
    "1-2-7": 1545,
}

# 小倉ミッドナイトA級7車・直近2年の3連複全集計。
# 候補（評価3～7）の基準複勝率を作るために使用します。
# 3-5-7は表に出ていないため0回として扱います。
TRIO_FULL_BASE_COUNTS = {
    "1-2-3": 101,
    "1-2-4": 88,
    "1-2-5": 57,
    "1-3-5": 48,
    "1-2-6": 44,
    "1-3-4": 43,
    "1-2-7": 30,
    "1-4-5": 29,
    "1-3-6": 24,
    "2-4-5": 23,
    "2-3-4": 21,
    "2-3-5": 20,
    "1-4-7": 19,
    "1-5-6": 19,
    "1-4-6": 19,
    "3-4-5": 16,
    "2-3-6": 13,
    "2-4-6": 13,
    "1-6-7": 11,
    "2-5-6": 11,
    "1-3-7": 11,
    "2-3-7": 10,
    "3-4-7": 9,
    "3-4-6": 9,
    "4-5-7": 8,
    "2-4-7": 8,
    "1-5-7": 7,
    "2-5-7": 7,
    "2-6-7": 6,
    "3-5-6": 5,
    "3-5-7": 0,
    "4-6-7": 2,
    "3-6-7": 2,
    "4-5-6": 2,
    "5-6-7": 1,
}
# 小倉2年分の3連複集計母数。
# 2車複基準と揃えるため738Rを基準にします。
TRIO_BASE_TOTAL_RACES = 738

TRIO_BASE_PLACE_COUNTS = {r: 0 for r in range(1, FIELD_SIZE + 1)}
for _trio_key, _cnt in TRIO_FULL_BASE_COUNTS.items():
    for _r in [int(x) for x in _trio_key.split("-")]:
        TRIO_BASE_PLACE_COUNTS[_r] += int(_cnt)

# 小倉2年分3連複全集計から逆算した評価別基準複勝率。
# 3連複1-2個別表では、1・2は軸として固定済みなので、評価3～7だけ補正に使います。
TRIO_BASE_PLACE_RATES = {
    r: round(100.0 * cnt / TRIO_BASE_TOTAL_RACES, 1)
    for r, cnt in TRIO_BASE_PLACE_COUNTS.items()
}


def place_state(diff) -> str:
    """評価別3着内率の基準差を状態表示する。"""
    if diff is None:
        return ""
    try:
        d = float(diff)
    except Exception:
        return ""
    if d >= 5.0:
        return "来すぎ"
    if d <= -5.0:
        return "基準未満"
    return "中庸"

TRIO_12_ALL_EXPECTED_AVG_PAY = round(
    sum(TRIO_12_ALL_BASE_COUNTS[k] * TRIO_12_ALL_BASE_AVG_PAYS[k] for k in TRIO_12_ALL_BASE_COUNTS)
    / max(1, sum(TRIO_12_ALL_BASE_COUNTS.values())),
    1,
)

# 3連複は実データの回数から想定率を置く。
# 以前の「2車複1-2×3」は理論近似だが、実測の3連複基準では高く出すぎるため使わない。
TRIO_12_INDIVIDUAL_EXPECTED_HIT_RATES = {
    k: round(100.0 * v / TRIO_BASE_TOTAL_RACES, 1)
    for k, v in TRIO_12_ALL_BASE_COUNTS.items()
}
TRIO_12_ALL_EXPECTED_HIT_RATE = round(
    100.0 * sum(TRIO_12_ALL_BASE_COUNTS.values()) / TRIO_BASE_TOTAL_RACES,
    1,
)
TRIO_12_ALL_EXPECTED_ROI = round((TRIO_12_ALL_EXPECTED_HIT_RATE / 100.0) * TRIO_12_ALL_EXPECTED_AVG_PAY / 500.0 * 100.0, 1)
TRIO_12_INDIVIDUAL_EXPECTED_ROIS = {
    k: round(TRIO_12_INDIVIDUAL_EXPECTED_HIT_RATES[k] * TRIO_12_ALL_BASE_AVG_PAYS[k] / 100.0, 1)
    for k in TRIO_12_ALL_BASE_COUNTS
}


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


def ksum_sanrenpuku_12_all(field_n: int) -> int:
    """3連複：評価1-評価2-全。7車なら5点、6車なら4点、5車なら3点。"""
    try:
        return max(0, int(field_n) - 2)
    except Exception:
        return 0


def hit_sanrenpuku_12_all(vorder: List[str], finish: List[str], field_n: int) -> bool:
    """3連複 1-2-全：評価1と評価2がともに3着内なら的中。"""
    if ksum_sanrenpuku_12_all(field_n) <= 0:
        return False
    if not vorder or len(finish) < 3:
        return False
    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    finish_ranks = {car_to_rank.get(car) for car in finish[:3]}
    return 1 in finish_ranks and 2 in finish_ranks




def sanrenpuku12_key(a: int, b: int, c: int) -> str:
    vals = sorted([int(a), int(b), int(c)])
    return f"{vals[0]}-{vals[1]}-{vals[2]}"


def ksum_sanrenpuku_12_individual(target: int, field_n: int) -> int:
    """3連複：評価1-評価2-target の1点。"""
    try:
        target = int(target)
        field_n = int(field_n)
    except Exception:
        return 0
    if field_n < 3:
        return 0
    if target in (1, 2):
        return 0
    if target > field_n:
        return 0
    return 1


def hit_sanrenpuku_12_individual(target: int, vorder: List[str], finish: List[str], field_n: int) -> bool:
    """3連複：評価1・評価2・target が3着内にそろえば的中。"""
    if ksum_sanrenpuku_12_individual(target, field_n) <= 0:
        return False
    if not vorder or len(finish) < 3:
        return False
    car_to_rank = {car: i + 1 for i, car in enumerate(vorder)}
    finish_ranks = {car_to_rank.get(car) for car in finish[:3]}
    return {1, 2, int(target)}.issubset(finish_ranks)


def sanrenpuku12_individual_row(label: str, rec: Dict[str, int], key: str) -> Dict:
    """3連複1-2個別用の表示行。2車複表と同じように想定差・回収差を見る。"""
    row = payout_row(label, rec)
    exp_rate = TRIO_12_INDIVIDUAL_EXPECTED_HIT_RATES.get(key)
    exp_avg_pay = TRIO_12_ALL_BASE_AVG_PAYS.get(key)
    exp_roi = TRIO_12_INDIVIDUAL_EXPECTED_ROIS.get(key)

    row["目"] = key
    row["想定的中率%"] = exp_rate
    row["基準平均配当"] = exp_avg_pay
    row["想定回収率%"] = exp_roi

    hit_rate = row.get("的中率%")
    avg_pay = row.get("平均配当")
    roi = row.get("回収率%")

    if hit_rate is not None and pd.notna(hit_rate) and exp_rate is not None:
        row["想定差"] = round(float(hit_rate) - float(exp_rate), 1)
    else:
        row["想定差"] = None
    row["状態"] = diff_status(row.get("想定差"), exp_rate)

    if avg_pay is not None and pd.notna(avg_pay) and exp_avg_pay:
        row["配当係数"] = round(float(avg_pay) / float(exp_avg_pay), 2)
        row["平均配当差"] = round(float(avg_pay) - float(exp_avg_pay), 1)
        if row["配当係数"] < 0.80:
            row["配当位置"] = "安すぎ"
            row["配当戻り余地"] = "あり"
        elif row["配当係数"] <= 1.30:
            row["配当位置"] = "基準付近"
            row["配当戻り余地"] = "中庸"
        else:
            row["配当位置"] = "高すぎ"
            row["配当戻り余地"] = "上振れ警戒"
    else:
        row["配当係数"] = None
        row["平均配当差"] = None
        row["配当位置"] = "未回収"
        row["配当戻り余地"] = ""

    if roi is not None and pd.notna(roi) and exp_roi is not None:
        row["回収差"] = round(float(roi) - float(exp_roi), 1)
    else:
        row["回収差"] = None

    # 2車複と同じ考え方で、後追い・過熱・未回収を避ける。
    h = float(row.get("的中H") or 0)
    diff = row.get("想定差")
    coef = row.get("配当係数")
    roi_diff = row.get("回収差")
    if h <= 0:
        reason = "未回収除外"
    elif diff is not None and diff >= 5.0:
        reason = "的中率過熱除外"
    elif roi_diff is not None and roi_diff >= 25.0:
        reason = "後追い除外"
    elif coef is not None and coef > 1.30:
        reason = "配当上振れ警戒"
    elif coef is not None and coef < 0.80:
        reason = "歪み枠"
    elif diff is not None and abs(float(diff)) <= 3.0:
        reason = "中庸枠"
    else:
        reason = "監視"
    row["総合候補理由"] = reason
    row["判定"] = ""
    return row

def sanrenpuku12_expected_rate(pair12_rate: float | None = None) -> float | None:
    """2車複1-2想定率×3で、1-2両方3着内率を概算。上限は100%。"""
    try:
        if pair12_rate is None:
            return float(TRIO_12_ALL_EXPECTED_HIT_RATE)
        return round(min(100.0, float(pair12_rate) * 3.0), 1)
    except Exception:
        return None


def sanrenpuku12_row(label: str, rec: Dict[str, int]) -> Dict:
    """3連複1-2-全用の表示行。小倉2年分3連複集計を固定想定として併記。"""
    row = payout_row(label, rec)
    exp_rate = sanrenpuku12_expected_rate()
    exp_avg_pay = float(TRIO_12_ALL_EXPECTED_AVG_PAY)
    exp_roi = float(TRIO_12_ALL_EXPECTED_ROI)

    row["想定1-2両方3着内率%"] = exp_rate
    row["想定平均配当"] = exp_avg_pay
    row["ゾーン想定回収率%"] = exp_roi

    if exp_rate and exp_rate > 0:
        row["7車5点_損益分岐平均配当"] = round((5.0 * 100.0) / (exp_rate / 100.0), 1)
    else:
        row["7車5点_損益分岐平均配当"] = None

    avg_pay = row.get("平均配当")
    roi = row.get("回収率%")
    hit_rate = row.get("的中率%")

    if avg_pay is not None and pd.notna(avg_pay):
        row["平均配当差"] = round(float(avg_pay) - exp_avg_pay, 1)
    else:
        row["平均配当差"] = None

    if roi is not None and pd.notna(roi):
        row["ゾーン回収差"] = round(float(roi) - exp_roi, 1)
    else:
        row["ゾーン回収差"] = None

    if hit_rate is not None and pd.notna(hit_rate) and exp_rate is not None:
        row["的中率差"] = round(float(hit_rate) - float(exp_rate), 1)
    else:
        row["的中率差"] = None

    return row


def nishafuku12_base_row(label: str, rec: Dict[str, int]) -> Dict:
    """2車複1-2そのものの基礎集計。1-2ゾーンの土台確認用。"""
    row = payout_row(label, rec)

    exp_rate = float(PAIR_BASE_HIT_RATE_DEFAULTS.get("1-2", 0.0))
    exp_avg_pay = float(PAIR_BASE_AVG_PAY_DEFAULTS.get("1-2", 0.0))
    exp_roi = round(exp_rate * exp_avg_pay / 100.0, 1) if exp_avg_pay > 0 else None

    row["想定的中率%"] = round(exp_rate, 1)
    row["基準平均配当"] = round(exp_avg_pay, 1)
    row["想定回収率%"] = exp_roi

    hit_rate = row.get("的中率%")
    avg_pay = row.get("平均配当")
    roi = row.get("回収率%")

    if hit_rate is not None and pd.notna(hit_rate):
        row["的中率差"] = round(float(hit_rate) - exp_rate, 1)
    else:
        row["的中率差"] = None

    if avg_pay is not None and pd.notna(avg_pay):
        row["平均配当差"] = round(float(avg_pay) - exp_avg_pay, 1)
    else:
        row["平均配当差"] = None

    if roi is not None and pd.notna(roi) and exp_roi is not None:
        row["回収差"] = round(float(roi) - float(exp_roi), 1)
    else:
        row["回収差"] = None

    return row


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
        "的中H": H,
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
        row["想定差"] = round(row["的中率%"] - expected_hit, 1)
    else:
        row["想定差"] = None

    return row


def expected_pair_hit_rate_from_pair12(a: int, b: int, pair12_counts: Dict[PairKey, int]) -> float | None:
    """1→2着評価分布から、評価a-bの2車複想定的中率を出す。順不同。"""
    total = sum(int(v) for v in pair12_counts.values())
    if total <= 0:
        return None
    hit = int(pair12_counts.get((a, b), 0)) + int(pair12_counts.get((b, a), 0))
    return round(100.0 * hit / total, 1)


def diff_status(diff, expected=None) -> str:
    """想定差の状態をざっくり表示。想定0%は候補対象外。"""
    if expected is not None and expected == 0:
        return "対象外"
    if diff is None:
        return ""
    if diff >= 10:
        return "当たりすぎ"
    if diff <= -10:
        return "当たらなすぎ"
    return "中庸"


def _clean_num_list(values):
    out = []
    for v in values:
        try:
            if pd.notna(v):
                out.append(float(v))
        except Exception:
            pass
    return out


def _median(values):
    vals = sorted(_clean_num_list(values))
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _deviation_stats(value, values):
    """平均差・中央値差・偏差値・基準位置を返す。少数候補でも落ちない軽量版。"""
    try:
        if value is None or pd.isna(value):
            return {"平均差": None, "中央値差": None, "偏差値": None, "基準位置": ""}
        x = float(value)
    except Exception:
        return {"平均差": None, "中央値差": None, "偏差値": None, "基準位置": ""}

    vals = _clean_num_list(values)
    if not vals:
        return {"平均差": None, "中央値差": None, "偏差値": None, "基準位置": ""}

    mean = sum(vals) / len(vals)
    med = _median(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 0 else 0.0
    sd = var ** 0.5

    mean_diff = round(x - mean, 1)
    med_diff = round(x - med, 1) if med is not None else None
    if sd > 0:
        dev = round(50.0 + 10.0 * (x - mean) / sd, 1)
    else:
        dev = 50.0

    if dev >= 60:
        pos = "高すぎ"
    elif dev >= 55:
        pos = "やや高い"
    elif dev <= 40:
        pos = "低すぎ"
    elif dev <= 45:
        pos = "やや低い"
    else:
        pos = "中庸"

    return {
        "平均差": mean_diff,
        "中央値差": med_diff,
        "偏差値": dev,
        "基準位置": pos,
    }


def render_sortable_table(df: pd.DataFrame, height: int = 470):
    """Streamlit標準のソート可能表。判定・型を先頭に寄せ、横スクロールなしで見やすくする。"""
    if df is None or df.empty:
        st.info("表示するデータがありません。")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
    )



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

# 前日まで：3連複 1-2-全（仮想全体）
agg_payout_sanrenpuku12_all_manual: Dict[str, Dict[str, int]] = {
    "仮想全体": new_payout_rec(),
}

# 前日まで：3連複 1-2 個別（1-2-3～1-2-7）
agg_payout_sanrenpuku12_individual_manual: Dict[str, Dict[str, Dict[str, int]]] = {
    "仮想全体": {k: new_payout_rec() for k in TRIO_12_ALL_BASE_COUNTS},
}



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
        cols_hdr = st.columns([0.8, 0.9, 2.8, 1.05, 1.0, 1.0])
        cols_hdr[0].markdown("**R**")
        cols_hdr[1].markdown("**頭数**")
        cols_hdr[2].markdown("**V評価（頭数ぶんの桁数）**")
        cols_hdr[3].markdown("**着順(～3桁)**")
        cols_hdr[4].markdown("**2車複**")
        cols_hdr[5].markdown("**3連複**")

        daily_inputs = []

        for i in range(1, 37):
            c1, c2, c3, c4, c5, c6 = st.columns([0.8, 0.9, 2.8, 1.05, 1.0, 1.0])

            rid = c1.text_input("", key=f"rid_{i}", value=str(i))
            field_n = c2.selectbox("", options=[7, 6, 5], index=0, key=f"field_n_{i}")
            vline = c3.text_input("", key=f"vline_{i}", value="")
            fin = c4.text_input("", key=f"fin_{i}", value="")
            pay_2f = c5.number_input("", key=f"pay2f_{i}", min_value=0, value=0, step=10)
            pay_3f = c6.number_input("", key=f"pay3f_{i}", min_value=0, value=0, step=10)
            pay_2t = 0

            daily_inputs.append(
                {
                    "rid": rid,
                    "field_n": field_n,
                    "vline": vline,
                    "fin": fin,
                    "pay_2t": pay_2t,
                    "pay_2f": pay_2f,
                    "pay_3f": pay_3f,
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
        pay_3f = int(item.get("pay_3f", 0))
        vorder = parse_rankline(vline, field_n)
        finish = parse_finish(fin)

        any_input = any([vline.strip(), fin.strip(), pay_2f > 0, pay_3f > 0])
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
                    "pay_3f": pay_3f,
                }
            )

    st.divider()
    st.markdown("### 入力後操作")
    components.html(
        """
        <div style="padding: 6px 0 12px 0;">
          <button
            onclick="window.parent.scrollTo({top: 0, behavior: 'smooth'});"
            style="
              font-size: 16px;
              font-weight: 700;
              padding: 10px 18px;
              border-radius: 10px;
              border: 1px solid #c9c9c9;
              background: #ffffff;
              cursor: pointer;
              box-shadow: 0 1px 4px rgba(0,0,0,0.12);
            "
          >
            ページトップに戻る
          </button>
        </div>
        """,
        height=64,
    )
    st.caption("入力後はこのボタンでページ上部へ戻り、上部タブから『前日までの集計（累積）』『分析結果』へ移動してください。")


# =========================
# B. 前日までの集計（累積）
# =========================
with tabs[1]:
    st.markdown('<a id="prev-aggregate"></a>', unsafe_allow_html=True)
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
        # 必要な確認は下の「個別2車複 引継ぎ入力」で行う。
        payout_inputs = []

        st.divider()

        st.markdown("## 個別2車複 引継ぎ入力（累積）")
        st.caption(
            "分析結果の『個別2車複 引継ぎ用累積表』をそのまま転記します。"
            "対象N・払戻合計SUM・的中Hだけ入力。KSUMは対象Nと同じ扱いで自動計算します。"
            " 1-7・2-7も低頻度確認用として残しますが、未回収なら候補には入りません。"
        )

        nishafuku_pair_inputs = []

        def _pair_input_block(title: str, pairs: List[Tuple[int, int]]):
            st.markdown(f"**{title}**")
            h_pair = st.columns([1.35, 0.85, 1.05, 0.75])
            h_pair[0].markdown("**型**")
            h_pair[1].markdown("**N**")
            h_pair[2].markdown("**SUM**")
            h_pair[3].markdown("**H**")

            for a, b in pairs:
                label = nishafuku_label(a, b)
                safe_key = label.replace(" ", "_").replace("-", "_")
                c0, c1, c2, c3 = st.columns([1.35, 0.85, 1.05, 0.75])
                c0.write(label.replace("2車複 ", ""))
                N = c1.number_input("", key=f"prev_pair_{safe_key}_N", min_value=0, value=0, label_visibility="collapsed")
                SUM = c2.number_input("", key=f"prev_pair_{safe_key}_SUM", min_value=0, value=0, step=10, label_visibility="collapsed")
                H = c3.number_input("", key=f"prev_pair_{safe_key}_H", min_value=0, value=0, label_visibility="collapsed")
                nishafuku_pair_inputs.append((label, int(N), int(SUM), int(H)))

        left_pairs = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
        right_pairs = [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
        col_left, col_right = st.columns(2)
        with col_left:
            _pair_input_block("評価1軸", left_pairs)
        with col_right:
            _pair_input_block("評価2軸", right_pairs)

        st.divider()

        st.markdown("## 3連複 1-2-全 引継ぎ入力（累積）")
        st.caption("全レースで1-2-全を買った仮想集計です。7車は1R=5点なので、KSUMはN×5で自動計算します。")
        sanrenpuku12_inputs = []
        h_sp = st.columns([1.4, 0.9, 1.1, 0.8])
        h_sp[0].markdown("**区分**")
        h_sp[1].markdown("**N**")
        h_sp[2].markdown("**SUM**")
        h_sp[3].markdown("**H**")
        for label in ["仮想全体"]:
            safe = label.replace(" ", "_")
            c0, c1, c2, c3 = st.columns([1.4, 0.9, 1.1, 0.8])
            c0.write(label)
            N = c1.number_input("", key=f"prev_sp12_{safe}_N", min_value=0, value=0, label_visibility="collapsed")
            SUM = c2.number_input("", key=f"prev_sp12_{safe}_SUM", min_value=0, value=0, step=10, label_visibility="collapsed")
            H = c3.number_input("", key=f"prev_sp12_{safe}_H", min_value=0, value=0, label_visibility="collapsed")
            sanrenpuku12_inputs.append((label, int(N), int(SUM), int(H)))

        st.markdown("## 3連複 1-2 個別 引継ぎ入力（累積）")
        st.caption("1-2-3～1-2-7を個別に転記します。2車複と同じように、対象N・払戻合計SUM・的中Hだけ入力。KSUMは対象Nと同じです。")
        sanrenpuku12_individual_inputs = []
        for label in ["仮想全体"]:
            st.markdown(f"**{label}**")
            h_tri = st.columns([1.2, 0.85, 1.05, 0.75])
            h_tri[0].markdown("**目**")
            h_tri[1].markdown("**N**")
            h_tri[2].markdown("**SUM**")
            h_tri[3].markdown("**H**")
            for key in TRIO_12_ALL_BASE_COUNTS:
                safe_label = label.replace(" ", "_")
                safe_key = key.replace("-", "_")
                c0, c1, c2, c3 = st.columns([1.2, 0.85, 1.05, 0.75])
                c0.write(key)
                N = c1.number_input("", key=f"prev_tri12ind_{safe_label}_{safe_key}_N", min_value=0, value=0, label_visibility="collapsed")
                SUM = c2.number_input("", key=f"prev_tri12ind_{safe_label}_{safe_key}_SUM", min_value=0, value=0, step=10, label_visibility="collapsed")
                H = c3.number_input("", key=f"prev_tri12ind_{safe_label}_{safe_key}_H", min_value=0, value=0, label_visibility="collapsed")
                sanrenpuku12_individual_inputs.append((label, key, int(N), int(SUM), int(H)))

        st.divider()

        nishafuku_set_inputs = []

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

    for label, N, SUM, H in nishafuku_pair_inputs:
        if any([N, SUM, H]) and label in agg_payout_nishafuku_manual:
            rec = agg_payout_nishafuku_manual[label]
            rec["N"] += int(N)
            rec["KSUM"] += int(N)
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)

    for label, N, SUM, H in sanrenpuku12_inputs:
        if any([N, SUM, H]) and label in agg_payout_sanrenpuku12_all_manual:
            rec = agg_payout_sanrenpuku12_all_manual[label]
            rec["N"] += int(N)
            rec["KSUM"] += int(N) * 5
            rec["SUM"] += int(SUM)
            rec["H"] += int(H)

    for label, key, N, SUM, H in sanrenpuku12_individual_inputs:
        if any([N, SUM, H]) and label in agg_payout_sanrenpuku12_individual_manual and key in agg_payout_sanrenpuku12_individual_manual[label]:
            rec = agg_payout_sanrenpuku12_individual_manual[label][key]
            rec["N"] += int(N)
            rec["KSUM"] += int(N)
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






# --- 3連複 1-2-全（日次） ---
payout_sanrenpuku12_all_daily: Dict[str, Dict[str, int]] = {
    "仮想全体": new_payout_rec(),
}
payout_sanrenpuku12_individual_daily: Dict[str, Dict[str, Dict[str, int]]] = {
    "仮想全体": {k: new_payout_rec() for k in TRIO_12_ALL_BASE_COUNTS},
}

for row in byrace_rows:
    vorder = row.get("vorder", [])
    finish = row.get("finish", [])
    field_n = int(row.get("field_n", len(vorder) or 0))

    if not vorder or field_n <= 0 or len(finish) < 3:
        continue

    ksum = ksum_sanrenpuku_12_all(field_n)
    if ksum <= 0:
        continue

    pay_3f = int(row.get("pay_3f", 0))
    is_hit = hit_sanrenpuku_12_all(vorder, finish, field_n)

    rec_all = payout_sanrenpuku12_all_daily["仮想全体"]
    rec_all["N"] += 1
    rec_all["KSUM"] += ksum
    if is_hit:
        rec_all["H"] += 1
        rec_all["SUM"] += pay_3f

    # 個別 1-2-3～1-2-7。全体は存在する評価だけ毎回1点仮想購入。
    for target in range(3, 8):
        key = sanrenpuku12_key(1, 2, target)
        one_ksum = ksum_sanrenpuku_12_individual(target, field_n)
        if one_ksum <= 0 or key not in TRIO_12_ALL_BASE_COUNTS:
            continue
        one_hit = hit_sanrenpuku_12_individual(target, vorder, finish, field_n)

        rec_ind_all = payout_sanrenpuku12_individual_daily["仮想全体"][key]
        rec_ind_all["N"] += 1
        rec_ind_all["KSUM"] += one_ksum
        if one_hit:
            rec_ind_all["H"] += 1
            rec_ind_all["SUM"] += pay_3f


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


payout_sanrenpuku12_all_total: Dict[str, Dict[str, int]] = {
    "仮想全体": new_payout_rec(),
}
for label in payout_sanrenpuku12_all_total.keys():
    add_rec(payout_sanrenpuku12_all_total[label], payout_sanrenpuku12_all_daily[label])
    add_rec(payout_sanrenpuku12_all_total[label], agg_payout_sanrenpuku12_all_manual[label])

payout_sanrenpuku12_individual_total: Dict[str, Dict[str, Dict[str, int]]] = {
    "仮想全体": {k: new_payout_rec() for k in TRIO_12_ALL_BASE_COUNTS},
}
for label in payout_sanrenpuku12_individual_total.keys():
    for key in TRIO_12_ALL_BASE_COUNTS:
        add_rec(payout_sanrenpuku12_individual_total[label][key], payout_sanrenpuku12_individual_daily[label][key])
        add_rec(payout_sanrenpuku12_individual_total[label][key], agg_payout_sanrenpuku12_individual_manual[label][key])


# =========================
# 出力：分析結果
# =========================
with tabs[2]:
    st.markdown('<a id="analysis-result"></a>', unsafe_allow_html=True)
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
    st.divider()

    st.markdown("### 個別2車複候補｜評価1・2軸 合成ペア比較")
    st.caption("標準棚/穴棚シミュレーターと波履歴は削除。個別ペアの偏差値・配当係数・未回収除外・配当戻り余地で3〜4点を選びます。1-2は低配当安定枠として別判定します。")

    st.markdown("#### 配当収束シミュレーション設定")
    st.caption("平均配当はペア別基準配当で判定します。想定ペア的%は小倉ミッドナイトA級7車・直近2年の固定的中率を使用します。")
    c_pay1, c_pay2, c_pay3 = st.columns([1, 1, 3])
    OVERHEAT_Z = c_pay1.number_input(
        "的中過熱偏差値",
        key="overheat_z_line",
        min_value=50.0,
        max_value=80.0,
        value=60.0,
        step=1.0,
        format="%.1f",
    )
    PAY_OVERHEAT_Z = c_pay2.number_input(
        "配当過熱偏差値",
        key="pay_overheat_z_line",
        min_value=50.0,
        max_value=90.0,
        value=60.0,
        step=1.0,
        format="%.1f",
    )
    c_pay3.write("ペアごとに基準配当を変え、さらに配当偏差値が高すぎるペアは過熱として候補から外します。")

    with st.expander("ペア別基準配当を確認・調整", expanded=False):
        st.caption("初期値：小倉ミッドナイトA級7車・直近2年。会場や条件を変える場合はここを上書きしてください。")
        pair_base_avg_pay = {}
        base_pairs = [
            ("1-2", 271), ("1-3", 436), ("1-4", 654),
            ("1-5", 1059), ("1-6", 1754), ("1-7", 1519),
            ("2-3", 881), ("2-4", 1333), ("2-5", 1869),
            ("2-6", 1657), ("2-7", 4092),
        ]
        for i in range(0, len(base_pairs), 4):
            cols = st.columns(4)
            for col, (pair_key, default_pay) in zip(cols, base_pairs[i:i+4]):
                pair_base_avg_pay[pair_key] = col.number_input(
                    pair_key,
                    key=f"pair_base_avg_pay_{pair_key.replace('-', '_')}",
                    min_value=100,
                    value=int(PAIR_BASE_AVG_PAY_DEFAULTS.get(pair_key, default_pay)),
                    step=10,
                )

    st.markdown("### 最終2車複候補｜評価1・2軸")
    st.caption("評価1軸・評価2軸の個別2車複を、安定枠・中庸枠・歪み枠に分けて比較します。本線は2点固定。歪み枠は本線に入れず、注として監視表示します。")
    st.caption("想定ペア的%は、現在の入力データではなく小倉ミッドA級7車・直近2年の固定値を強制使用します。例：1-2=22.6、2-4=5.7、2-7=3.0。")

    c_final1, c_final2, c_final3, c_final4, c_final5 = st.columns([1, 1, 1, 1, 2])
    FINAL_POINT_N = c_final1.number_input(
        "本線点数",
        key="final_point_n_axis12",
        min_value=2,
        max_value=2,
        value=2,
        step=1,
        help="実戦で買う柱の点数。現在は2点固定です。",
    )
    MIN_EXPECTED_PAIR_RATE = c_final2.number_input(
        "最低想定ペア的%",
        key="min_expected_pair_rate_axis12",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=0.5,
        format="%.1f",
    )
    NOTE_MAX_N = c_final3.number_input(
        "注表示最大",
        key="note_max_n_axis12",
        min_value=0,
        max_value=6,
        value=3,
        step=1,
        help="歪み枠を注として表示する最大数。予算が増えた時の追加候補です。",
    )
    ROI_DIFF_LIMIT = c_final4.number_input(
        "回収差許容",
        key="roi_diff_limit_axis12",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        format="%.1f",
        help="実回収率−想定回収率がこの値を超えたペアは後追い扱いで候補から外します。",
    )
    PAY_RETURN_ONLY = c_final5.checkbox(
        "未回収除外＋配当戻り優先",
        key="pay_return_only_axis12",
        value=True,
        help="ONの場合、未回収ペアを除外し、配当安すぎ〜基準付近を優先。ただし回収率180%以上・的中偏差値/配当偏差値が過熱ライン以上は除外します。",
    )
    st.caption("評価1・評価2を両方候補化。未回収・過熱・想定回収率超えすぎを除外し、本線は安定枠＋中庸枠を優先して2点まで。歪み枠は候補欄ではなく『注』欄に表示します。")

    pair_rows = []
    for axis in [1, 2]:
        for opp in range(1, 8):
            if opp == axis:
                continue
            # 2車複は順不同なので、評価2軸の相手1（2-1）は評価1-2と重複。
            # 表示・候補選定ともに1-2側だけを残す。
            if axis == 2 and opp == 1:
                continue

            a, b = sorted((axis, opp))
            label = nishafuku_label(a, b)
            rec = payout_nishafuku_total.get(label, new_payout_rec())
            row = payout_row(label, rec)

            pair_key = f"{a}-{b}"
            # 想定ペア的%は現在入力データから計算しない。
            # 小倉ミッドA級7車・直近2年の固定値をそのまま使用する。
            expected_pair = PAIR_BASE_HIT_RATE_DEFAULTS.get(pair_key)
            expected_pair = round(float(expected_pair), 1) if expected_pair is not None else None
            row["軸"] = f"評価{axis}"
            row["軸番号"] = axis
            row["相手"] = opp
            row["ペアキー"] = pair_key
            row["想定ペア的%"] = expected_pair

            if row["的中率%"] is not None and expected_pair is not None:
                diff = round(float(row["的中率%"] ) - float(expected_pair), 1)
            else:
                diff = None

            row["想定差"] = diff
            row["状態"] = diff_status(diff, expected_pair)

            pair_base_pay = int(pair_base_avg_pay.get(f"{a}-{b}", PAIR_BASE_AVG_PAY_DEFAULTS.get(f"{a}-{b}", 1200)))
            row["ペア基準配当"] = pair_base_pay

            if expected_pair is not None and pair_base_pay > 0:
                row["想定回収率%"] = round(float(expected_pair) * float(pair_base_pay) / 100.0, 1)
            else:
                row["想定回収率%"] = None

            actual_roi = row.get("回収率%")
            if actual_roi is not None and pd.notna(actual_roi) and row["想定回収率%"] is not None:
                row["回収差"] = round(float(actual_roi) - float(row["想定回収率%"]), 1)
            else:
                row["回収差"] = None

            avg_pay = row.get("平均配当")
            if avg_pay is not None and pd.notna(avg_pay) and pair_base_pay > 0:
                row["配当係数"] = round(float(avg_pay) / float(pair_base_pay), 2)
                row["配当差"] = round(float(avg_pay) - float(pair_base_pay), 1)
            else:
                row["配当係数"] = None
                row["配当差"] = None

            row["配当位置"] = ""
            row["配当戻り余地"] = ""
            row["総合候補理由"] = ""
            row["判定"] = ""
            pair_rows.append(row)

    df_pairs = pd.DataFrame(pair_rows)

    if not df_pairs.empty and df_pairs["想定差"].notna().any():
        candidate_mask = (
            df_pairs["想定差"].notna()
            & df_pairs["想定ペア的%"].notna()
            & (df_pairs["想定ペア的%"] > float(MIN_EXPECTED_PAIR_RATE))
        )

        # 評価1・2の全候補を同一母集団として、平均差・中央値差・偏差値を出す。
        diff_values = df_pairs.loc[candidate_mask, "想定差"].tolist() if candidate_mask.any() else []
        # 配当偏差値は「平均配当そのもの」ではなく、
        # ペア基準配当との差（配当差）を母集団にして計算する。
        # 例：1-2の平均348円・基準271円なら配当差+77円。
        pay_values = df_pairs.loc[candidate_mask & df_pairs["配当差"].notna(), "配当差"].tolist() if candidate_mask.any() else []

        for idx in df_pairs.index:
            stats = _deviation_stats(df_pairs.loc[idx, "想定差"], diff_values)
            df_pairs.loc[idx, "平均差"] = stats.get("平均差")
            df_pairs.loc[idx, "中央値差"] = stats.get("中央値差")
            df_pairs.loc[idx, "偏差値"] = stats.get("偏差値")
            df_pairs.loc[idx, "基準位置"] = stats.get("基準位置")

            pay_stats = _deviation_stats(df_pairs.loc[idx, "配当差"], pay_values)
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
            expected_median = _median(df_pairs.loc[candidate_mask, "想定ペア的%"].tolist())
            df_pairs["_expected_ok"] = df_pairs["想定ペア的%"].apply(
                lambda x: bool(pd.notna(x) and expected_median is not None and float(x) >= float(expected_median))
            )
            df_pairs["_below_base"] = (
                (df_pairs["中央値差"].notna() & (df_pairs["中央値差"] < 0))
                | (df_pairs["偏差値"].notna() & (df_pairs["偏差値"] < 50))
            )
            df_pairs["_overhit"] = df_pairs["偏差値"].notna() & (df_pairs["偏差値"] >= float(OVERHEAT_Z))
            df_pairs["_pay_dev_overheat"] = df_pairs["配当偏差値"].notna() & (df_pairs["配当偏差値"] >= float(PAY_OVERHEAT_Z))
            df_pairs["_cold"] = df_pairs["偏差値"].notna() & (df_pairs["偏差値"] <= 40)
            df_pairs["_pay_low"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] < 0.80)
            df_pairs["_pay_near"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] >= 0.80) & (df_pairs["配当係数"] <= 1.30)
            df_pairs["_pay_high"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"] > 1.30)

            # 最終候補対象：
            # H=0（未回収）は反発点が読めないため除外。
            # 回収率180%以上、的中率偏差値・配当偏差値が過熱ライン以上、または実回収率が想定回収率を上回りすぎた場合は過熱として原則除外。
            # その上で、買い目を「安定枠／中庸枠／歪み枠」に分けて選ぶ。
            df_pairs["_has_hit"] = df_pairs["的中H"].fillna(0).astype(float) > 0
            df_pairs["_roi_overheat"] = df_pairs["回収率%"].notna() & (df_pairs["回収率%"].astype(float) >= 180.0)
            df_pairs["_roi_follow_over"] = df_pairs["回収差"].notna() & (df_pairs["回収差"].astype(float) > float(ROI_DIFF_LIMIT))
            df_pairs["_not_overheated"] = ~df_pairs["_overhit"] & ~df_pairs["_roi_overheat"] & ~df_pairs["_pay_dev_overheat"] & ~df_pairs["_roi_follow_over"]
            df_pairs["_coef_core"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"].astype(float) >= 0.50) & (df_pairs["配当係数"].astype(float) <= 1.20)
            df_pairs["_coef_too_low"] = df_pairs["配当係数"].notna() & (df_pairs["配当係数"].astype(float) < 0.50)

            # 安定枠：1-2のような低配当・安定ペアを土台にする。
            # ペア基準比では多少高くても、絶対配当が安く、偏差値・回収率が過熱していなければ候補に残す。
            df_pairs["_stable_lowpay_12"] = (
                (df_pairs["ペアキー"] == "1-2")
                & df_pairs["_has_hit"]
                & df_pairs["偏差値"].notna()
                & (df_pairs["偏差値"].astype(float) >= 45.0)
                & (df_pairs["偏差値"].astype(float) <= 55.0)
                & df_pairs["回収率%"].notna()
                & (df_pairs["回収率%"].astype(float) >= 70.0)
                & (df_pairs["回収率%"].astype(float) <= 130.0)
                & df_pairs["平均配当"].notna()
                & (df_pairs["平均配当"].astype(float) >= 300.0)
                & (df_pairs["平均配当"].astype(float) <= 700.0)
            )

            # 中庸枠：主力株枠。候補が2点以上あるなら歪み枠より優先する。
            # 条件は「過熱なし・的中実績あり・配当係数が基準付近・偏差値が中庸〜やや高め・回収率が暴れていない」。
            df_pairs["_middle_core"] = (
                candidate_mask
                & df_pairs["_has_hit"]
                & df_pairs["_not_overheated"]
                & df_pairs["配当係数"].notna()
                & (df_pairs["配当係数"].astype(float) >= 0.80)
                & (df_pairs["配当係数"].astype(float) <= 1.30)
                & df_pairs["偏差値"].notna()
                & (df_pairs["偏差値"].astype(float) >= 45.0)
                & (df_pairs["偏差値"].astype(float) <= 58.0)
                & df_pairs["回収率%"].notna()
                & (df_pairs["回収率%"].astype(float) >= 60.0)
                & (df_pairs["回収率%"].astype(float) <= 160.0)
            )

            # 歪み枠：評価番号では固定しない。条件だけで選ぶ。
            # 配当係数が低く、下振れ〜中庸で、未回収でも過熱でもないペア。
            df_pairs["_distortion_core"] = (
                candidate_mask
                & df_pairs["_has_hit"]
                & df_pairs["_not_overheated"]
                & df_pairs["配当係数"].notna()
                & (df_pairs["配当係数"].astype(float) >= 0.50)
                & (df_pairs["配当係数"].astype(float) < 0.80)
                & df_pairs["偏差値"].notna()
                & (df_pairs["偏差値"].astype(float) >= 40.0)
                & (df_pairs["偏差値"].astype(float) <= 58.0)
            )

            final_candidate_mask = candidate_mask & df_pairs["_has_hit"] & df_pairs["_not_overheated"]
            if PAY_RETURN_ONLY:
                final_candidate_mask = final_candidate_mask & (
                    df_pairs["_stable_lowpay_12"] | df_pairs["_middle_core"] | df_pairs["_distortion_core"]
                )

            df_pairs["資産枠"] = ""
            df_pairs.loc[df_pairs["_stable_lowpay_12"], "資産枠"] = "安定"
            df_pairs.loc[df_pairs["_middle_core"], "資産枠"] = "中庸"
            df_pairs.loc[df_pairs["_distortion_core"], "資産枠"] = "歪み"

            df_pairs.loc[df_pairs["_pay_low"], "配当戻り余地"] = "あり"
            df_pairs.loc[df_pairs["_pay_near"], "配当戻り余地"] = "中庸"
            df_pairs.loc[df_pairs["_pay_high"], "配当戻り余地"] = "上振れ警戒"
            df_pairs.loc[df_pairs["_stable_lowpay_12"], "配当戻り余地"] = "低配当安定"

            df_pairs.loc[~df_pairs["_has_hit"], "総合候補理由"] = "未回収除外"
            df_pairs.loc[df_pairs["_has_hit"] & df_pairs["_roi_overheat"], "総合候補理由"] = "回収率過熱除外"
            df_pairs.loc[df_pairs["_has_hit"] & df_pairs["_roi_follow_over"] & ~df_pairs["_roi_overheat"], "総合候補理由"] = "後追い除外"
            df_pairs.loc[df_pairs["_has_hit"] & df_pairs["_overhit"] & ~df_pairs["_roi_overheat"] & ~df_pairs["_roi_follow_over"], "総合候補理由"] = "的中率過熱除外"
            df_pairs.loc[df_pairs["_has_hit"] & df_pairs["_pay_dev_overheat"] & ~df_pairs["_roi_overheat"] & ~df_pairs["_overhit"] & ~df_pairs["_roi_follow_over"], "総合候補理由"] = "配当過熱除外"
            df_pairs.loc[df_pairs["_stable_lowpay_12"], "総合候補理由"] = "安定枠"
            df_pairs.loc[df_pairs["_middle_core"], "総合候補理由"] = "中庸枠"
            df_pairs.loc[df_pairs["_distortion_core"], "総合候補理由"] = "歪み枠"

            # 枠別の優先順位。
            # 本線は2点固定：安定枠＋中庸枠を柱にする。
            # 歪み枠は本線には入れず、注として監視表示する。
            df_pairs["_枠内順位"] = 999.0
            df_pairs.loc[df_pairs["_stable_lowpay_12"], "_枠内順位"] = (
                (df_pairs["偏差値"].astype(float) - 50.0).abs()
                + (df_pairs["回収率%"].astype(float) - 100.0).abs() / 20.0
            )
            df_pairs.loc[df_pairs["_middle_core"], "_枠内順位"] = (
                (df_pairs["偏差値"].astype(float) - 52.0).abs()
                + (df_pairs["配当係数"].astype(float) - 1.0).abs() * 10.0
                + (df_pairs["回収率%"].astype(float) - 100.0).abs() / 35.0
            )
            df_pairs.loc[df_pairs["_distortion_core"], "_枠内順位"] = (
                (df_pairs["配当係数"].astype(float) - 0.70).abs() * 10.0
                + (df_pairs["偏差値"].astype(float) - 50.0).abs() / 2.0
                + (df_pairs["回収率%"].astype(float) - 100.0).abs() / 50.0
            )

            def _pick_unique(source_df: pd.DataFrame, limit: int, already: set) -> list:
                if limit <= 0 or source_df.empty:
                    return []
                picked = []
                for idx, r in source_df.sort_values(["_枠内順位", "配当係数", "偏差値", "回収率%", "軸番号", "相手"]).iterrows():
                    key = str(r.get("ペアキー"))
                    if key in already:
                        continue
                    picked.append(idx)
                    already.add(key)
                    if len(picked) >= limit:
                        break
                return picked

            selected_idx = []
            note_idx = []
            used_pairs = set()
            target_n = int(FINAL_POINT_N)

            stable_df = df_pairs.loc[final_candidate_mask & df_pairs["_stable_lowpay_12"]]
            middle_df = df_pairs.loc[final_candidate_mask & df_pairs["_middle_core"]]
            distortion_df = df_pairs.loc[final_candidate_mask & df_pairs["_distortion_core"]]

            # 本線：まず安定枠を最大1点。
            selected_idx += _pick_unique(stable_df, 1, used_pairs)

            # 本線：残りは中庸枠を優先。歪み枠は本線には入れない。
            selected_idx += _pick_unique(middle_df, target_n - len(selected_idx), used_pairs)

            # それでも2点に届かない場合のみ、歪み以外の未過熱・実績あり候補で補完。
            # ここでも歪み枠は注へ回す。
            if len(selected_idx) < target_n:
                fallback_df = df_pairs.loc[
                    final_candidate_mask
                    & ~df_pairs.index.isin(selected_idx)
                    & ~df_pairs["_distortion_core"]
                ].copy()
                fallback_df["_枠内順位"] = fallback_df["_枠内順位"].fillna(999.0)
                selected_idx += _pick_unique(fallback_df, target_n - len(selected_idx), used_pairs)

            # 注：歪み枠は予算増時の追加候補として表示。複数可。
            note_used_pairs = set(used_pairs)
            note_idx += _pick_unique(distortion_df, int(NOTE_MAX_N), note_used_pairs)

            if not selected_idx and PAY_RETURN_ONLY:
                st.warning("未回収除外＋配当戻り優先では本線候補がありません。必要ならチェックを外して広め候補を確認してください。")

            df_pairs.loc[selected_idx, "判定"] = "本線"
            df_pairs.loc[note_idx, "判定"] = "注"

            recommended_pairs = []
            for idx in selected_idx:
                pair_key = str(df_pairs.loc[idx, "ペアキー"])
                recommended_pairs.append(pair_key)
            recommended_pairs = sorted(recommended_pairs, key=lambda x: tuple(int(v) for v in x.split("-")))
            recommended_text = " / ".join(recommended_pairs)

            note_pairs = []
            for idx in note_idx:
                pair_key = str(df_pairs.loc[idx, "ペアキー"])
                note_pairs.append(pair_key)
            note_pairs = sorted(note_pairs, key=lambda x: tuple(int(v) for v in x.split("-")))
            note_text = " / ".join(note_pairs)

            if recommended_text:
                st.success(f"現在の推奨2車複本線：{recommended_text}")
            if note_text:
                st.info(f"注：{note_text}")

            drop_cols = [
                "_expected_ok", "_below_base", "_overhit", "_pay_dev_overheat", "_cold",
                "_pay_low", "_pay_near", "_pay_high", "_has_hit", "_roi_overheat", "_roi_follow_over", "_coef_core", "_coef_too_low", "_not_overheated",
                "_stable_lowpay_12", "_middle_core", "_distortion_core", "_枠内順位",
            ]
            df_pairs = df_pairs.drop(columns=[c for c in drop_cols if c in df_pairs.columns])

            drop_cols = [
                "_expected_ok", "_below_base", "_overhit", "_pay_dev_overheat", "_cold",
                "_pay_low", "_pay_near", "_pay_high", "_has_hit", "_roi_overheat", "_roi_follow_over", "_coef_core", "_coef_too_low", "_not_overheated", "_候補優先",
            ]
            df_pairs = df_pairs.drop(columns=[c for c in drop_cols if c in df_pairs.columns])
        else:
            st.info("候補対象となる相手がありません。想定ペア的%が最低ライン以下の組み合わせは除外しています。")
    else:
        st.info("候補を出すには、個別2車複データが必要です。")

    preferred_pair_cols = [
        "判定",
        "型",
        "対象N",
        "的中H",
        "的中率%",
        "想定ペア的%",
        "想定差",
        "平均差",
        "中央値差",
        "偏差値",
        "基準位置",
        "状態",
        "平均配当",
        "ペア基準配当",
        "想定回収率%",
        "回収率%",
        "回収差",
        "配当係数",
        "配当差",
        "配当偏差値",
        "配当位置",
        "配当戻り余地",
        "資産枠",
        "総合候補理由",
    ]
    df_pairs = df_pairs[[c for c in preferred_pair_cols if c in df_pairs.columns]]
    render_sortable_table(df_pairs, height=470)

    st.markdown("### 1-2ゾーン基礎集計")
    st.caption("全体に対して、2車複1-2そのものがどれくらい機能しているかを確認します。3連複個別表は、この1-2ゾーンの上に乗せる買い目選別です。")

    rec_12_base = payout_nishafuku_total.get(nishafuku_label(1, 2), new_payout_rec())
    rec_12_trio_all = payout_sanrenpuku12_all_total.get("仮想全体", new_payout_rec())
    base_rows = [
        nishafuku12_base_row("2車複 1-2｜全体", rec_12_base),
        sanrenpuku12_row("3連複 1-2-全｜全体", rec_12_trio_all),
    ]
    df_12_base = pd.DataFrame(base_rows)
    base_cols = [
        "型",
        "対象N",
        "総点数KSUM",
        "投資額換算",
        "払戻合計SUM",
        "的中H",
        "的中率%",
        "想定的中率%",
        "想定1-2両方3着内率%",
        "的中率差",
        "平均配当",
        "基準平均配当",
        "想定平均配当",
        "平均配当差",
        "回収率%",
        "想定回収率%",
        "ゾーン想定回収率%",
        "回収差",
        "ゾーン回収差",
    ]
    st.dataframe(
        df_12_base[[c for c in base_cols if c in df_12_base.columns]],
        use_container_width=True,
        hide_index=True,
    )

    # 1-2の土台を、短い文章でも確認できるようにする
    try:
        n_all = int(rec_12_base.get("N", 0))
        h_2f = int(rec_12_base.get("H", 0))
        h_3f = int(rec_12_trio_all.get("H", 0))
        if n_all > 0:
            st.caption(
                f"1-2基礎：全体{n_all}R中、2車複1-2は{h_2f}R、"
                f"3連複1-2-全条件（1と2が両方3着内）は{h_3f}Rです。"
            )
    except Exception:
        pass

    st.markdown("### 3連複 1-2-全 ゾーン検証")
    st.caption("入力済み全レースで3連複1-2-全を買った仮想集計です。想定平均配当は小倉2年分3連複全集計の1-2-全加重平均756円を使用します。")

    with st.expander("3連複1-2-全 固定想定値", expanded=False):
        st.write(
            f"想定的中率：{TRIO_12_ALL_EXPECTED_HIT_RATE:.1f}% ／ "
            f"想定平均配当：{TRIO_12_ALL_EXPECTED_AVG_PAY:.1f}円 ／ "
            f"ゾーン想定回収率：{TRIO_12_ALL_EXPECTED_ROI:.1f}%"
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "目": k,
                    "回数": TRIO_12_ALL_BASE_COUNTS[k],
                    "平均配当": TRIO_12_ALL_BASE_AVG_PAYS[k],
                }
                for k in TRIO_12_ALL_BASE_COUNTS
            ]),
            use_container_width=True,
            hide_index=True,
        )

    sp_rows = []
    for label in ["仮想全体"]:
        rec = payout_sanrenpuku12_all_total.get(label, new_payout_rec())
        sp_rows.append(sanrenpuku12_row("3連複 1-2-全｜全体", rec))
    df_sp12 = pd.DataFrame(sp_rows)
    sp_cols = [
        "型",
        "対象N",
        "総点数KSUM",
        "投資額換算",
        "払戻合計SUM",
        "的中H",
        "的中率%",
        "平均配当",
        "想定平均配当",
        "平均配当差",
        "的中率差",
        "回収率%",
        "想定1-2両方3着内率%",
        "7車5点_損益分岐平均配当",
        "ゾーン想定回収率%",
        "ゾーン回収差",
    ]
    st.dataframe(df_sp12[[c for c in sp_cols if c in df_sp12.columns]], use_container_width=True, hide_index=True)

    st.markdown("#### 3連複 1-2 個別候補｜2車複方式")
    st.caption("1-2-3～1-2-7を個別に判定します。2車複表と同じく、想定差・配当係数・回収差で過熱目を避け、中庸/歪み枠を推奨します。")
    trio_pick_n = st.number_input(
        "推奨3連複 最大点数",
        key="trio12_pick_n",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    trio_rows = []
    for key in TRIO_12_ALL_BASE_COUNTS:
        rec = payout_sanrenpuku12_individual_total["仮想全体"].get(key, new_payout_rec())
        trio_rows.append(sanrenpuku12_individual_row(f"3連複 {key}", rec, key))
    df_trio_ind = pd.DataFrame(trio_rows)

    if not df_trio_ind.empty:
        # 評価（3～7）の複勝率補正。
        # 1・2は軸として固定済みなので、多重評価を避けるため補正対象にしない。
        for idx in df_trio_ind.index:
            try:
                target_eval = int(str(df_trio_ind.loc[idx, "目"]).split("-")[-1])
            except Exception:
                target_eval = None

            df_trio_ind.loc[idx, "評価"] = target_eval
            base_place = TRIO_BASE_PLACE_RATES.get(target_eval) if target_eval is not None else None
            df_trio_ind.loc[idx, "基準複勝率%"] = base_place

            cur_place = None
            if target_eval in rank_total:
                rec_rank = rank_total.get(target_eval, {"N": 0, "C1": 0, "C2": 0, "C3": 0})
                n_rank = int(rec_rank.get("N", 0))
                if n_rank > 0:
                    cur_place = round(100.0 * (int(rec_rank.get("C1", 0)) + int(rec_rank.get("C2", 0)) + int(rec_rank.get("C3", 0))) / n_rank, 1)
            df_trio_ind.loc[idx, "現在複勝率%"] = cur_place

            if cur_place is not None and base_place is not None:
                place_diff = round(float(cur_place) - float(base_place), 1)
            else:
                place_diff = None
            df_trio_ind.loc[idx, "複勝差"] = place_diff
            df_trio_ind.loc[idx, "複勝状態"] = place_state(place_diff)

        # 推奨判定：未回収・明確な過熱・大幅上振れは除外。
        # さらに評価の波を加味する。
        # 来すぎなら後追い警戒で除外。
        # 評価6・7が基準未満なら「低評価不振」として除外し、ただ高配当だから買う形を避ける。
        df_trio_ind["_place_over"] = df_trio_ind["複勝状態"].eq("来すぎ")
        df_trio_ind["_low_rank_cold"] = (
            df_trio_ind["評価"].fillna(0).astype(float).ge(6)
            & df_trio_ind["複勝状態"].eq("基準未満")
        )
        df_trio_ind["_place_bonus"] = (
            df_trio_ind["評価"].fillna(0).astype(float).between(3, 5)
            & df_trio_ind["複勝状態"].eq("基準未満")
        )

        df_trio_ind["補正理由"] = ""
        df_trio_ind.loc[df_trio_ind["_place_over"], "補正理由"] = "来すぎ"
        df_trio_ind.loc[df_trio_ind["_low_rank_cold"], "補正理由"] = "低評価不振"
        df_trio_ind.loc[df_trio_ind["_place_bonus"], "補正理由"] = "戻り余地"

        df_trio_ind["_eligible"] = (
            (df_trio_ind["的中H"].fillna(0).astype(float) > 0)
            & ~(df_trio_ind["総合候補理由"].isin(["未回収除外", "的中率過熱除外", "後追い除外", "配当上振れ警戒"]))
            & ~df_trio_ind["_place_over"]
            & ~df_trio_ind["_low_rank_cold"]
        )
        df_trio_ind["_score"] = 9999.0
        for idx in df_trio_ind.index:
            diff = df_trio_ind.loc[idx, "想定差"]
            coef = df_trio_ind.loc[idx, "配当係数"]
            roi_diff = df_trio_ind.loc[idx, "回収差"]
            score = 0.0
            try:
                score += abs(float(diff)) if pd.notna(diff) else 20.0
            except Exception:
                score += 20.0
            try:
                score += abs(float(coef) - 1.0) * 8.0 if pd.notna(coef) else 10.0
            except Exception:
                score += 10.0
            try:
                if pd.notna(roi_diff) and float(roi_diff) > 0:
                    score += float(roi_diff) / 20.0
            except Exception:
                pass
            try:
                if bool(df_trio_ind.loc[idx, "_place_bonus"]):
                    score -= 2.0
            except Exception:
                pass
            df_trio_ind.loc[idx, "_score"] = score

        selected_trio_idx = list(
            df_trio_ind.loc[df_trio_ind["_eligible"]]
            .sort_values(["_score", "目"])
            .head(int(trio_pick_n))
            .index
        )
        df_trio_ind.loc[selected_trio_idx, "判定"] = "推奨"
        recommended_trio = df_trio_ind.loc[selected_trio_idx, "目"].tolist()
        if recommended_trio:
            st.success("現在の推奨3連複：" + " / ".join(recommended_trio))
        else:
            st.warning("現在の推奨3連複はありません。1-2-全ではなくケン寄りです。")

        trio_cols = [
            "判定", "目", "評価", "現在複勝率%", "基準複勝率%", "複勝差", "複勝状態", "補正理由",
            "対象N", "的中H", "的中率%", "想定的中率%", "想定差",
            "平均配当", "基準平均配当", "平均配当差", "配当係数", "配当位置", "配当戻り余地",
            "回収率%", "想定回収率%", "回収差", "状態", "総合候補理由",
        ]
        st.dataframe(
            df_trio_ind[[c for c in trio_cols if c in df_trio_ind.columns]],
            use_container_width=True,
            hide_index=True,
            height=260,
        )

    st.markdown("#### 3連複 1-2-全 引継ぎ用累積表")
    sp_carry_rows = []
    for label in ["仮想全体"]:
        rec = payout_sanrenpuku12_all_total.get(label, new_payout_rec())
        row = sanrenpuku12_row(label, rec)
        sp_carry_rows.append({
            "区分": label,
            "対象N": row.get("対象N"),
            "払戻合計SUM": row.get("払戻合計SUM"),
            "的中H": row.get("的中H"),
            "的中率%": row.get("的中率%"),
            "平均配当": row.get("平均配当"),
            "想定平均配当": row.get("想定平均配当"),
            "平均配当差": row.get("平均配当差"),
            "回収率%": row.get("回収率%"),
            "ゾーン想定回収率%": row.get("ゾーン想定回収率%"),
            "ゾーン回収差": row.get("ゾーン回収差"),
        })
    st.dataframe(pd.DataFrame(sp_carry_rows), use_container_width=True, hide_index=True, height=130)

    st.markdown("#### 3連複 1-2 個別 引継ぎ用累積表")
    tri_carry_rows = []
    for label in ["仮想全体"]:
        for key in TRIO_12_ALL_BASE_COUNTS:
            rec = payout_sanrenpuku12_individual_total[label].get(key, new_payout_rec())
            row = sanrenpuku12_individual_row(label, rec, key)
            tri_carry_rows.append({
                "区分": label,
                "目": key,
                "対象N": row.get("対象N"),
                "払戻合計SUM": row.get("払戻合計SUM"),
                "的中H": row.get("的中H"),
                "的中率%": row.get("的中率%"),
                "平均配当": row.get("平均配当"),
                "基準平均配当": row.get("基準平均配当"),
                "想定的中率%": row.get("想定的中率%"),
                "想定回収率%": row.get("想定回収率%"),
                "回収率%": row.get("回収率%"),
                "回収差": row.get("回収差"),
            })
    st.dataframe(pd.DataFrame(tri_carry_rows), use_container_width=True, hide_index=True, height=260)

    st.divider()

    st.markdown("### 個別2車複 引継ぎ用累積表")
    st.caption("次回の『個別2車複 引継ぎ入力』へ転記する表です。対象N・払戻合計SUM・的中Hだけ入力すれば、KSUMは自動で対象Nと同じになります。")

    carry_rows = []
    for a, b in NISHAFUKU_PAIRS:
        label = nishafuku_label(a, b)
        rec = payout_nishafuku_total.get(label, new_payout_rec())
        row = payout_row(label, rec)
        carry_rows.append({
            "型": label,
            "対象N": row.get("対象N"),
            "払戻合計SUM": row.get("払戻合計SUM"),
            "的中H": row.get("的中H"),
            "的中率%": row.get("的中率%"),
            "平均配当": row.get("平均配当"),
            "ペア基準配当": PAIR_BASE_AVG_PAY_DEFAULTS.get(f"{a}-{b}"),
            "想定ペア的%": PAIR_BASE_HIT_RATE_DEFAULTS.get(f"{a}-{b}"),
            "想定回収率%": round(float(PAIR_BASE_HIT_RATE_DEFAULTS.get(f"{a}-{b}", 0)) * float(PAIR_BASE_AVG_PAY_DEFAULTS.get(f"{a}-{b}", 0)) / 100.0, 1),
            "回収率%": row.get("回収率%"),
            "回収差": round(float(row.get("回収率%")) - (float(PAIR_BASE_HIT_RATE_DEFAULTS.get(f"{a}-{b}", 0)) * float(PAIR_BASE_AVG_PAY_DEFAULTS.get(f"{a}-{b}", 0)) / 100.0), 1) if row.get("回収率%") is not None else None,
        })
    st.dataframe(pd.DataFrame(carry_rows), use_container_width=True, hide_index=True, height=430)

    st.markdown("### 買い目確認")
    st.write("今日入力の個別2車複：評価1・評価2軸に必要なペアを自動集計")
    st.write("削除済み：標準棚/穴棚シミュレーター、波履歴、直近3回傾き")
