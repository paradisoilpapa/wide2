# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re
import math, json, requests
from statistics import mean, pstdev
from itertools import combinations
from datetime import datetime, date, time, timedelta, timezone

def _grep_self(pattern: str, path: str = __file__, context: int = 2):
    """
    grep -n の代わり：このファイル(path)を読み、patternを含む行番号を出す
    context: 前後に何行表示するか
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    hits = []
    for i, line in enumerate(lines, 1):
        if pattern in line:
            hits.append(i)

    if not hits:
        print(f"[SELF-GREP] not found: {pattern!r} in {path}")
        return []

    print(f"[SELF-GREP] found {len(hits)} hit(s): {hits}  pattern={pattern!r}")
    for ln in hits:
        s = max(1, ln - context)
        e = min(len(lines), ln + context)
        print("-----")
        for j in range(s, e + 1):
            mark = ">>" if j == ln else "  "
            print(f"{mark}{j:5d}: {lines[j-1].rstrip()}")
    return hits



# ==============================
# 偏差値T（車番→T）自動検出ユーティリティ
# ==============================
def _extract_car_t_map_from_obj(obj):
    """
    obj から「車番→偏差値T(dict)」を取り出す。
    - dict: {1: 52.3, "4": 47.1, ...}
    - Series: indexが車番
    - 1列DataFrame: indexが車番
    """
    if obj is None:
        return None

    # dict
    if isinstance(obj, dict) and obj:
        out = {}
        for k, v in obj.items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas Series
    if isinstance(obj, pd.Series) and not obj.empty:
        out = {}
        for k, v in obj.to_dict().items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas DataFrame（1列だけ偏差値が入ってる想定）
    if isinstance(obj, pd.DataFrame) and (not obj.empty):
        if obj.shape[1] >= 1:
            s = obj.iloc[:, 0]
            return _extract_car_t_map_from_obj(s)

    return None


def _looks_like_t_map(tmap, active_cars=None):
    if not isinstance(tmap, dict) or not tmap:
        return False

    keys = [k for k in tmap.keys() if str(k).isdigit()]
    if len(keys) < 4:
        return False

    vals = []
    for k in keys:
        try:
            vals.append(float(tmap[k]))
        except Exception:
            pass

    if len(vals) < 4:
        return False

    in_range = [v for v in vals if 10.0 <= v <= 90.0]
    if len(in_range) / len(vals) < 0.8:
        return False

    m = sum(in_range) / len(in_range)
    if not (25.0 <= m <= 75.0):
        return False

    if active_cars:
        ac = [str(x) for x in active_cars if str(x).isdigit()]
        if ac:
            hit = sum(1 for x in ac if x in tmap)
            if hit / len(ac) < 0.6:
                return False

    return True


def _pick_hensachi_source_from_globals(g, active_cars=None):
    """
    globals() から偏差値Tソースを自動選別して (tmap, name, score) を返す
    """
    best = None
    best_name = None
    best_score = -1.0

    for name, obj in g.items():
        if name.startswith("__"):
            continue
        tmap = _extract_car_t_map_from_obj(obj)
        if not tmap:
            continue
        if not _looks_like_t_map(tmap, active_cars=active_cars):
            continue

        ac = [str(x) for x in (active_cars or []) if str(x).isdigit()]
        hit = sum(1 for x in ac if x in tmap) if ac else len(tmap)
        coverage = (hit / len(ac)) if ac else 0.5

        vals = [float(v) for v in tmap.values() if isinstance(v, (int, float))]
        uniq = len(set(round(v, 2) for v in vals)) / max(1, len(vals))

        score = coverage * 0.7 + uniq * 0.3

        if score > best_score:
            best_score = score
            best = tmap
            best_name = name

    return best, best_name, best_score


# =========================================================
# 必須：グローバル共通部品（参照より先に必ず定義）
# =========================================================

def _digits_of_line(ln):
    s = "".join(ch for ch in str(ln) if ch.isdigit())
    return [int(ch) for ch in s] if s else []

# _PATTERNS をどこかで for で回しているなら、最低限ここで存在させる
_PATTERNS = []   # ← まず NameError を止めるための保険（本来は下で登録する）




# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 統合版）", layout="wide")

# ==============================
# ★ 新規パラメータ（偏差値＆推奨ロジック）
# ==============================
HEN_W_SB   = 0.20   # SB重み
HEN_W_PROF = 0.30   # 脚質重み
HEN_W_IN   = 0.50   # 入着重み（縮約3着内率）
HEN_DEC_PLACES = 1  # 偏差値 小数一桁

HEN_THRESHOLD = 55.0     # 偏差値クリア閾値
HEN_STRONG_ONE = 60.0    # 単独強者の目安

MAX_TICKETS = 6          # 買い目最大点数

# 推奨ラベル判定用（クリア台数→方針）
# k>=5:「2車複・ワイド」中心（広く） / k=3,4:「3連複」 / k=1,2:「状況次第（軸流し寄り）」 / k=0:ケン
LABEL_MAP = {
    "wide_qn": lambda k: k >= 5,
    "trio":    lambda k: 3 <= k <= 4,
    "axis":    lambda k: k in (1,2),
    "ken":     lambda k: k == 0,
}

# 期待値レンジ（内部基準で使用可。画面非表示）
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60

# ==============================
# 既存：風・会場・マスタ
# ==============================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035,
    "無風": 0.0
}
WIND_MODE = "speed_only"
WIND_SIGN = -1
WIND_GAIN = 3.0
WIND_CAP  = 0.10
WIND_ZERO = 1.5
SPECIAL_DIRECTIONAL_VELODROMES = {"弥彦", "前橋"}

SESSION_HOUR = {"モーニング": 8, "デイ": 11, "ナイター": 18, "ミッドナイト": 22}
JST = timezone(timedelta(hours=9))

BASE_BY_KAKU = {"逃":1.58, "捲":1.65, "差":1.79, "マ":1.45}

KEIRIN_DATA = {
    "函館":{"bank_angle":30.6,"straight_length":51.3,"bank_length":400},
    "青森":{"bank_angle":32.3,"straight_length":58.9,"bank_length":400},
    "いわき平":{"bank_angle":32.9,"straight_length":62.7,"bank_length":400},
    "弥彦":{"bank_angle":32.4,"straight_length":63.1,"bank_length":400},
    "前橋":{"bank_angle":36.0,"straight_length":46.7,"bank_length":335},
    "取手":{"bank_angle":31.5,"straight_length":54.8,"bank_length":400},
    "宇都宮":{"bank_angle":25.8,"straight_length":63.3,"bank_length":500},
    "大宮":{"bank_angle":26.3,"straight_length":66.7,"bank_length":500},
    "西武園":{"bank_angle":29.4,"straight_length":47.6,"bank_length":400},
    "京王閣":{"bank_angle":32.2,"straight_length":51.5,"bank_length":400},
    "立川":{"bank_angle":31.2,"straight_length":58.0,"bank_length":400},
    "松戸":{"bank_angle":29.8,"straight_length":38.2,"bank_length":333},
    "川崎":{"bank_angle":32.2,"straight_length":58.0,"bank_length":400},
    "平塚":{"bank_angle":31.5,"straight_length":54.2,"bank_length":400},
    "小田原":{"bank_angle":35.6,"straight_length":36.1,"bank_length":333},
    "伊東":{"bank_angle":34.7,"straight_length":46.6,"bank_length":333},
    "静岡":{"bank_angle":30.7,"straight_length":56.4,"bank_length":400},
    "名古屋":{"bank_angle":34.0,"straight_length":58.8,"bank_length":400},
    "岐阜":{"bank_angle":32.3,"straight_length":59.3,"bank_length":400},
    "大垣":{"bank_angle":30.6,"straight_length":56.0,"bank_length":400},
    "豊橋":{"bank_angle":33.8,"straight_length":60.3,"bank_length":400},
    "富山":{"bank_angle":33.7,"straight_length":43.0,"bank_length":333},
    "松坂":{"bank_angle":34.4,"straight_length":61.5,"bank_length":400},
    "四日市":{"bank_angle":32.3,"straight_length":62.4,"bank_length":400},
    "福井":{"bank_angle":31.5,"straight_length":52.8,"bank_length":400},
    "奈良":{"bank_angle":33.4,"straight_length":38.0,"bank_length":333},
    "向日町":{"bank_angle":30.5,"straight_length":47.3,"bank_length":400},
    "和歌山":{"bank_angle":32.3,"straight_length":59.9,"bank_length":400},
    "岸和田":{"bank_angle":30.9,"straight_length":56.7,"bank_length":400},
    "玉野":{"bank_angle":30.6,"straight_length":47.9,"bank_length":400},
    "広島":{"bank_angle":30.8,"straight_length":57.9,"bank_length":400},
    "防府":{"bank_angle":34.7,"straight_length":42.5,"bank_length":333},
    "高松":{"bank_angle":33.3,"straight_length":54.8,"bank_length":400},
    "小松島":{"bank_angle":29.8,"straight_length":55.5,"bank_length":400},
    "高知":{"bank_angle":24.5,"straight_length":52.0,"bank_length":500},
    "松山":{"bank_angle":34.0,"straight_length":58.6,"bank_length":400},
    "小倉":{"bank_angle":34.0,"straight_length":56.9,"bank_length":400},
    "久留米":{"bank_angle":31.5,"straight_length":50.7,"bank_length":400},
    "武雄":{"bank_angle":32.0,"straight_length":64.4,"bank_length":400},
    "佐世保":{"bank_angle":31.5,"straight_length":40.2,"bank_length":400},
    "別府":{"bank_angle":33.7,"straight_length":59.9,"bank_length":400},
    "熊本":{"bank_angle":34.3,"straight_length":60.3,"bank_length":400},
    "手入力":{"bank_angle":30.0,"straight_length":52.0,"bank_length":400},
}
VELODROME_MASTER = {
    "函館":{"lat":41.77694,"lon":140.76283,"home_azimuth":None},
    "青森":{"lat":40.79717,"lon":140.66469,"home_azimuth":None},
    "いわき平":{"lat":37.04533,"lon":140.89150,"home_azimuth":None},
    "弥彦":{"lat":37.70778,"lon":138.82886,"home_azimuth":None},
    "前橋":{"lat":36.39728,"lon":139.05778,"home_azimuth":None},
    "取手":{"lat":35.90175,"lon":140.05631,"home_azimuth":None},
    "宇都宮":{"lat":36.57197,"lon":139.88281,"home_azimuth":None},
    "大宮":{"lat":35.91962,"lon":139.63417,"home_azimuth":None},
    "西武園":{"lat":35.76983,"lon":139.44686,"home_azimuth":None},
    "京王閣":{"lat":35.64294,"lon":139.53372,"home_azimuth":None},
    "立川":{"lat":35.70214,"lon":139.42300,"home_azimuth":None},
    "松戸":{"lat":35.80417,"lon":139.91119,"home_azimuth":None},
    "川崎":{"lat":35.52844,"lon":139.70944,"home_azimuth":None},
    "平塚":{"lat":35.32547,"lon":139.36342,"home_azimuth":None},
    "小田原":{"lat":35.25089,"lon":139.14947,"home_azimuth":None},
    "伊東":{"lat":34.954667,"lon":139.092639,"home_azimuth":None},
    "静岡":{"lat":34.973722,"lon":138.419417,"home_azimuth":None},
    "名古屋":{"lat":35.175560,"lon":136.854028,"home_azimuth":None},
    "岐阜":{"lat":35.414194,"lon":136.783917,"home_azimuth":None},
    "大垣":{"lat":35.361389,"lon":136.628444,"home_azimuth":None},
    "豊橋":{"lat":34.770167,"lon":137.417250,"home_azimuth":None},
    "富山":{"lat":36.757250,"lon":137.234833,"home_azimuth":None},
    "松坂":{"lat":34.564611,"lon":136.533833,"home_azimuth":None},
    "四日市":{"lat":34.965389,"lon":136.634500,"home_azimuth":None},
    "福井":{"lat":36.066889,"lon":136.253722,"home_azimuth":None},
    "奈良":{"lat":34.681111,"lon":135.823083,"home_azimuth":None},
    "向日町":{"lat":34.949222,"lon":135.708389,"home_azimuth":None},
    "和歌山":{"lat":34.228694,"lon":135.171833,"home_azimuth":None},
    "岸和田":{"lat":34.477500,"lon":135.369389,"home_azimuth":None},
    "玉野":{"lat":34.497333,"lon":133.961389,"home_azimuth":None},
    "広島":{"lat":34.359778,"lon":132.502889,"home_azimuth":None},
    "防府":{"lat":34.048778,"lon":131.568611,"home_azimuth":None},
    "高松":{"lat":34.345936,"lon":134.061994,"home_azimuth":None},
    "小松島":{"lat":34.005667,"lon":134.594556,"home_azimuth":None},
    "高知":{"lat":33.566694,"lon":133.526083,"home_azimuth":None},
    "松山":{"lat":33.808889,"lon":132.742333,"home_azimuth":None},
    "小倉":{"lat":33.885722,"lon":130.883167,"home_azimuth":None},
    "久留米":{"lat":33.316667,"lon":130.547778,"home_azimuth":None},
    "武雄":{"lat":33.194083,"lon":130.023083,"home_azimuth":None},
    "佐世保":{"lat":33.161667,"lon":129.712833,"home_azimuth":None},
    "別府":{"lat":33.282806,"lon":131.460472,"home_azimuth":None},
    "熊本":{"lat":32.789167,"lon":130.754722,"home_azimuth":None},
    "手入力":{"lat":None,"lon":None,"home_azimuth":None},
}

# --- 印別実測率（統計） ---
# NOTE: KO（隊列ノックアウト）には使わない。混ぜると「統計が順位をワープさせる」ため。
RANK_STATS_TOTAL = {
    "◎": {"p1": 0.261, "pTop2": 0.459, "pTop3": 0.617},
    "〇": {"p1": 0.235, "pTop2": 0.403, "pTop3": 0.533},
    "▲": {"p1": 0.175, "pTop2": 0.331, "pTop3": 0.484},
    "△": {"p1": 0.133, "pTop2": 0.282, "pTop3": 0.434},
    "×": {"p1": 0.109, "pTop2": 0.242, "pTop3": 0.390},
    "α": {"p1": 0.059, "pTop2": 0.167, "pTop3": 0.295},
    "無": {"p1": 0.003, "pTop2": 0.118, "pTop3": 0.256},
}

def compute_weighted_rank_from_carfr_text(carfr_text: str):
    """
    統計混入スコア（FR×印別実測率）は現在は使用しない。
    互換のため関数だけ残し、常に空を返す。
    """
    return []






# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.007   # 0.010 → 0.007
KO_STEP_SIGMA = 0.35   # 0.4 → 0.35


# ◎ライン格上げ
LINE_BONUS_ON_TENKAI = {"優位"}
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}
LINE_BONUS_CAP = 0.10
PROB_U = {"second": 0.00, "thirdplus": 0.00}

# --- 安定度（着順分布）をT本体に入れるための重み ---
STAB_W_IN3  = 0.10   # 3着内率の重み
STAB_W_OUT  = 0.12   # 着外率の重み（マイナス補正）
STAB_W_LOWN = 0.05   # サンプル不足補正
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45
def _stab_n0(n: int) -> int:
    """サンプル不足時の事前分布の強さ（nが小さいほど強く効かせる）"""
    if n <= 6: return 12
    if n <= 14: return 8
    if n <= 29: return 5
    return 3
# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): return max(a, min(b, x))

def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s

def zscore_val(x, xs):
    xs = np.array(xs, dtype=float); m, s = float(np.mean(xs)), float(np.std(xs))
    return 0.0 if s==0 else (float(x)-m)/s

# ==============================
# H：最終ホーム地力補正
# ==============================
H_SCORE_SCALE = float(globals().get("H_SCORE_SCALE", 0.045))
H_SCORE_CAP   = float(globals().get("H_SCORE_CAP", 0.075))

def calc_h_score_map(H: dict, active_cars: list[int]) -> dict[int, float]:
    """
    Hをレース内z化して、車番ごとの相対H評価を作る。
    絶対値ではなく、そのレース内でHが高いか低いかを見る。
    """
    vals = np.array(
        [float(H.get(int(n), 0.0)) for n in active_cars],
        dtype=float
    )

    if len(vals) < 2:
        return {int(n): 0.0 for n in active_cars}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))

    if sd < 1e-12:
        return {int(n): 0.0 for n in active_cars}

    return {
        int(n): float((float(H.get(int(n), 0.0)) - mu) / sd)
        for n in active_cars
    }


def h_home_bonus(no: int, role: str, H_Z: dict[int, float]) -> float:
    """
    H補正。
    ライン先頭・単騎を中心に加点。
    番手・三番手は薄く反映。
    """
    role_mult = {
        "head": 1.00,
        "single": 0.70,
        "second": 0.30,
        "thirdplus": 0.15,
    }.get(role, 0.20)

    raw = H_SCORE_SCALE * float(H_Z.get(int(no), 0.0)) * role_mult
    return round(clamp(raw, -H_SCORE_CAP, H_SCORE_CAP), 3)


def h_lead_line_bonus(
    no: int,
    role: str,
    H: dict,
    B: dict,
    line_def: dict,
    home_top_gid,
) -> float:
    """
    H主導ラインの先頭車だけを下支えする補正。
    目的：H主導ライン先頭がKO最下位まで沈む現象を防ぐ。
    """
    try:
        if home_top_gid is None:
            return 0.0

        members = line_def.get(home_top_gid, [])
        if not members:
            return 0.0

        head = int(members[0])

        # H主導ラインの先頭車だけ対象
        if int(no) != head:
            return 0.0

        # 役割が先頭でないなら対象外
        if role != "head":
            return 0.0

        h_val = float(H.get(int(no), 0.0) or 0.0)
        b_val = float(B.get(int(no), 0.0) or 0.0)

        # Hが低いなら補正しない
        if h_val < 3.0:
            return 0.0

        # Hを主、Bを補助にする
        bonus = 0.035 + 0.004 * h_val + 0.002 * b_val

        # 暴走防止
        return round(clamp(bonus, 0.0, 0.090), 3)

    except Exception:
        return 0.0
    raw = H_SCORE_SCALE * float(H_Z.get(int(no), 0.0)) * role_mult
    return round(clamp(raw, -H_SCORE_CAP, H_SCORE_CAP), 3)


def t_score_from_finite(values: np.ndarray, eps: float = 1e-9):
    """NaNを除いた母集団でT=50+10*(x-μ)/σを作り、NaNは50に置換して返す"""
    v = values.astype(float, copy=True)
    finite = np.isfinite(v)
    k = int(finite.sum())
    if k < 2:
        return np.full_like(v, 50.0), (float("nan") if k==0 else float(v[finite][0])), 0.0, k
    mu = float(np.mean(v[finite]))
    sd = float(np.std(v[finite], ddof=0))
    if (not np.isfinite(sd)) or sd < eps:
        return np.full_like(v, 50.0), mu, 0.0, k
    T = 50.0 + 10.0 * ((v - mu) / sd)
    T[~finite] = 50.0
    return T, mu, sd, k

def extract_car_list(s, n_cars=None):
    """
    ライン入力文字列から車番を抽出する。
    出走数n_carsでは車番を制限しない。
    5車立てでも 12346 のような欠番ありを許可する。
    """
    cars = []
    seen = set()

    for ch in str(s):
        if not ch.isdigit():
            continue

        v = int(ch)

        # 競輪の車番として1〜9だけ許可
        if 1 <= v <= 9 and v not in seen:
            cars.append(v)
            seen.add(v)

    return cars
def build_line_maps(lines):
    labels = "ABCDEFG"
    line_def = {labels[i]: lst for i,lst in enumerate(lines) if lst}
    car_to_group = {c:g for g,mem in line_def.items() for c in mem}
    return line_def, car_to_group

def role_in_line(car, line_def):
    for g, mem in line_def.items():
        if car in mem:
            if len(mem)==1: return 'single'
            idx = mem.index(car)
            return ['head','second','thirdplus'][idx] if idx<3 else 'thirdplus'
    return 'single'
# =====================================================
# ラスト半周補正：番手差し・前で動ける上位補正
# =====================================================

LAST_HALF_ENABLE = True

# ラスト半周補正の全体上限
LAST_HALF_CAP = 0.050

# 番手補正の上限
LAST_HALF_SECOND_CAP = 0.050

# 先頭・単騎の前で動ける補正の上限
LAST_HALF_FRONT_CAP = 0.040


def _is_top_third(rank_val, top_third_limit: int) -> bool:
    """
    レース内上位1/3判定。
    7車なら3位以内。
    """
    try:
        return int(rank_val) <= int(top_third_limit)
    except Exception:
        return False


def calc_last_half_role_bonus(
    role: str,
    kaku: str,
    tenscore: float,
    leader_tenscore: float,
    race_avg_tenscore: float,
    h_count: float = 0.0,
    b_count: float = 0.0,
    race_score_rank=None,
    ko_score_rank=None,
    tenkai_score_rank=None,
    top_third_limit: int = 3,
    scenario_top_count: int = 0,
    p1_rate=None,
    p2_rate=None,
    p3_rate=None,
):
    """
    ラスト半周〜ゴール前の個人戦補正。

    思想：
    ラスト半周までは団体戦。
    ラスト半周からは個人戦。
    そのため、位置ではなく「実際に着を取れる個人成績」で補正する。

    使用するもの：
    ・1着率
    ・2着内率
    ・3着内率

    使わないもの：
    ・番手位置だけの加点
    ・H/Bだけの加点
    ・自力だから加点
    ・単騎だから加点
    ・H主導3番手以降だから加点
    """

    if not LAST_HALF_ENABLE:
        return 0.0, []

    bonus = 0.0
    reasons = []

    try:
        role = str(role)

        def _rate(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        p1 = _rate(p1_rate)
        p2 = _rate(p2_rate)
        p3 = _rate(p3_rate)

        # ---------------------------------------------
        # 個人戦補正
        # ---------------------------------------------
        # 勝ち切れる個人力を強めに評価
        if p1 is not None and p1 >= 0.20:
            bonus += 0.025
            reasons.append(f"1着率{p1 * 100:.0f}%以上")

        # 2着内率は評価するが、1着率より軽くする
        if p2 is not None and p2 >= 0.30:
            bonus += 0.010
            reasons.append(f"2着内率{p2 * 100:.0f}%以上")

        # 3着内率は、2着内率もある場合だけ補正
        # 3着に残るだけの選手をラスト半周個人力として過大評価しない
        if (
            p3 is not None
            and p3 >= 0.40
            and p2 is not None
            and p2 >= 0.30
        ):
            bonus += 0.010
            reasons.append(f"3着内率{p3 * 100:.0f}%以上")

        # ---------------------------------------------
        # 役割別上限
        # 位置で加点はしない。
        # ただし3番手以降だけは暴走防止で上限を低くする。
        # ---------------------------------------------
        if role == "thirdplus":
            role_cap = 0.030
        else:
            role_cap = 0.050

        bonus = clamp(bonus, 0.0, role_cap)
        bonus = clamp(bonus, -LAST_HALF_CAP, LAST_HALF_CAP)

        if not reasons:
            reasons.append("補正なし")

        return round(float(bonus), 3), reasons

    except Exception as e:
        return 0.0, [f"ラスト半周補正エラー:{e}"]

# ==============================

# =====================================================
# 混戦度判定
#   平均得点ではなく、競走得点1位と2位の差で見る
#   High   = 上位差が大きく、順当寄り
#   Middle = 標準
#   Low    = 上位差が小さく、波乱気味
#
#   ※スコア補正には使わない。表示・検証用。
# =====================================================
def calc_race_compactness(ratings_val: dict, active_cars: list):
    vals = []

    for no in active_cars:
        try:
            v = float(ratings_val.get(int(no), 0.0))
            if v > 0:
                vals.append(v)
        except Exception:
            pass

    if len(vals) < 2:
        return {
            "label": "未判定",
            "top1": 0.0,
            "top2": 0.0,
            "top_gap": None,
        }

    vals = sorted(vals, reverse=True)

    top1 = vals[0]
    top2 = vals[1]
    top_gap = top1 - top2

    if top_gap >= 2.00:
        label = "順当寄り"
    elif top_gap >= 1.00:
        label = "標準"
    else:
        label = "波乱気味"

    return {
        "label": label,
        "top1": float(top1),
        "top2": float(top2),
        "top_gap": float(top_gap),
    }

# H：最終ホーム想定ライン
# ==============================
def calc_home_line_scores(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> dict:
    """
    H = 最終ホーム先頭通過回数を使って、
    最終周回ホームで前に出やすいラインを評価する。
    ※本体スコアには混ぜず、展開表示用。
    """
    scores = {}

    for gid, members in line_def.items():
        mem = [int(x) for x in members if int(x) in active_cars]
        if not mem:
            continue

        head = mem[0]
        second = mem[1] if len(mem) >= 2 else None
        third = mem[2] if len(mem) >= 3 else None

        head_h = float(H.get(head, 0))
        second_h = float(H.get(second, 0)) if second is not None else 0.0
        third_h = float(H.get(third, 0)) if third is not None else 0.0

        # 単騎は自分のHをそのまま見る
        if len(mem) == 1:
            score = head_h
        else:
            # ライン先頭のHを主役、番手・三番手は補助
            score = head_h * 0.75 + second_h * 0.15 + third_h * 0.10

        # 同点時の微差用：Bをほんの少しだけ見る
        score += float(B.get(head, 0)) * 0.01

        scores[gid] = round(score, 3)

    return scores


def make_home_line_order(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> list:
    """
    最終ホーム想定ライン順を返す。
    """
    scores = calc_home_line_scores(line_def, H, B, active_cars)

    return sorted(
        scores.keys(),
        key=lambda gid: scores.get(gid, 0.0),
        reverse=True
    )


def format_home_line_order(line_def: dict, order: list) -> str:
    """
    A/B/Cなどのgid順を、実際のライン文字列に変換する。
    例：['B','C','A'] → 26　37　145
    """
    parts = []

    for gid in order:
        members = line_def.get(gid, [])
        if members:
            parts.append("".join(str(int(x)) for x in members))

    return "　".join(parts) if parts else "—"


# 単騎を全体的に抑える共通係数（あとでいじれるようにする）
SINGLE_NERF = float(globals().get("SINGLE_NERF", 0.85))  # 0.80〜0.88くらいで調整

def pos_coeff(role, line_factor):
    base_map = {
        'head':      1.00,
        'second':    0.72,   # 0.70→0.72に少し上げてライン2番手をちゃんと評価
        'thirdplus': 0.55,
        'single':    0.52,   # 0.90 → 0.52 にドンと落とす
    }
    base = base_map.get(role, 0.52)
    if role == 'single':
        base *= SINGLE_NERF      # ここでさらに細かく落とせる
    return base * line_factor


def tenscore_correction(tenscores):
    n = len(tenscores)
    if n<=2: return [0.0]*n
    df = pd.DataFrame({"得点":tenscores})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n,8)
    baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def track_effective_ratio(track_name: str,
                           alpha_goal: float = 0.50,
                           beta_corner: float = 0.25) -> float:
    d = KEIRIN_DATA.get(track_name)
    if not d:
        return 0.50
    lap  = float(d.get("bank_length", 400))
    home = float(d.get("straight_length", 52.0))
    back = 2.0 * home  # ゴール前は半分の仮定
    corner_total = max(lap - home - back, 0.0)
    L_eff = back + alpha_goal * home + beta_corner * corner_total
    ratio = (L_eff / lap) if lap > 0 else 0.50
    return clamp(ratio, 0.20, 0.90)

def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    s = max(0.0, float(wind_speed))
    WIND_ZERO   = float(globals().get("WIND_ZERO", 0.0))
    WIND_SIGN   = float(globals().get("WIND_SIGN", 1.0))
    WIND_GAIN   = float(globals().get("WIND_GAIN", 1.0))
    WIND_CAP    = float(globals().get("WIND_CAP", 0.06))
    WIND_MODE   = globals().get("WIND_MODE", "scalar")
    WIND_COEFF  = globals().get("WIND_COEFF", {})
    SPECIAL_DIRECTIONAL_VELODROMES = globals().get("SPECIAL_DIRECTIONAL_VELODROMES", set())

    try:
        s_state_track = st.session_state.get("track", "")
    except Exception:
        s_state_track = ""

    # --- 風速→基礎量 ---
    if s <= WIND_ZERO:
        base = 0.0
    elif s <= 5.0:
        base = 0.006 * (s - WIND_ZERO)
    elif s <= 8.0:
        base = 0.021 + 0.008 * (s - 5.0)
    else:
        base = 0.045 + 0.010 * min(s - 8.0, 4.0)

    # --- 位置係数 ---
    pos = {'head':1.00,'second':0.85,'single':0.75,'thirdplus':0.65}.get(role, 0.75)

    # ===== ★ここ①：強風ほど番手・後位を不利にする =====
    wind01 = clamp((s - WIND_ZERO) / (8.0 - WIND_ZERO), 0.0, 1.0)
    track_ratio = track_effective_ratio(s_state_track)
    wind_eff01 = wind01 * track_ratio

    if role in ("second", "thirdplus"):
        pos *= (1.0 - 0.20 * wind_eff01)   # 最大20%だけ削る

    # --- 脚質（自力） ---
    prof = 0.35 + 0.65 * float(prof_escape)
    val = base * pos * prof

    # --- 風向き（既存） ---
    if (WIND_MODE == "directional") or (s >= 7.0 and s_state_track in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(
            s * wd * (0.30 + 0.70 * float(prof_escape)) * 0.6,
            -0.03, 0.03
        )
        val += dir_term

    # ===== ★ここ②：会場ごとに風の効きをスケール =====
    val *= clamp(track_ratio / 0.50, 0.60, 1.40)

    val = (val * float(WIND_SIGN)) * float(WIND_GAIN)
    return round(clamp(val, -float(WIND_CAP), float(WIND_CAP)), 3)


# === 直線ラスト200m（残脚）補正｜33バンク対応版 ==============================
# 33（<=340m）は「先行ペナ弱め／差し・追込ボーナス控えめ」へ最適化
L200_ESC_PENALTY = float(globals().get("L200_ESC_PENALTY", -0.06))  # 先行は垂れやすい（基本）
L200_SASHI_BONUS = float(globals().get("L200_SASHI_BONUS", +0.03))  # 差しは伸びやすい
L200_MARK_BONUS  = float(globals().get("L200_MARK_BONUS",  +0.02))  # 追込は少し上げ

L200_GRADE_GAIN  = globals().get("L200_GRADE_GAIN", {
    "F2": 1.18, "F1": 1.10, "G": 1.05, "GIRLS": 0.95, "TOTAL": 1.00
})

# 短走路増幅：旧1.15 → 33はむしろ緩和（0.85）
L200_SHORT_GAIN_33   = float(globals().get("L200_SHORT_GAIN_33", 0.85))
L200_SHORT_GAIN_OTH  = float(globals().get("L200_SHORT_GAIN_OTH", 1.00))
L200_LONG_RELAX      = float(globals().get("L200_LONG_RELAX", 0.90))
L200_CAP             = float(globals().get("L200_CAP", 0.08))
L200_WET_GAIN        = float(globals().get("L200_WET_GAIN", 1.15))

# 33専用 成分別スケーリング
L200_33_ESC_MULT   = float(globals().get("L200_33_ESC_MULT", 0.80))  # 逃ペナ 20%縮小
L200_33_SASHI_MULT = float(globals().get("L200_33_SASHI_MULT", 0.85))# 差し  15%縮小
L200_33_MARK_MULT  = float(globals().get("L200_33_MARK_MULT", 0.90)) # 追込  10%縮小

def _grade_key_from_class(race_class: str) -> str:
    if "ガール" in race_class: return "GIRLS"
    if "Ｓ級" in race_class or "S級" in race_class: return "G"
    if "チャレンジ" in race_class: return "F2"
    if "Ａ級" in race_class or "A級" in race_class: return "F1"
    return "TOTAL"

def l200_adjust(role: str,
                straight_length: float,
                bank_length: float,
                race_class: str,
                prof_escape: float,    # 逃
                prof_sashi: float,     # 差
                prof_oikomi: float,    # マ
                is_wet: bool = False) -> float:
    """
    ラスト200mの“残脚”を脚質×バンク×グレードで調整した無次元値（±）を返す。
    ※ ENV合計（total_raw）には足さず、独立柱として z 化→anchor_score へ。
    """
    esc_term   = L200_ESC_PENALTY * float(prof_escape)
    sashi_term = L200_SASHI_BONUS * float(prof_sashi)
    mark_term  = L200_MARK_BONUS  * float(prof_oikomi)

    is_33 = float(bank_length) <= 340.0
    if is_33:
        esc_term   *= L200_33_ESC_MULT
        sashi_term *= L200_33_SASHI_MULT
        mark_term  *= L200_33_MARK_MULT

    base = esc_term + sashi_term + mark_term

    if is_33:
        base *= L200_SHORT_GAIN_33
    else:
        base *= L200_SHORT_GAIN_OTH

    if float(straight_length) >= 60.0:
        base *= L200_LONG_RELAX

    base *= float(L200_GRADE_GAIN.get(_grade_key_from_class(race_class), 1.0))

    if is_wet:
        base *= L200_WET_GAIN

    pos_factor = {'head':1.00,'second':0.85,'thirdplus':0.70,'single':0.80}.get(role, 0.80)
    base *= pos_factor

    return round(clamp(base, -float(L200_CAP), float(L200_CAP)), 3)


def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi, bank_length=None):
    straight_factor = (float(straight_length)-40.0)/10.0
    angle_factor = (float(bank_angle)-25.0)/5.0
    total = clamp(-0.1*straight_factor + 0.1*angle_factor, -0.05, 0.05)
    return round(total*prof_escape - 0.5*total*prof_sashi, 3)

def bank_length_adjust(bank_length, prof_oikomi):
    delta = clamp((float(bank_length)-411.0)/100.0, -0.05, 0.05)
    return round(delta*prof_oikomi, 3)

# --- ラインSBボーナス（33mは自動で半減） --------------------
def compute_lineSB_bonus(line_def, S, B, line_factor=1.0, exclude=None, cap=0.06, enable=True):
    """
    33m系（<=340）では自動で効きを半減:
      - LINE_SB_33_MULT（既定0.5）を line_factor に乗算
      - LINE_SB_CAP_33_MULT（既定0.5）を cap に乗算
    """
    if not enable or not line_def:
        return ({g: 0.0 for g in line_def.keys()} if line_def else {}), {}

    # 33かどうかの自動推定
    try:
        bank_len = st.session_state.get("bank_length", st.session_state.get("track_length", None))
    except Exception:
        bank_len = globals().get("BANK_LENGTH", None)

    eff_line_factor = float(line_factor)
    eff_cap = float(cap)

    if bank_len is not None:
        try:
            if float(bank_len) <= 340.0:
                mult = float(globals().get("LINE_SB_33_MULT", 0.50))
                capm = float(globals().get("LINE_SB_CAP_33_MULT", 0.50))
                eff_line_factor *= mult
                eff_cap *= capm
        except Exception:
            pass

    # ライン内の位置重み（単騎を下げる）
    w_pos_base = {
        "head":      1.00,
        "second":    0.55,
        "thirdplus": 0.38,
        "single":    0.34,
    }

    # ラインごとのS/B集計
    Sg = {}
    Bg = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for car in mem:
            if exclude is not None and car == exclude:
                continue
            role = role_in_line(car, line_def)
            w = w_pos_base[role] * eff_line_factor
            s += w * float(S.get(car, 0))
            b += w * float(B.get(car, 0))
        Sg[g] = s
        Bg[g] = b

    # ラインごとの“強さ”スコア
    raw = {}
    for g in line_def.keys():
        s = Sg[g]
        b = Bg[g]
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)

    # z化してボーナス化
    zz = zscore_list(list(raw.values())) if raw else []
    bonus = {}
    for i, g in enumerate(raw.keys()):
        bonus[g] = clamp(0.02 * float(zz[i]), -eff_cap, eff_cap)

    return bonus, raw


# ==============================
# KO Utilities（ここから下を1かたまりで）
# ==============================

def _role_of(car, mem):
    """ラインの中での役割を返す（head / second / thirdplus / single）"""
    if len(mem) == 1:
        return "single"
    idx = mem.index(car)
    return ["head", "second", "thirdplus"][idx] if idx < 3 else "thirdplus"


# KOでも、ライン強度でも、同じ位置重みを使う
LINE_W_POS = {
    "head":      1.00,
    "second":    0.55,
    "thirdplus": 0.38,
    "single":    0.34,
}


def _line_strength_raw(line_def, S, B, line_factor: float = 1.0) -> dict:
    """
    KOやトップ2ライン抽出で使う“生のライン強度”
    compute_lineSB_bonus と式をそろえてある
    """
    if not line_def:
        return {}

    w_pos = {k: v * float(line_factor) for k, v in LINE_W_POS.items()}

    raw: dict[str, float] = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for c in mem:
            role = _role_of(c, mem)
            w = w_pos.get(role, 0.34)
            s += w * float(S.get(c, 0))
            b += w * float(B.get(c, 0))
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)
    return raw


def _top2_lines(line_def, S, B, line_factor=1.0):
    """ラインの中から強い2本を取る"""
    raw = _line_strength_raw(line_def, S, B, line_factor)
    order = sorted(raw.keys(), key=lambda g: raw[g], reverse=True)
    return (order[0], order[1]) if len(order) >= 2 else (order[0], None) if order else (None, None)


def _extract_role_car(line_def, gid, role_name):
    """指定ラインのhead/secondを抜く"""
    if gid is None or gid not in line_def:
        return None
    mem = line_def[gid]
    if role_name == "head":
        return mem[0] if len(mem) >= 1 else None
    if role_name == "second":
        return mem[1] if len(mem) >= 2 else None
    return None


def _ko_order(v_base_map,
              line_def,
              S,
              B,
              line_factor: float = 1.0,
              gap_delta: float = 0.007):
    """
    KO用の並びを作る
    1) 上2ラインのhead
    2) 上2ラインのsecond
    3) 残りのラインの残りをスコア順
    4) その他の車番
    同じライン内でスコア差が gap_delta 以内なら寄せる
    """
    cars = list(v_base_map.keys())

    # ラインが無いときはふつうにスコア順
    if not line_def or len(line_def) < 1:
        return [c for c, _ in sorted(v_base_map.items(), key=lambda x: x[1], reverse=True)]

    g1, g2 = _top2_lines(line_def, S, B, line_factor)

    head1 = _extract_role_car(line_def, g1, "head")
    head2 = _extract_role_car(line_def, g2, "head")
    sec1  = _extract_role_car(line_def, g1, "second")
    sec2  = _extract_role_car(line_def, g2, "second")

    others: list[int] = []
    if g1:
        mem = line_def[g1]
        if len(mem) >= 3:
            others += mem[2:]
    if g2:
        mem = line_def[g2]
        if len(mem) >= 3:
            others += mem[2:]
    for g, mem in line_def.items():
        if g not in {g1, g2}:
            others += mem

    order: list[int] = []

    # 1) headをスコア順で
    head_pair = [x for x in [head1, head2] if x is not None]
    order += sorted(head_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 2) secondをスコア順で
    sec_pair = [x for x in [sec1, sec2] if x is not None]
    order += sorted(sec_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 3) 残りラインの残り（重複を落とす）
    others = list(dict.fromkeys([c for c in others if c is not None]))
    others_sorted = sorted(others, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    order += [c for c in others_sorted if c not in order]

    # 4) まだ出てない車を最後に
    for c in cars:
        if c not in order:
            order.append(c)

    # ライン内の小差詰め
    def _same_group(a, b):
        if a is None or b is None:
            return False
        ga = next((g for g, mem in line_def.items() if a in mem), None)
        gb = next((g for g, mem in line_def.items() if b in mem), None)
        return ga is not None and ga == gb

        i = 0
    while i < len(order) - 2:
        a, b, c = order[i], order[i + 1], order[i + 2]
        if _same_group(a, b):
            vx = v_base_map.get(b, 0.0) - v_base_map.get(c, 0.0)
            # b と c の差が小さいなら入れ替えて “寄せる”
            if vx >= -gap_delta:
                order[i + 1], order[i + 2] = order[i + 2], order[i + 1]
        i += 1

    return order


def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed * (1.0 + E_MIN), needed * (1.0 + E_MAX)


def apply_anchor_line_bonus(score_raw: dict[int, float],
                            line_of: dict[int, str],   # ★ int→str に直す
                            role_map: dict[int, str],
                            anchor: int,
                            tenkai: str) -> dict[int, float]:


    a_line = line_of.get(anchor, None)
    is_on = (tenkai in LINE_BONUS_ON_TENKAI) and (a_line is not None)
    score_adj: dict[int, float] = {}
    for i, s in score_raw.items():
        bonus = 0.0
        if is_on and line_of.get(i) == a_line and i != anchor:
            role = role_map.get(i, "single")
            bonus = min(max(0.0, LINE_BONUS.get(role, 0.0)), LINE_BONUS_CAP)
        score_adj[i] = s + bonus
    return score_adj


from typing import Optional, Dict

def format_rank_all(score_map: Dict[int, float], P_floor_val: Optional[float] = None) -> str:
    order = sorted(score_map.keys(), key=lambda k: (-score_map[k], k))
    rows = []
    for i in order:
        if P_floor_val is None:
            rows.append(f"{i}")
        else:
            rows.append(f"{i}" if score_map[i] >= P_floor_val else f"{i}(P未満)")
    return " ".join(rows)




# ==============================
# 風の自動取得（Open-Meteo / 時刻固定）
# ==============================
def fetch_openmeteo_hour(lat, lon, target_dt_naive):
    import numpy as np
    d = target_dt_naive.strftime("%Y-%m-%d")
    base = "https://api.open-meteo.com/v1/forecast"
    urls = [
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m,precipitation,weather_code"
         "&timezone=Asia%2FTokyo"
         "&windspeed_unit=ms"
         f"&start_date={d}&end_date={d}", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,precipitation,weather_code"
         "&timezone=Asia%2FTokyo"
         "&windspeed_unit=ms"
         f"&start_date={d}&end_date={d}", False),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m,precipitation,weather_code"
         "&timezone=Asia%2FTokyo"
         "&windspeed_unit=ms"
         "&past_days=2&forecast_days=2", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,precipitation,weather_code"
         "&timezone=Asia%2FTokyo"
         "&windspeed_unit=ms"
         "&past_days=2&forecast_days=2", False),
    ]
    last_err = None
    for url, with_dir in urls:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            j = r.json().get("hourly", {})
            times = [datetime.fromisoformat(t) for t in j.get("time", [])]
            if not times:
                raise RuntimeError("empty hourly times")
            diffs = [abs((t - target_dt_naive).total_seconds()) for t in times]
            k = int(np.argmin(diffs))
            sp = j.get("wind_speed_10m", [])
            di = j.get("wind_direction_10m", []) if with_dir else []
            pr = j.get("precipitation", [])
            wc = j.get("weather_code", [])

            speed = float(sp[k]) if k < len(sp) else float("nan")
            deg = float(di[k]) if with_dir and k < len(di) and di[k] is not None else None
            precip = float(pr[k]) if k < len(pr) and pr[k] is not None else 0.0
            weather_code = int(wc[k]) if k < len(wc) and wc[k] is not None else None

            return {
                "time": times[k],
                "speed_ms": speed,
                "deg": deg,
                "precipitation": precip,
                "weather_code": weather_code,
                "diff_min": diffs[k] / 60.0,
             }
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Open-Meteo取得失敗（最後のエラー: {last_err}）")


# ==============================
# サイドバー：開催情報 / バンク・風・頭数
# ==============================

# --- 会場差分（得意会場平均を標準）ヘルパー（このブロック内に自己完結）
FAVORABLE_VENUES = ["名古屋","いわき平","前橋","立川","宇都宮","岸和田","高知"]

def _std_from_venues(names):
    Ls = [KEIRIN_DATA[v]["straight_length"] for v in names if v in KEIRIN_DATA]
    Th = [KEIRIN_DATA[v]["bank_angle"]      for v in names if v in KEIRIN_DATA]
    Cs = [KEIRIN_DATA[v]["bank_length"]     for v in names if v in KEIRIN_DATA]
    return (float(np.mean(Th)), float(np.mean(Ls)), float(np.mean(Cs)))

TH_STD, L_STD, C_STD = _std_from_venues(FAVORABLE_VENUES)

_ALL_L  = np.array([KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA], float)
_ALL_TH = np.array([KEIRIN_DATA[k]["bank_angle"]      for k in KEIRIN_DATA], float)
SIG_L  = float(np.std(_ALL_L))  if np.std(_ALL_L)  > 1e-9 else 1.0
SIG_TH = float(np.std(_ALL_TH)) if np.std(_ALL_TH) > 1e-9 else 1.0

def venue_z_terms(straight_length: float, bank_angle: float, bank_length: float):
    zL  = (float(straight_length) - L_STD)  / SIG_L
    zTH = (float(bank_angle)      - TH_STD) / SIG_TH
    if bank_length >= 480: dC = +0.4
    elif bank_length >= 380: dC = 0.0
    else: dC = -0.4
    return zL, zTH, dC

def venue_mix(zL, zTH, dC):
    # 直線長↑＝差し/捲り寄り(−)、カント↑＝先行/スピード勝負(+)、333短周長＝ライン寄り(−)
    return float(clamp(0.50*zTH - 0.35*zL - 0.30*dC, -1.0, +1.0))


# ==============================
# ★ 風取得ユーティリティ（名前衝突を解消）
# ==============================

# 1) 取得ターゲット時刻を作る（JST基準・tzなしdatetime）
def build_openmeteo_target_dt(jst_date, race_slot: str):
    h = SESSION_HOUR.get(race_slot, 11)
    if isinstance(jst_date, datetime):
        jst_date = jst_date.date()
    try:
        y, m, d = jst_date.year, jst_date.month, jst_date.day
    except Exception:
        dt = pd.to_datetime(str(jst_date))
        y, m, d = dt.year, dt.month, dt.day
    return datetime(y, m, d, h, 0, 0)




# ==============================
# UI
# ==============================
st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)

track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox(
    "競輪場（プリセット）",
    track_names,
    index=track_names.index("川崎") if "川崎" in track_names else 0
)
info = KEIRIN_DATA[track]
st.session_state["track"] = track

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_day = st.sidebar.date_input("日付（風取得用）", value=date.today())

wind_dir = st.sidebar.selectbox(
    "風向", ["無風","左上","上","右上","左","右","左下","下","右下"],
    index=0, key="wind_dir_input"
)

wind_speed_default = st.session_state.get("wind_speed", 3.0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 60.0, float(wind_speed_default), 0.1)

with st.sidebar.expander("🌀 風をAPIで自動取得（Open-Meteo）", expanded=False):
    st.sidebar.caption("基準時刻：モ=8時 / デ=11時 / ナ=18時 / ミ=22時（JST・tzなしで取得）")

    if st.sidebar.button("APIで取得→風速に反映", use_container_width=True):
        info_xy = VELODROME_MASTER.get(track)
        if not info_xy or info_xy.get("lat") is None or info_xy.get("lon") is None:
            st.sidebar.error(f"{track} の座標が未登録です（VELODROME_MASTER に lat/lon を入れてください）")
        else:
            try:
                target = build_openmeteo_target_dt(race_day, race_time)
                data = fetch_openmeteo_hour(info_xy["lat"], info_xy["lon"], target)

                st.session_state["wind_speed"] = round(float(data["speed_ms"]), 2)

                precip = float(data.get("precipitation", 0.0) or 0.0)
                weather_code = data.get("weather_code", None)

                st.session_state["precipitation"] = precip
                st.session_state["weather_code"] = weather_code
                st.session_state["is_wet"] = bool(precip >= 0.3)

                st.sidebar.success(
                    f"{track} {target:%Y-%m-%d %H:%M} "
                    f"風速 {st.session_state['wind_speed']:.1f} m/s "
                    f"降水 {precip:.1f}mm/h "
                    f"（API側と{data['diff_min']:.0f}分ズレ）"
                )
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"取得に失敗：{e}")



straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle      = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length     = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)
st.session_state["bank_length"] = float(bank_length)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox(
    "開催日",
    ["初日", "2日目", "3日目", "4日目", "5日目", "最終日"],
    0
)

DAY_LAP_ADD = {
    "初日": 1,
    "2日目": 2,
    "3日目": 3,
    "4日目": 4,
    "5日目": 5,
    "最終日": 6,
}

eff_laps = int(base_laps) + DAY_LAP_ADD[day_label]

race_class = st.sidebar.selectbox(
    "級別",
    ["Ｓ級", "Ａ級", "Ａ級チャレンジ", "ガールズ", "アドバンス"],
    0
)

is_girls_like = race_class in ("ガールズ", "アドバンス")

# === 会場styleを「得意会場平均」を基準に再定義
zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bank_length)
style_raw = venue_mix(zL, zTH, dC)

# 天候による自動バイアス補正
precip = float(st.session_state.get("precipitation", 0.0) or 0.0)

if precip >= 5.0:
    weather_override = 0.6
elif precip >= 2.0:
    weather_override = 0.4
elif precip >= 0.3:
    weather_override = 0.2
else:
    weather_override = 0.0

manual_override = st.sidebar.slider(
    "会場バイアス補正（−2差し ←→ +2先行）",
    -2.0, 2.0, 0.0, 0.1
)

override = clamp(manual_override + weather_override, -2.0, 2.0)

st.sidebar.caption(
    f"天候自動補正：{weather_override:+.1f} / 最終バイアス補正：{override:+.1f}"
)

style = clamp(style_raw + 0.25 * override, -1.0, +1.0)



CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
    "アドバンス":     {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

DAY_FACTOR = {
    "初日": 1.00,
    "2日目": 1.00,
    "3日目": 0.99,
    "4日目": 0.98,
    "5日目": 0.97,
    "最終日": 0.96,
}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

# ===== 日程・級別・頭数で“周回疲労の効き”を薄くシフト（出力には出さない） =====
DAY_SHIFT = {
    "初日": -0.5,
    "2日目": 0.0,
    "3日目": +0.2,
    "4日目": +0.4,
    "5日目": +0.6,
    "最終日": +0.8,
}
CLASS_SHIFT = {
    "Ｓ級": 0.0,
    "Ａ級": +0.10,
    "Ａ級チャレンジ": +0.20,
    "ガールズ": -0.10,
    "アドバンス": -0.10,
}
HEADCOUNT_SHIFT = {5: -0.20, 6: -0.10, 7: -0.05, 8: 0.0, 9: +0.10}

def fatigue_extra(eff_laps: int, day_label: str, n_cars: int, race_class: str) -> float:
    d = float(DAY_SHIFT.get(day_label, 0.0))
    c = float(CLASS_SHIFT.get(race_class, 0.0))
    h = float(HEADCOUNT_SHIFT.get(int(n_cars), 0.0))
    x = (float(eff_laps) - 2.0) + d + c + h
    return max(0.0, x)

# === PATCH-L200:（以下そのまま） ==========================================
# ...（あなたの last200_bonus 以降は変更なし）

fatigue_value = fatigue_extra(eff_laps, day_label, n_cars, race_class)

globals()["fatigue_value"] = float(fatigue_value)
globals()["fatigue_extra_value"] = float(fatigue_value)

# sidebarの直後あたり（straight_length/style/wind_speedが確定した後）
globals()["straight_length"] = float(straight_length)
globals()["bank_length"]     = float(bank_length)
globals()["bank_angle"]      = float(bank_angle)
globals()["style"]           = float(style)
globals()["wind_speed"]      = float(wind_speed)
globals()["race_class"]      = str(race_class)
globals()["n_cars"]          = int(n_cars)
globals()["day_label"] = str(day_label)
globals()["eff_laps"]  = int(eff_laps)
    


# ==============================
# メイン：入力
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き：統合版）⭐")
st.caption(f"風補正モード: {WIND_MODE}（'speed_only'=風速のみ / 'directional'=向きも薄く考慮）")

# ←★ここに貼る（1回だけ走らせる）
if "_DID_SELF_GREP" not in st.session_state:
    st.session_state["_DID_SELF_GREP"] = True
    _grep_self("KO使用スコア", __file__, context=6)
    _grep_self("KO使用スコア（降順）", __file__, context=6)
    _grep_self("ko_text", __file__, context=6)
# →★ここまで


st.subheader("２０２６/５/２更新")
if "race_no_main" not in st.session_state:
    st.session_state["race_no_main"] = 1
c1, c2, c3 = st.columns([6,2,2])
with c1:
    race_no_input = st.number_input("R", min_value=1, max_value=12, step=1,
                                    value=int(st.session_state["race_no_main"]),
                                    key="race_no_input")
with c2:
    prev_clicked = st.button("◀ 前のR", use_container_width=True)
with c3:
    next_clicked = st.button("次のR ▶", use_container_width=True)
if prev_clicked:
    st.session_state["race_no_main"] = max(1, int(race_no_input) - 1); st.rerun()
elif next_clicked:
    st.session_state["race_no_main"] = min(12, int(race_no_input) + 1); st.rerun()
else:
    st.session_state["race_no_main"] = int(race_no_input)
race_no = int(st.session_state["race_no_main"])

# ==============================
# メイン入力：通常入力 → 反映ボタンで計算用データを固定
# ※スコア計算ロジックは元コードから変更しない
# ==============================

# ライン構成（最大7：単騎も1ライン）
line_inputs_live = [
    st.text_input("ライン1（例：123）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：456）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：789）", key="line_3", max_chars=9),
    st.text_input("ライン4（任意）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
    st.text_input("ライン8（任意）", key="line_8", max_chars=9),
    st.text_input("ライン9（任意）", key="line_9", max_chars=9),
]
n_cars = int(n_cars)
lines_live = [extract_car_list(x, n_cars) for x in line_inputs_live if str(x).strip()]
line_def_live, car_to_group_live = build_line_maps(lines_live)
active_cars_live = sorted({c for lst in lines_live for c in lst}) if lines_live else list(range(1, n_cars+1))

# 5〜9車対応：ライン入力漏れチェック
if len(active_cars_live) != int(n_cars):
    st.warning(
        f"出走数{n_cars}に対して、ライン入力済みは{len(active_cars_live)}車です。"
        " ライン入力漏れを確認してください。"
    )

# ←←← ここに入れる
def input_float_text(label: str, key: str, placeholder: str = ""):
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "":
        return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

# →→→ ここまで

st.subheader("個人データ（直近4か月：回数）")
cols = st.columns(len(active_cars_live))
ratings_live, S_live, H_live, B_live = {}, {}, {}, {}

k_esc_live, k_mak_live, k_sashi_live, k_mark_live = {}, {}, {}, {}
x1_live, x2_live, x3_live, x_out_live = {}, {}, {}, {}

for i, no in enumerate(active_cars_live):
    with cols[i]:
        st.markdown(f"**{no}番**")
        ratings_live[no] = input_float_text("得点（空欄可）", key=f"pt_{no}", placeholder="例: 55.0")
        S_live[no] = st.number_input("S", 0, 99, 0, key=f"s_{no}")
        H_live[no] = st.number_input("H", 0, 99, 0, key=f"h_{no}")
        B_live[no] = st.number_input("B", 0, 99, 0, key=f"b_{no}")
        k_esc_live[no]   = st.number_input("逃", 0, 99, 0, key=f"ke_{no}")
        k_mak_live[no]   = st.number_input("捲", 0, 99, 0, key=f"km_{no}")
        k_sashi_live[no] = st.number_input("差", 0, 99, 0, key=f"ks_{no}")
        k_mark_live[no]  = st.number_input("マ", 0, 99, 0, key=f"kk_{no}")
        x1_live[no]  = st.number_input("1着", 0, 99, 0, key=f"x1_{no}")
        x2_live[no]  = st.number_input("2着", 0, 99, 0, key=f"x2_{no}")
        x3_live[no]  = st.number_input("3着", 0, 99, 0, key=f"x3_{no}")
        x_out_live[no]= st.number_input("着外", 0, 99, 0, key=f"xo_{no}")

# =====================================================
# コメントチェック表
#   前検コメントを見て手動チェック
#   自力：自力 / 自力自在 / 自力基本 / 自分で / 前で 等
#   番手：○○君 / ○○へ / 任せる / 近畿勢 等
#   競り：競り対象の車番にチェック
# =====================================================
st.subheader("コメントチェック")

jiryoku_comment_live = {}
target_comment_live = {}
seri_comment_live = {}

comment_cols = st.columns(len(active_cars_live))

for i, no in enumerate(active_cars_live):
    no = int(no)
    with comment_cols[i]:
        st.markdown(f"**{no}番**")

        jiryoku_comment_live[no] = st.checkbox(
            "自力",
            value=False,
            key=f"jiryoku_comment_r{race_no}_{no}"
        )

        target_comment_live[no] = st.checkbox(
            "番手",
            value=False,
            key=f"target_comment_r{race_no}_{no}"
        )

        seri_comment_live[no] = st.checkbox(
            "競り",
            value=False,
            key=f"seri_comment_r{race_no}_{no}"
        )

st.markdown("---")

apply_input = st.button(
    "入力を反映して計算する",
    type="primary",
    use_container_width=True,
    key="apply_input_main"
)

if apply_input:
    st.session_state["race_snapshot"] = {
        "line_inputs": list(line_inputs_live),
        "lines": [list(x) for x in lines_live],
        "line_def": {g: list(mem) for g, mem in line_def_live.items()},
        "car_to_group": dict(car_to_group_live),
        "active_cars": list(active_cars_live),

        "ratings": dict(ratings_live),
        "S": dict(S_live),
        "H": dict(H_live),
        "B": dict(B_live),

        "k_esc": dict(k_esc_live),
        "k_mak": dict(k_mak_live),
        "k_sashi": dict(k_sashi_live),
        "k_mark": dict(k_mark_live),

        "x1": dict(x1_live),
        "x2": dict(x2_live),
        "x3": dict(x3_live),
        "x_out": dict(x_out_live),

        "jiryoku_comment": dict(jiryoku_comment_live),
        "target_comment": dict(target_comment_live),
        "seri_comment": dict(seri_comment_live),
    }

snapshot = st.session_state.get("race_snapshot")

if snapshot is None:
    st.info("入力後、『入力を反映して計算する』を押すと本計算します。")
    st.stop()

# ==============================
# ここから下は、反映済みデータだけで計算する
# ==============================

line_inputs = snapshot["line_inputs"]
lines = snapshot["lines"]
line_def = snapshot["line_def"]
car_to_group = snapshot["car_to_group"]
active_cars = snapshot["active_cars"]

ratings = snapshot["ratings"]
S = snapshot["S"]
H = snapshot["H"]
B = snapshot["B"]

k_esc = snapshot["k_esc"]
k_mak = snapshot["k_mak"]
k_sashi = snapshot["k_sashi"]
k_mark = snapshot["k_mark"]

x1 = snapshot["x1"]
x2 = snapshot["x2"]
x3 = snapshot["x3"]
x_out = snapshot["x_out"]

jiryoku_comment = snapshot.get("jiryoku_comment", {})
target_comment = snapshot.get("target_comment", {})
seri_comment = snapshot.get("seri_comment", {})

globals()["jiryoku_comment"] = jiryoku_comment
globals()["target_comment"] = target_comment
globals()["seri_comment"] = seri_comment

st.caption(
    "反映済みデータで計算中："
    f"車番={active_cars} ／ "
    f"ライン={'　'.join(''.join(map(str, ln)) for ln in lines) if lines else 'なし'}"
)

# 反映済みデータの整合チェック
if len(active_cars) != int(n_cars):
    st.error(
        f"出走数{n_cars}に対して、反映済みラインは{len(active_cars)}車です。"
        f" 反映済み車番: {active_cars}"
    )
    st.stop()

dup_check = []
for lst in lines:
    dup_check.extend(lst)

dups = sorted([x for x in set(dup_check) if dup_check.count(x) >= 2])

if dups:
    st.error(f"同じ車番が複数ラインに入っています: {dups}")
    st.stop()

ratings_val = {no: (float(ratings[no]) if ratings[no] is not None else 55.0) for no in active_cars}

# =====================================================
# 混戦度判定：競走得点1位と2位の差
# ※ active_cars / ratings_val が確定した後で実行する
# =====================================================
race_compact = calc_race_compactness(ratings_val, active_cars)
race_compact_label = race_compact.get("label", "未判定")
race_compact_gap = race_compact.get("top_gap", None)

globals()["race_compact_label"] = race_compact_label
globals()["race_compact_gap"] = race_compact_gap
globals()["race_compact"] = race_compact

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)
home_line_order = make_home_line_order(line_def, H, B, active_cars)
home_line_text = format_home_line_order(line_def, home_line_order)

home_top_gid = home_line_order[0] if home_line_order else None

# H主導ライン判定
# Hスコアが低すぎる場合は「主導なし」とする
home_top_score = float(home_line_scores.get(home_top_gid, 0.0)) if home_top_gid is not None else 0.0

if home_top_gid is not None and home_top_score >= 1.0:
    home_top_line = format_home_line_order(line_def, [home_top_gid])
else:
    home_top_line = "主導なし"



# 1着・2着の縮約（級別×会場の事前分布を混ぜる）
def prior_by_class(cls, style_adj):
    if "ガール" in cls: p1,p2 = 0.18,0.24
    elif "Ｓ級" in cls: p1,p2 = 0.22,0.26
    elif "チャレンジ" in cls: p1,p2 = 0.18,0.22
    else: p1,p2 = 0.20,0.25
    p1 += 0.010*style_adj; p2 -= 0.005*style_adj
    return clamp(p1,0.05,0.60), clamp(p2,0.05,0.60)

def n0_by_n(n):
    if n<=6: return 12
    if n<=14: return 8
    if n<=29: return 5
    return 3

# === 1〜3着＋着外を “ちゃんと” Form に反映する版（ここだけ置換） ===
p1_eff, p2_eff, p3_eff, pout_eff = {}, {}, {}, {}

for no in active_cars:
    n = x1[no] + x2[no] + x3[no] + x_out[no]

    # 既存：クラス×脚質の prior（あなたの関数をそのまま使う）
    p1_prior, p2_prior = prior_by_class(race_class, style)

    # 追加：3着＆着外の prior（まずは固定で安全運用）
    p3_prior   = 0.10
    pout_prior = 0.55

    n0 = n0_by_n(n)

    if n == 0:
        p1_eff[no], p2_eff[no] = p1_prior, p2_prior
        p3_eff[no]             = p3_prior
        pout_eff[no]           = pout_prior
    else:
        p1_eff[no]  = clamp((x1[no]    + n0*p1_prior ) / (n + n0), 0.0, 0.40)
        p2_eff[no]  = clamp((x2[no]    + n0*p2_prior ) / (n + n0), 0.0, 0.50)
        p3_eff[no]  = clamp((x3[no]    + n0*p3_prior ) / (n + n0), 0.0, 0.55)
        pout_eff[no]= clamp((x_out[no] + n0*pout_prior) / (n + n0), 0.0, 0.95)

    # 合計が暴れない安全弁（1-3着を優先して整える）
    s123 = p1_eff[no] + p2_eff[no] + p3_eff[no]
    if s123 > 0.95:
        scale = 0.95 / s123
        p1_eff[no] *= scale
        p2_eff[no] *= scale
        p3_eff[no] *= scale

    pout_eff[no] = clamp(1.0 - (p1_eff[no] + p2_eff[no] + p3_eff[no]), 0.0, 0.95)

# ★Form：1〜3着を評価、着外は減点（ここが効く）
Form = {
    no: (3.0*p1_eff[no] + 2.0*p2_eff[no] + 1.0*p3_eff[no] - 1.2*pout_eff[no])
    for no in active_cars
}

# === Form 偏差値化（平均50, SD10）
form_list = [Form[n] for n in active_cars]
form_T, mu_form, sd_form, _ = t_score_from_finite(np.array(form_list))
form_T_map = {n: float(form_T[i]) for i, n in enumerate(active_cars)}


# --- 脚質プロフィール（会場適性：得意会場平均基準のstyleを掛ける）
prof_base, prof_escape, prof_sashi, prof_oikomi = {}, {}, {}, {}
for no in active_cars:
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark = 0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    prof_escape[no]=esc; prof_sashi[no]=sashi; prof_oikomi[no]=mark
    base = esc*BASE_BY_KAKU["逃"] + mak*BASE_BY_KAKU["捲"] + sashi*BASE_BY_KAKU["差"] + mark*BASE_BY_KAKU["マ"]
    vmix = style
    venue_bonus = 0.06 * vmix * ( +1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark )
    prof_base[no] = base + clamp(venue_bonus, -0.06, +0.06)

# ==============================
# level_rating_scale 保険定義
# ==============================
if "level_rating_scale" not in globals():
    level_rating_scale = 1.0

# ======== 個人補正（得点/脚質上位/着順分布） ========
ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i,no in enumerate(ratings_sorted)}
def tenscore_bonus(no):
    r = ratings_rank[no]
    top_n = min(3, len(active_cars))
    bottom_n = min(3, len(active_cars))
    if r <= top_n: return +0.03
    if r >= len(active_cars)-bottom_n+1: return -0.02
    return 0.0
def topk_bonus(k_dict, topn=3, val=0.02):
    order = sorted(k_dict.items(), key=lambda x:(x[1], -x[0]), reverse=True)
    grant = set([no for i,(no,v) in enumerate(order) if i<topn])
    return {no:(val if no in grant else 0.0) for no in k_dict}
esc_bonus   = topk_bonus(k_esc,   topn=3, val=0.02)
mak_bonus   = topk_bonus(k_mak,   topn=3, val=0.02)
sashi_bonus = topk_bonus(k_sashi, topn=3, val=0.015)
mark_bonus  = topk_bonus(k_mark,  topn=3, val=0.01)
def finish_bonus(no):
    tot = x1[no]+x2[no]+x3[no]+x_out[no]
    if tot == 0: return 0.0
    in3 = (x1[no]+x2[no]+x3[no]) / tot
    out = x_out[no] / tot
    bonus = 0.0
    if in3 > 0.50: bonus += 0.03
    if out > 0.70: bonus -= 0.03
    if out < 0.40: bonus += 0.02
    return bonus
extra_bonus = {}
for no in active_cars:
    total = (tenscore_bonus(no) +
             esc_bonus.get(no,0.0) + mak_bonus.get(no,0.0) +
             sashi_bonus.get(no,0.0) + mark_bonus.get(no,0.0) +
             finish_bonus(no))
    extra_bonus[no] = clamp(total, -0.10, +0.10)

# ===== 会場個性を“個人スコア”に浸透：bank系補正（差し替え案） =====

def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi, bank_length=None):
    pe = float(prof_escape or 0.0)
    ps = float(prof_sashi  or 0.0)

    # bank_lengthが渡っていない場合の扱いを決める（例：0.0扱い or venue既定値）
    bl = float(bank_length or 0.0)

    zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bl)

    base = clamp(0.06*zTH - 0.05*zL - 0.03*dC, -0.08, +0.08)
    out  = base * pe - 0.5 * base * ps
    return round(out, 3)


def bank_length_adjust(bank_length, prof_oikomi):
    po = float(prof_oikomi or 0.0)
    L  = float(bank_length or 0.0)
    dC = (+0.4 if L >= 480 else 0.0 if L >= 380 else -0.4)

    out = 0.03 * (-dC) * po
    return round(out, 3)



# --- 安定度（着順分布）をT本体に入れるための重み（強化版） ---
STAB_W_IN3  = 0.18   # 3着内の寄与
STAB_W_OUT  = 0.22   # 着外のペナルティ
STAB_W_LOWN = 0.06   # サンプル不足ペナルティ
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45

def stability_score(no: int) -> float:
    n1 = x1.get(no, 0); n2 = x2.get(no, 0); n3 = x3.get(no, 0); nOut = x_out.get(no, 0)
    n  = n1 + n2 + n3 + nOut
    if n <= 0:
        return 0.0
    # 少サンプル縮約（この関数内で完結）
    if n <= 6:    n0 = 12
    elif n <= 14: n0 = 8
    elif n <= 29: n0 = 5
    else:         n0 = 3

    in3  = (n1 + n2 + n3 + n0*STAB_PRIOR_IN3) / (n + n0)
    out_ = (nOut          + n0*STAB_PRIOR_OUT) / (n + n0)

    bonus = 0.0
    bonus += STAB_W_IN3 * (in3 - STAB_PRIOR_IN3) * 2.0
    bonus -= STAB_W_OUT * (out_ - STAB_PRIOR_OUT) * 2.0

    if n < 10:
        bonus -= STAB_W_LOWN * (10 - n) / 10.0

    # キャップ：nに応じて段階的に広げる（±0.35〜±0.45）
    cap = 0.35
    if n >= 15: cap = 0.45
    elif n >= 10: cap = 0.40

    return clamp(bonus, -cap, +cap)

# ===== SBなし合計（環境補正 + 得点微補正 + 個人補正 + 周回疲労 + 安定度） =====
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}


# ==============================
# L200_RAW（観測用）を先に作る：ここでは laps_adj 等は一切計算しない
# ==============================
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir",   wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

L200_RAW = {}
for no in active_cars:
    role = role_in_line(no, line_def)

    # --- L200（残脚）生値を計算：ENV合計には“入れない”観測用 ---
    l200 = l200_adjust(
        role=role,
        straight_length=straight_length,
        bank_length=bank_length,
        race_class=race_class,
        prof_escape=float(prof_escape[no]),
        prof_sashi=float(prof_sashi[no]),
        prof_oikomi=float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)  # 雨トグル未実装なら False のまま
    )
    L200_RAW[int(no)] = float(l200)


# ==============================
# rows（本体計算）ここで laps_adj を計算して使う（2重計算しない）
# ==============================
rows = []

# H：最終ホーム地力補正マップ
H_Z = calc_h_score_map(H, active_cars)

_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir", wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

for no in active_cars:
    no = int(no)
    role = role_in_line(no, line_def)

    # =====================================================
    # 周回疲労（DAY×頭数×級別を反映）
    # =====================================================
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    extra = min(extra, 3.0)   # 応急上限（暴走止め）

    fatigue_scale = (
        1.0  if race_class == "Ｓ級" else
        1.1  if race_class == "Ａ級" else
        1.2  if race_class == "Ａ級チャレンジ" else
        1.05
    )

    # =====================================================
    # 周回疲労補正
    # =====================================================
    laps_adj = (
        -0.10 * extra * (1.0 if float(prof_escape[no]) > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if float(prof_oikomi[no]) > 0.4 else 0.0)
    ) * fatigue_scale

    # ガールズは周回疲労を弱める
    if is_girls_like:
        laps_adj *= 0.3

    # 周回疲労の暴走防止
    laps_adj = clamp(laps_adj, -0.22, 0.18)

    # =====================================================
    # コメント補正
    #   自力：本人をプラス補正
    #   番手：本人ではなく、前の自力先頭をライン連動で格上げ
    #   競り：競り対象者を減点
    # =====================================================
    jiryoku_comment_map = globals().get("jiryoku_comment", {})
    target_comment_map  = globals().get("target_comment", {})
    seri_comment_map    = globals().get("seri_comment", {})

    is_jiryoku_comment = bool(jiryoku_comment_map.get(int(no), False))
    is_seri_comment    = bool(seri_comment_map.get(int(no), False))

        # -----------------------------------------------------
    # 自力コメント補正
    # -----------------------------------------------------
    jiryoku_comment_bonus = 0.0

    if is_jiryoku_comment:
        # 基本加点
        jiryoku_comment_bonus = 0.120

        # ライン先頭の自力コメントは、実際に動く役割なので追加
        if role == "head":
            jiryoku_comment_bonus += 0.020

        # H主導ラインの先頭なら、さらに追加
        try:
            h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
            if h_line and int(h_line[0]) == int(no):
                jiryoku_comment_bonus += 0.030
        except Exception:
            pass

        # ガールズはラインがないため少し薄め
        if is_girls_like:
            jiryoku_comment_bonus *= 0.60

    jiryoku_comment_bonus = clamp(jiryoku_comment_bonus, 0.0, 0.170)
    # -----------------------------------------------------
    # ライン連動補正
    #   後ろの選手が「番手・目標」チェックありなら、
    #   その前のライン先頭を少し格上げする。
    #   例：42で2が「小原君」なら、4を少し救う。
    # -----------------------------------------------------
    line_cushion_bonus = 0.0

    try:
        gid = car_to_group.get(int(no), None)
        members = line_def.get(gid, []) if gid is not None else []

        # 自分がそのラインの先頭かどうか
        is_line_head = bool(members and int(members[0]) == int(no))

        if is_line_head:
            behind_members = [int(x) for x in members[1:]]

            has_target_behind = any(
                bool(target_comment_map.get(int(x), False))
                for x in behind_members
            )

            if has_target_behind:
                # 番手・後位が前を指名しているなら、先頭車を少し救う
                line_cushion_bonus = 0.040

                # H主導ラインの先頭なら、ライン成立度を少し上乗せ
                try:
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    if h_line and int(h_line[0]) == int(no):
                        line_cushion_bonus += 0.020
                except Exception:
                    pass

    except Exception:
        line_cushion_bonus = 0.0

    line_cushion_bonus = clamp(line_cushion_bonus, 0.0, 0.060)

    # -----------------------------------------------------
    # 競り補正
    #   ライン入力は崩さず、競り対象者だけを減点する。
    #   例：12345 67 のまま、2・3に競りチェック。
    # -----------------------------------------------------
    seri_penalty = 0.0

    if is_seri_comment:
        seri_penalty = -0.100

        # ガールズは基本的に競りの意味が薄いので弱め
        if is_girls_like:
            seri_penalty *= 0.50

    seri_penalty = clamp(seri_penalty, -0.120, 0.0)

    # =====================================================
    # 環境・個人補正（既存）
    # =====================================================
    wind     = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    bank_b   = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no], bank_length)
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv    = extra_bonus.get(no, 0.0)
    stab     = stability_score(no)  # 安定度
    h_bonus  = h_home_bonus(no, role, H_Z)

    l200 = l200_adjust(
        role, straight_length, bank_length, race_class,
        float(prof_escape[no]), float(prof_sashi[no]), float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)
    )

    # =====================================================
    # 合計スコア
    # =====================================================
    total_raw = (
        prof_base[no]
        + wind
        + cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0)
        + bank_b
        + length_b
        + laps_adj
        + indiv
        + stab
        + h_bonus
        + l200
        + jiryoku_comment_bonus
        + line_cushion_bonus
        + seri_penalty
    )

    rows.append([
        no, role,
        round(prof_base[no], 3),
        round(wind, 3),
        round(cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0), 3),
        round(bank_b, 3),
        round(length_b, 3),
        round(laps_adj, 3),
        round(indiv, 3),
        round(stab, 3),
        round(h_bonus, 3),
        round(l200, 3),
        round(jiryoku_comment_bonus, 3),
        round(line_cushion_bonus, 3),
        round(seri_penalty, 3),
        float(total_raw)
    ])

df = pd.DataFrame(rows, columns=[
    "車番", "役割", "脚質基準(会場)", "風補正", "得点補正", "バンク補正",
    "周長補正", "周回補正", "個人補正", "安定度", "H補正", "ラスト200",
    "自力コメント補正", "ライン連動補正", "競り補正",
    "合計_SBなし_raw",
])

# ===== [PATCH] dfの型を確定させ、SBなし母集団(v_wo/v_final)を必ず作る =====
# 1) dfが空のときも落とさない
if df is None or len(df) == 0:
    st.warning("DEBUG: df（SBなし内訳）が空です。rowsが生成されていない可能性。")
    v_wo = {int(no): 0.0 for no in active_cars}
else:
    # 2) 車番を必ずintにする（★最重要：ここがズレると全部emptyになる）
    df["車番"] = df["車番"].astype(int)

    # 3) v_wo を df から必ず生成（全車キー保証）
    v_wo = {int(r["車番"]): float(r["合計_SBなし_raw"]) for _, r in df.iterrows()}
    for no in active_cars:
        ino = int(no)
        if ino not in v_wo:
            v_wo[ino] = 0.0

# 4) v_final は最低でも v_wo を引き継ぐ（KOが走らない/空でも落ちない）
v_final = dict(v_wo)

# 5) df_sorted_pure をここで確定（アンカー選定が安定）
df_sorted_pure = pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [float(v_final[int(k)]) for k in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)


    


# === ここは df = pd.DataFrame(...) の直後に貼るだけ ===

# --- fallback: note_sections が無い環境でも落ちないように ---
ns = globals().get("note_sections", None)
if not isinstance(ns, list):
    ns = []
    globals()["note_sections"] = ns
note_sections = ns


# ❶ バンク分類を“みなし直線/周長”から決定（33 / 400 / 500）
def _bank_str_from_lengths(bank_length: float) -> str:
    try:
        bl = float(bank_length)
    except:
        bl = 400.0
    if bl <= 340.0:   # 333系
        return "33"
    elif bl >= 480.0: # 500系
        return "500"
    return "400"

# ❷ 会場の“有利脚質”セット
def _favorable_styles(bank_str: str) -> set[str]:
    if bank_str == "33":   # 33＝先行系・ライン寄り
        return {"逃げ", "マーク"}
    if bank_str == "500":  # 500＝差し・マーク寄り
        return {"差し", "マーク"}
    return {"まくり", "差し"}  # 既定=400

# ❸ 役割の日本語化（lineの並びから）
def _role_jp(no: int, line_def: dict) -> str:
    r = role_in_line(no, line_def)
    return {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(r, "単騎")


# ❹ 入力の“逃/捲/差/マ”から、その選手の実脚質を決定（同点時はライン位置でブレない決め方）
def _dominant_style(no: int) -> str:
    vec = [("逃げ", k_esc.get(no,0)), ("まくり", k_mak.get(no,0)),
           ("差し", k_sashi.get(no,0)), ("マーク", k_mark.get(no,0))]
    m = max(v for _,v in vec)
    cand = [s for s,v in vec if v == m and m > 0]
    if cand:
        # タイブレーク：先頭>番手>三番手>単騎 を優先（先行気味→差し→マークの順）
        pr = {"先頭":3,"番手":2,"三番手":1,"単騎":0}
        role = role_in_line(no, line_def)
        role_pr = {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(role,"単騎")
        if "逃げ" in cand: return "逃げ"
        # 残りはライン位置で“差し”優先、その次に“マーク”
        if "差し" in cand and pr.get(role_pr,0) >= 2: return "差し"
        if "マーク" in cand: return "マーク"
        return cand[0]
    # 出走履歴ゼロなら位置で決める
    role = role_in_line(no, line_def)
    return {"head":"逃げ","second":"差し","thirdplus":"マーク","single":"まくり"}.get(role,"まくり")

# ❺ Rider 構造体（このファイル上部で既に宣言済みなら再定義不要）
from dataclasses import dataclass
@dataclass
class Rider:
    num: int; hensa: float; line_id: int; role: str; style: str

# ❻ 偏差値（Tスコア）を “合計_SBなし_raw” から作る（なければ Form で代用）
# ❻ 安定版：偏差値（Tスコア）を安全に作る
def _hensa_map_from_df(df: pd.DataFrame) -> dict[int,float]:
    col = "合計_SBなし_raw" if "合計_SBなし_raw" in df.columns else None

    # 生値ベクトルを取る（欠損があればフォールバックして補完）
    base = []
    for no in active_cars:
        try:
            v = float(df.loc[df["車番"]==no, col].values[0]) if col else float(form_T_map[no])
        except:
            v = float(form_T_map[no])  # fallback（=従来 Form 偏差値）
        base.append(v)

    base = np.array(base, dtype=float)

    # === 分散チェック：標準偏差が小さすぎる場合の暴走回避 ===
    sd = np.std(base)
    if sd < 1e-6:   # ← 安定化の本丸
        # 全員ほぼ同じ → 差が「無い」ので偏差値の差も付けない
        return {no: 50.0 for no in active_cars}

    # 通常の偏差値化
    T = 50 + 10 * (base - np.mean(base)) / sd

    # 浮動誤差対策で丸め
    T = np.clip(T, 20, 80)

    return {no: float(T[i]) for i,no in enumerate(active_cars)}


# ❼ RIDERS を“実データ”で構築（脚質は ❹、偏差値は ❻）
bank_str = _bank_str_from_lengths(bank_length)
hensa_map = _hensa_map_from_df(df)
RIDERS = []
for no in active_cars:
    # ラインIDは“そのラインの先頭車番”を代表IDに
    gid = None
    for g, mem in line_def.items():
        if no in mem:
            gid = mem[0]; break
    if gid is None: gid = no
    RIDERS.append(
        Rider(
            num=int(no),
            hensa=float(hensa_map[no]),
            line_id=int(gid),
            role=_role_jp(no, line_def),
            style=_dominant_style(no),
        )
    )

# ❽ フォーメーション（本命−2−全）：1列目=有利脚質内の偏差値最大
def _pick_axis(riders: list[Rider], bank_str: str) -> Rider:
    fav = _favorable_styles(bank_str)
    cand = [r for r in riders if r.style in fav]
    if not cand:
        raise ValueError(f"有利脚質{sorted(fav)}に該当0（bank={bank_str} / style誤りの可能性）")
    return max(cand, key=lambda r: r.hensa)

def _role_priority(bank_str: str) -> dict[str,int]:
    return ({"マーク":3,"番手":2,"三番手":1,"先頭":0} if bank_str=="33"
            else {"番手":3,"マーク":2,"三番手":1,"先頭":0})

from typing import Optional, List

def _pick_support(riders: List["Rider"], first: "Rider", bank_str: str) -> Optional["Rider"]:
    pr = _role_priority(bank_str)
    same = [r for r in riders if r.line_id==first.line_id and r.num!=first.num]
    if not same:
        return None
    same.sort(key=lambda r: (pr.get(r.role,0), r.hensa), reverse=True)
    return same[0]


# 印（◎→▲→偏差値補完）
def _read_marks_idmap() -> dict[int,str]:
    rm = globals().get("result_marks") or globals().get("marks") or {}
    out={}
    if isinstance(rm, dict):
        if any(isinstance(k,int) or (isinstance(k,str) and k.isdigit()) for k in rm.keys()):
            for k,v in rm.items():
                try: out[int(k)] = ("○" if str(v) in ("○","〇") else str(v))
                except: pass
        else:
            for sym,vid in rm.items():
                try: out[int(vid)] = ("○" if str(sym) in ("○","〇") else str(sym))
                except: pass
    return out

def _pick_partner(riders: list[Rider], used: set[int]) -> int|None:
    id2sym = _read_marks_idmap()
    for want in ("◎","▲"):
        t = next((i for i,s in id2sym.items() if i not in used and s==want), None)
        if t is not None: return t
    # 補完：偏差値上位
    rest = sorted([r for r in riders if r.num not in used], key=lambda r: r.hensa, reverse=True)
    return rest[0].num if rest else None

def make_trio_formation_final(riders: list[Rider], bank_str: str) -> str:
    first = _pick_axis(riders, bank_str)
    support = _pick_support(riders, first, bank_str)
    used = {first.num} | ({support.num} if support else set())
    partner = _pick_partner(riders, used)
    second = []
    if support: second.append(support.num)
    if partner is not None: second.append(partner)
    if len(second) < 2:
        # 2車に満たなければ偏差値補完
        rest = sorted([r.num for r in riders if r.num not in ({first.num}|set(second))],
                      key=lambda n: next(rr.hensa for rr in riders if rr.num==n),
                      reverse=True)
        if rest: second.append(rest[0])
    second = sorted(set(second))[:2]
    return f"三連複フォーメーション：{first.num}－{','.join(map(str, second))}－全"


mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# --- SBなし(母集団) を df から「全車ぶん必ず」作る（None防止） ---
sb_map = {int(r["車番"]): float(r.get("合計_SBなし", 0.0)) for _, r in df.iterrows()}

# df が空 / sb_map が空のときは、全車0で母集団を作る（5車・欠番・SB未入力でも止めない）
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}
    

# === [PATCH-A] 安定度をENVから分離し、各柱をレース内z化（SD固定） ===
SD_FORM = 0.28
SD_ENV  = 0.20
SD_STAB = 0.12
SD_L200 = float(globals().get("SD_L200", 0.22))  # ← 追加。まず0.22〜0.30で様子見

# 安定度（raw）と、ENVのベース（= 合計_SBなし_raw から安定度だけ除いたもの）
STAB_RAW = {int(df.loc[i, "車番"]): float(df.loc[i, "安定度"]) for i in df.index}
ENV_BASE = {
    int(df.loc[i, "車番"]): (
        float(df.loc[i, "合計_SBなし_raw"])
        - float(df.loc[i, "安定度"])
        - float(df.loc[i, "ラスト200"])
    )
    for i in df.index
}

# ENV → z
_env_arr = np.array([float(ENV_BASE.get(n, np.nan)) for n in active_cars], dtype=float)
_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask])); sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0
_den_env = (sd_env if sd_env > 1e-12 else 1.0)
ENV_Z = {int(n): (float(ENV_BASE.get(n, mu_env)) - mu_env) / _den_env for n in active_cars}

# FORM（すでに form_T_map は作ってある前提） → z
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}

# STAB（安定度 raw） → z
_stab_arr = np.array([float(STAB_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m2 = np.isfinite(_stab_arr)
if int(_m2.sum()) >= 2:
    mu_st = float(np.mean(_stab_arr[_m2])); sd_st = float(np.std(_stab_arr[_m2]))
else:
    mu_st, sd_st = 0.0, 1.0
_den_st = (sd_st if sd_st > 1e-12 else 1.0)
STAB_Z = {int(n): (float(STAB_RAW.get(n, mu_st)) - mu_st) / _den_st for n in active_cars}

# L200（残脚）→ z
_l200_arr = np.array([float(L200_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m3 = np.isfinite(_l200_arr)
if int(_m3.sum()) >= 2:
    mu_l2 = float(np.mean(_l200_arr[_m3])); sd_l2 = float(np.std(_l200_arr[_m3]))
else:
    mu_l2, sd_l2 = 0.0, 1.0
_den_l2 = (sd_l2 if sd_l2 > 1e-12 else 1.0)
L200_Z = {int(n): (float(L200_RAW.get(n, mu_l2)) - mu_l2) / _den_l2 for n in active_cars}

# ===== KO方式（印に混ぜず：展開・ケンで利用） =====

# 0) SBなし(母集団) を df から確実に作る（全車）
sb_map = {int(k): float(v) for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))}

# ★必須：dfが空でも全車0で母集団を作る
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}

# 1) key 欠損チェック
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")
    # st.stop()

# 2) 値が None/NaN チェック
bad = [
    int(n) for n in active_cars
    if (int(n) in sb_map) and (
        sb_map[int(n)] is None or
        (isinstance(sb_map[int(n)], float) and np.isnan(sb_map[int(n)]))
    )
]
if bad:
    st.error(f"SBなし(母集団) の値が None/NaN: {bad} / values={[sb_map[int(n)] for n in bad]}")
    # st.stop()

# 3) KO入力に使う母集団（全車）
v_wo = dict(sb_map)

# 4) 以降 KO
_is_girls = is_girls_like
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale_raw = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale
KO_SCALE_MAX = 0.45
ko_scale = min(ko_scale_raw, KO_SCALE_MAX)

if ko_scale > 0.0 and line_def and len(line_def) >= 1 and v_wo:
    # --- KO順序（_ko_order が落ちる/不正でも必ずフォールバックで作る） ---
    try:
        ko_order = _ko_order(
            v_wo, line_def, S, B,
            line_factor=line_factor_eff,
            gap_delta=KO_GAP_DELTA
        )
    except Exception as e:
        # Streamlitで原因を見たいならコメント解除
        # st.warning(f"_ko_order fallback: {type(e).__name__}: {e}")
        ko_order = None

    # ★重要：ko_order が None/空/欠損でも「全車」を必ず含める
    ko_order = [int(c) for c in (ko_order or []) if int(c) in v_wo]
    rest = [int(c) for c in v_wo.keys() if int(c) not in set(ko_order)]
    rest = sorted(rest, key=lambda c: float(v_wo[int(c)]), reverse=True)
    ko_order = ko_order + rest  # ← 全車を必ず含める（ここが最重要）

    # ここ以降は ko_order が必ず全車になるので安全
    vals = [float(v_wo[c]) for c in v_wo.keys()]
    mu0  = float(np.mean(vals))
    sd0  = float(np.std(vals) + 1e-12)
    KO_STEP_SIGMA_LOCAL = max(0.25, KO_STEP_SIGMA * 0.7)
    step = KO_STEP_SIGMA_LOCAL * sd0
    # ★new_scores は「全車のベース」から開始して KO で上書き
    new_scores = dict(v_wo)

    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * float(v_wo[int(car)]) + ko_scale * (
            mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step
        )
        new_scores[int(car)] = float(blended)

    v_final = dict(new_scores)

else:
    # KOしない時も「全車保持」
    if v_wo:
        ko_order = sorted(v_wo.keys(), key=lambda c: float(v_wo[c]), reverse=True)
        v_final = dict(v_wo)
    else:
        ko_order = []
        v_final = {}

# --- 純SBなしランキング（KOまで／格上げ前）
df_sorted_pure = (pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [round(float(v_final[int(c)]), 6) for c in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True))


# ===== 印用（既存の安全弁を維持） =====
FINISH_WEIGHT   = globals().get("FINISH_WEIGHT", 6.0)
FINISH_WEIGHT_G = globals().get("FINISH_WEIGHT_G", 3.0)
POS_BONUS  = globals().get("POS_BONUS", {0: 0.0, 1: -0.6, 2: -0.9, 3: -1.2, 4: -1.4})
POS_WEIGHT = globals().get("POS_WEIGHT", 1.0)
SMALL_Z_RATING = globals().get("SMALL_Z_RATING", 0.01)
FINISH_CLIP = globals().get("FINISH_CLIP", 4.0)
TIE_EPSILON  = globals().get("TIE_EPSILON", 0.8)

# --- p2のZ化など（従来どおり） ---
p2_list = [float(p2_eff.get(n, 0.0)) for n in active_cars]
if len(p2_list) >= 1:
    mu_p2  = float(np.mean(p2_list))
    sd_p2  = float(np.std(p2_list) + 1e-12)
else:
    mu_p2, sd_p2 = 0.0, 1.0
p2z_map = {n: (float(p2_eff.get(n, 0.0)) - mu_p2) / sd_p2 for n in active_cars}
p1_eff_safe = {n: float(p1_eff.get(n, 0.0)) if 'p1_eff' in globals() and p1_eff is not None else 0.0 for n in active_cars}
p2only_map = {n: max(0.0, float(p2_eff.get(n, 0.0)) - float(p1_eff_safe.get(n, 0.0))) for n in active_cars}
zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
zt_map = {n: float(zt[i]) for i, n in enumerate(active_cars)} if active_cars else {}


# === [PATCH-1] ENV/FORM をレース内で z 化し、目標SDを掛ける（anchor_score の前に置く） ===
SD_FORM = 0.28   # Balanced 既定
SD_ENV  = 0.20

# ENV = v_final（風・会場・周回疲労・個人補正・安定度 等を含む“Form以外”）
# ENV = v_final を int キー前提に揃える
_env_arr = np.array([float(v_final.get(int(n), np.nan)) for n in active_cars], dtype=float)

_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask]))
    sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0

_den = sd_env if sd_env > 1e-12 else 1.0
ENV_Z = {int(n): (float(v_final.get(int(n), mu_env)) - mu_env) / _den for n in active_cars}


# FORM = form_T_map（T=50, SD=10）→ z 化
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}


# --- ここで必ず定義してから使う（NameError防止） ---
line_sb_enable = bool(globals().get("line_sb_enable", (race_class != "ガールズ")))

def _pos_idx(no: int) -> int:
    g = car_to_group.get(no)
    if g is None or g not in line_def:
        return 4  # 単騎/不明は最後方（POS_BONUS[4]）

    grp = line_def[g]  # 例: [5,2,6] みたいな並び
    try:
        return max(0, grp.index(no))
    except ValueError:
        return 4  # グループに居ないなら最後方扱い


bonus_init, _ = compute_lineSB_bonus(
    line_def, S, B,
    line_factor=line_factor_eff,
    exclude=None, cap=cap_SB_eff,
    enable=line_sb_enable
)

def anchor_score(no: int) -> float:
    role = role_in_line(no, line_def)
    sb = float(
        bonus_init.get(car_to_group.get(no, None), 0.0)
        * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    )
    pos_term = (POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)) if line_sb_enable else 0.0
    env_term  = SD_ENV  * float(ENV_Z.get(int(no), 0.0))
    form_term = SD_FORM * float(FORM_Z.get(int(no), 0.0))
    stab_term = SD_STAB * float(STAB_Z.get(int(no), 0.0))
    l200_term = SD_L200 * float(L200_Z.get(int(no), 0.0))
    tiny      = SMALL_Z_RATING * float(zt_map.get(int(no), 0.0))
    return env_term + form_term + stab_term + l200_term + sb + pos_term + tiny




# === デバッグ表示（必要なときだけ / anchor_score定義の後, 印出力の前） ===
# for no in active_cars:
#     role = role_in_line(no, line_def)
#     sb_dbg  = bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
#     pos_dbg = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
#     form_dbg = SD_FORM * FORM_Z.get(no, 0.0)
#     env_dbg  = SD_ENV  * ENV_Z.get(no, 0.0)
#     stab_dbg = (SD_STAB * STAB_Z.get(no, 0.0)) if 'STAB_Z' in globals() else 0.0
#     tiny_dbg = SMALL_Z_RATING * zt_map.get(no, 0.0)

#     total = form_dbg + env_dbg + stab_dbg + sb_dbg + pos_dbg + tiny_dbg
#     st.write(no, {
#         "form": round(form_dbg, 4),
#         "env":  round(env_dbg, 4),
#         "stab": round(stab_dbg, 4),
#         "sb":   round(sb_dbg, 4),
#         "pos":  round(pos_dbg, 4),
#         "tiny": round(tiny_dbg, 4),
#         "TOTAL(anchor_score期待値)": round(total, 4),
#     })



# ===== ◎候補抽出（既存ロジック維持）
cand_sorted = sorted(active_cars, key=lambda n: anchor_score(n), reverse=True)
C = cand_sorted[:min(3, len(cand_sorted))]
ratings_sorted2 = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank2 = {n: i+1 for i,n in enumerate(ratings_sorted2)}
ALLOWED_MAX_RANK = globals().get("ALLOWED_MAX_RANK", 5)

guarantee_top_rating = True
if guarantee_top_rating and (race_class == "ガールズ") and len(ratings_sorted2) >= 1:
    top_rating_car = ratings_sorted2[0]
    if top_rating_car not in C:
        C = [top_rating_car] + [c for c in C if c != top_rating_car]
        C = C[:min(3, len(cand_sorted))]

ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)

# ===== ANCHOR 選定（SBなし母集団ベース）+ 安全弁 + DEBUG =====
ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)

# --- DEBUG（必要ならOFFにできる） ---
DBG_ANCHOR = bool(globals().get("DBG_ANCHOR", True))

def _safe_int(x, default=1):
    try:
        return int(x)
    except Exception:
        return int(default)

# df_sorted_pure が空なら、active_cars を母集団として使う（落下防止）
df_pure_empty = (df_sorted_pure is None) or (len(df_sorted_pure) == 0)

if df_pure_empty:
    base_order = [int(x) for x in list(active_cars)[:]]  # 1..7
else:
    # 念のため int 化
    base_order = df_sorted_pure["車番"].astype(int).tolist()

# rank_pure（SBなしランキング順位）
rank_pure = {int(no): i + 1 for i, no in enumerate(base_order)}

# 候補プール：C の中で SBなし上位K位
cand_pool = [int(c) for c in C if rank_pure.get(int(c), 999) <= ANCHOR_CAND_SB_TOPK]

# もし空なら、SBなし上位K位から直接作る
if not cand_pool:
    cand_pool = [int(no) for no in base_order[:min(ANCHOR_CAND_SB_TOPK, len(base_order))]]

# 最終フォールバック（どれも無い場合）
fallback_no = int(active_cars[0]) if active_cars else 1

# anchor_no_pre（まずは候補プール内で anchor_score 最大）
if cand_pool:
    anchor_no_pre = max(cand_pool, key=lambda x: anchor_score(int(x)))
else:
    anchor_no_pre = fallback_no

anchor_no = anchor_no_pre

# 同点圏（TIE_EPSILON以内）なら ratings_rank2 で決める
top2 = sorted(cand_pool, key=lambda x: anchor_score(int(x)), reverse=True)[:2]
if len(top2) >= 2:
    s1 = float(anchor_score(int(top2[0])))
    s2 = float(anchor_score(int(top2[1])))
    if (s1 - s2) < TIE_EPSILON:
        better_by_rating = min(top2, key=lambda x: ratings_rank2.get(int(x), 999))
        anchor_no = int(better_by_rating)

# SBなし上位N位縛り
if rank_pure.get(int(anchor_no), 999) > ANCHOR_REQUIRE_TOP_SB:
    pool = [int(c) for c in cand_pool if rank_pure.get(int(c), 999) <= ANCHOR_REQUIRE_TOP_SB]
    if pool:
        anchor_no = max(pool, key=lambda x: anchor_score(int(x)))
    else:
        anchor_no = int(base_order[0]) if base_order else fallback_no

    st.caption(
        f"※ ◎は『SBなし 上位{ANCHOR_REQUIRE_TOP_SB}位以内』縛りで {anchor_no_pre}→{anchor_no} に調整。"
    )



# ===== confidence 算出（anchor_score のギャップ/分散）=====
role_map = {int(no): role_in_line(int(no), line_def) for no in active_cars}

cand_scores = [float(anchor_score(int(no))) for no in C] if len(C) >= 2 else [0.0, 0.0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = float(cand_scores_sorted[0] - cand_scores_sorted[1]) if len(cand_scores_sorted) >= 2 else 0.0

# v_final が空のときは spread=0 で落ちないように（confidenceは混戦寄りになる）
spread = float(np.std(list(v_final.values()))) if isinstance(v_final, dict) and len(v_final) >= 2 else 0.0
norm = conf_gap / (spread if spread > 1e-6 else 1.0)
confidence = "優位" if norm >= 1.0 else ("互角" if norm >= 0.5 else "混戦")

# ===== 格上げ（v_final が空でも落ちないように）=====
if not isinstance(v_final, dict) or len(v_final) == 0:
    # downstream を落とさないための最小母集団（全車0）
    v_final = {int(no): 0.0 for no in active_cars}

score_adj_map = apply_anchor_line_bonus(v_final, car_to_group, role_map, int(anchor_no), confidence)

df_sorted_wo = pd.DataFrame({
    "車番": [int(c) for c in active_cars],
    "合計_SBなし": [
        round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6)
        for c in active_cars
    ]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(
    df_sorted_wo["車番"].astype(int).tolist(),
    df_sorted_wo["合計_SBなし"].round(3).tolist()
))
# ==============================
# ★ レース内T偏差値 → 印 → 買い目 → note出力（2車系対応＋会場個性浸透版）
# ==============================
import math
import numpy as np
import pandas as pd
import streamlit as st

import re
from typing import List

def parse_line_str(line_str: str) -> List[List[int]]:
    s = (line_str or "").strip()
    if not s:
        return []
    s = s.replace("　", " ")
    groups = [g for g in s.split(" ") if g]
    lines = []
    for g in groups:
        nums = [int(ch) for ch in re.findall(r"\d", g)]
        if nums:
            lines.append(nums)
    return lines

def initial_queue_from_lines(lines: List[List[int]]) -> List[int]:
    q = []
    used = set()
    for group in lines:
        for n in group:
            if n not in used:
                q.append(n)
                used.add(n)
    return q

def estimate_finaljump_queue(initial_queue: List[int], score_rank: List[int], k: float = 2.2) -> List[int]:
    if not score_rank:
        return []
    if not initial_queue:
        return score_rank[:]
    pos = {n: i for i, n in enumerate(initial_queue)}
    nmax = max(len(score_rank), 1)
    power = {n: (nmax - i) for i, n in enumerate(score_rank)}  # 1位が最大
    def key(n: int) -> float:
        p0 = pos.get(n, 10_000)
        pw = power.get(n, 0)
        return p0 - k * pw
    return sorted(score_rank, key=key)

def arrow_format(order: List[int]) -> str:
    return " → ".join(str(n) for n in order)


HEN_DEC_PLACES = 1
EPS = 1e-12

# ====== ユーティリティ ======
def coerce_score_map(d, n_cars: int) -> dict[int, float]:
    out: dict[int, float] = {}
    t = str(type(d)).lower()
    if "pandas.core.frame" in t:
        df_ = d
        car_col = "車番" if "車番" in df_.columns else None
        if car_col is None:
            for c in df_.columns:
                if np.issubdtype(df_[c].dtype, np.integer):
                    car_col = c; break
        score_col = None
        for cand in ["合計_SBなし","SBなし","スコア","score","SB_wo","SB"]:
            if cand in df_.columns:
                score_col = cand; break
        if score_col is None:
            for c in df_.columns:
                if c == car_col: continue
                if np.issubdtype(df_[c].dtype, np.number):
                    score_col = c; break
        if car_col is not None and score_col is not None:
            for _, r in df_.iterrows():
                try:
                    i = int(r[car_col]); x = float(r[score_col])
                except Exception:
                    continue
                out[i] = x
    elif "pandas.core.series" in t:
        for k, v in d.to_dict().items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif hasattr(d, "items"):
        for k, v in d.items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif isinstance(d, (list, tuple, np.ndarray)):
        arr = list(d)
        if len(arr) == n_cars and all(not isinstance(x,(list,tuple,dict)) for x in arr):
            for idx, v in enumerate(arr, start=1):
                try: out[idx] = float(v)
                except Exception: out[idx] = np.nan
        else:
            for it in arr:
                if isinstance(it,(list,tuple)) and len(it) >= 2:
                    try:
                        i = int(it[0]); x = float(it[1])
                        out[i] = x
                    except Exception:
                        continue
    for i in range(1, int(n_cars)+1):
        out.setdefault(i, np.nan)
    return out









def _format_rank_from_array(ids, arr):
    pairs = [(i, float(arr[idx])) for idx, i in enumerate(ids)]
    pairs.sort(key=lambda kv: ((1,0) if not np.isfinite(kv[1]) else (0,-kv[1]), kv[0]))
    return " ".join(str(i) for i,_ in pairs)

# ====== ここから処理本体 ======

# 1) 母集団車番
try:
    USED_IDS = sorted(int(i) for i in (active_cars if active_cars else range(1, n_cars+1)))
except Exception:
    USED_IDS = list(range(1, int(n_cars)+1))
M = len(USED_IDS)

# 2) SBなしのソース（df優先→velobi_wo）
score_map_from_df = coerce_score_map(globals().get("df_sorted_wo", None), n_cars)
score_map_vwo     = coerce_score_map(globals().get("velobi_wo", None),   n_cars)
SB_BASE_MAP = score_map_from_df if any(np.isfinite(list(score_map_from_df.values()))) else score_map_vwo

# 偏差値母集団は「SBなし（KO適用後＆格上げ前後どちらか）」に固定
SB_BASE_MAP = {int(i): float(score_adj_map.get(int(i), v_final.get(int(i), np.nan))) for i in USED_IDS}



# 3) スコア配列（スコア順表示と偏差値母集団を共用）
xs_base_raw = np.array([SB_BASE_MAP.get(i, np.nan) for i in USED_IDS], dtype=float)

# 4) 偏差値T（レース内：平均50・SD10、NaN→50）
xs_race_t, mu_sb, sd_sb, k_finite = t_score_from_finite(xs_base_raw)




missing = ~np.isfinite(xs_base_raw)
if missing.any():
    sb_for_sort = {i: SB_BASE_MAP.get(i, -1e18) for i in USED_IDS}
    idxs = np.where(missing)[0].tolist()
    idxs.sort(key=lambda ii: (-float(sb_for_sort.get(USED_IDS[ii], -1e18)), USED_IDS[ii]))
    k = len(idxs); delta = 0.12; center = (k - 1)/2.0 if k > 1 else 0.0
    for r, ii in enumerate(idxs):
        xs_race_t[ii] = 50.0 + delta * (center - r)

# 5) dict化・表示用
race_t = {USED_IDS[idx]: float(round(xs_race_t[idx], HEN_DEC_PLACES)) for idx in range(M)}

# === 5.5) クラス別ライン偏差値ボーナス（ライン間→ライン内：低T優先 3:2:1） ===
# クラス別の総ポイント（Girlsは無効）
CLASS_LINE_POOL = {
    "Ｓ級":           21.0,
    "Ａ級":           15.0,
    "Ａ級チャレンジ":  9.0,
    "ガールズ":        0.0,
}
pool_total = float(CLASS_LINE_POOL.get(race_class, 0.0))

def _line_rank_weights(n_lines: int) -> list[float]:
    # 2本: 3:2 / 3本: 5:4:3 / 4本以上: 6,5,4,3,2,1...
    if n_lines <= 1: return [1.0]
    if n_lines == 2: return [3.0, 2.0]
    if n_lines == 3: return [5.0, 4.0, 3.0]
    base = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    if n_lines <= len(base): return base[:n_lines]
    ext = base[:]
    while len(ext) < n_lines:
        ext.append(max(1.0, ext[-1]-1.0))
    return ext[:n_lines]

def _in_line_weights(members_sorted_lowT_first: list[int]) -> dict[int, float]:
    # ライン内は「低T優先で 3:2:1、4人目以降0」→合計1に正規化
    raw = [3.0, 2.0, 1.0]
    w = {}
    for i, car in enumerate(members_sorted_lowT_first):
        w[int(car)] = (raw[i] if i < len(raw) else 0.0)
    s = sum(w.values())
    return {k: (v/s if s > 0 else 0.0) for k, v in w.items()}

_lines = list((globals().get("line_def") or {}).values())
if pool_total > 0.0 and _lines:
    # ライン強度＝そのラインの race_t 平均
    line_scores = []
    for mem in _lines:
        if not mem: 
            continue
        avg_t = float(np.mean([race_t.get(int(c), 50.0) for c in mem]))
        line_scores.append((tuple(mem), avg_t))
    # 強い順に並べてライン間ポイント配分
    line_scores.sort(key=lambda x: (-x[1], x[0]))
    rank_w = _line_rank_weights(len(line_scores))
    sum_rank_w = float(sum(rank_w)) if rank_w else 1.0
    line_share = {}
    for (mem, _avg), wr in zip(line_scores, rank_w):
        line_share[mem] = pool_total * (float(wr) / sum_rank_w)

    # 各ラインの配分を「低T→高T」の順に 3:2:1 で割り振り
    bonus_map = {int(i): 0.0 for i in USED_IDS}
    for mem, share in line_share.items():
        mem = list(mem)
        mem_sorted_lowT = sorted(mem, key=lambda c: (race_t.get(int(c), 50.0), int(c)))
        w_in = _in_line_weights(mem_sorted_lowT)  # 合計1
        for car in mem_sorted_lowT:
            bonus_map[int(car)] += share * w_in[int(car)]

    # 偏差値に加算（xs_race_tが計算本体。race_tは表示用に丸め直す）
    for idx, car in enumerate(USED_IDS):
        add = float(bonus_map.get(int(car), 0.0))
        xs_race_t[idx] = float(xs_race_t[idx]) + add
        race_t[int(car)] = float(round(xs_race_t[idx], HEN_DEC_PLACES))
# ← この後に既存の race_z 計算が続く



# ==============================
# 偏差値テーブル（SBなし母集団）＋欠損ガード
# ==============================
race_z = (xs_race_t - 50.0) / 10.0

# --- SBなし(母集団) を map として確定（KO入力もここを使う） ---
# USED_IDS と xs_base_raw は「同じ順番」で対応している前提
sb_map = {}
for cid, x in zip(USED_IDS, xs_base_raw):
    try:
        if x is None:
            continue
        xf = float(x)
        if not np.isfinite(xf):
            continue
        sb_map[int(cid)] = xf
    except Exception:
        pass

# --- 欠損チェック（None連発の犯人特定） ---
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")


# zipで短くなってる可能性チェック
if len(xs_base_raw) != len(USED_IDS):
    st.error("xs_base_raw と USED_IDS の長さが一致していません。zip が途中で切れて欠損になります。")


# --- 表（hen_df）を sb_map から作る：Noneは明示的にNoneで残す ---
hen_df = pd.DataFrame({
    "車": USED_IDS,
    "SBなし(母集団)": [sb_map.get(int(cid), None) for cid in USED_IDS],
    "偏差値T(レース内)": [race_t[int(cid)] for cid in USED_IDS],
}).sort_values(["偏差値T(レース内)", "車"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### 偏差値（レース内T＝平均50・SD10｜SBなしと同一母集団）")
st.caption(f"μ={mu_sb if np.isfinite(mu_sb) else 'nan'} / σ={sd_sb:.6f} / 有効件数k={k_finite}")
st.dataframe(hen_df, use_container_width=True)

# 7) 印（◎〇▲）＝ T↓ → SBなし↓ → 車番↑（βは除外）
if "select_beta" not in globals():
    def select_beta(cars): return None
if "enforce_alpha_eligibility" not in globals():
    def enforce_alpha_eligibility(m): return m

# ===== βラベル付与（単なる順位ラベル） =====
def assign_beta_label(result_marks: dict[str, int], used_ids: list[int], df_sorted) -> dict[str, int]:
    marks = dict(result_marks)
    # 6車以下は出さない（集計仕様）
    if len(used_ids) <= 6:
        return marks
    # 既にβがあれば何もしない
    if "β" in marks:
        return marks
    try:
        last_car = int(df_sorted.loc[len(df_sorted) - 1, "車番"])
        if last_car not in marks.values():
            marks["β"] = last_car
    except Exception:
        pass
    return marks


# ===== 印の採番（β廃止→無印で保持）========================================
# 依存: USED_IDS, race_t, xs_base_raw, line_def, car_to_group が上で定義済み

# スコアの補助（安定のため race_t 優先→同点は sb_base でタイブレーク）
sb_base = {
    int(USED_IDS[idx]): float(xs_base_raw[idx]) if np.isfinite(xs_base_raw[idx]) else float("-inf")
    for idx in range(len(USED_IDS))
}

def _race_t_val(i: int) -> float:
    try:
        return float(race_t.get(int(i), 50.0))
    except Exception:
        return 50.0

# === βは作らない。全員を候補にして上位から印を振る
seed_pool = list(map(int, USED_IDS))
order_by_T = sorted(
    seed_pool,
    key=lambda i: (-_race_t_val(i), -sb_base.get(i, float("-inf")), i)
)

result_marks: dict[str,int] = {}
reasons: dict[int,str] = {}

# ◎〇▲ を上位から
for mk, car in zip(["◎","〇","▲"], order_by_T):
    result_marks[mk] = int(car)

# ◎の同ラインを優先して残り印（△, ×, α）を埋める
line_def     = globals().get("line_def", {}) or {}
car_to_group = globals().get("car_to_group", {}) or {}
anchor_no    = result_marks.get("◎", None)

mates_sorted: list[int] = []
if anchor_no is not None:
    a_gid = car_to_group.get(anchor_no, None)
    if a_gid is not None and a_gid in line_def:
        used_now = set(result_marks.values())
        mates_sorted = sorted(
            [int(c) for c in line_def[a_gid] if int(c) not in used_now],
            key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
        )

used = set(result_marks.values())
overall_rest = [int(c) for c in USED_IDS if int(c) not in used]
overall_rest = sorted(
    overall_rest,
    key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
)

# 同ライン優先 → 残りスコア順
tail_priority = mates_sorted + [c for c in overall_rest if c not in mates_sorted]

for mk in ["△","×","α"]:
    if mk in result_marks:
        continue
    if not tail_priority:
        break
    no = int(tail_priority.pop(0))
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"

# === 無印の集合（＝上の印が付かなかった残り全員）
marked_ids = set(result_marks.values())
no_mark_ids = [int(c) for c in USED_IDS if int(c) not in marked_ids]
# 表示はT優先・同点はsb_base
no_mark_ids = sorted(
    no_mark_ids,
    key=lambda x: (-_race_t_val(int(x)), -sb_base.get(int(x), float("-inf")), int(x))
)

# ===== 以降のUI出力での使い方 ==============================================
# ・印の一行（note用）: 既存の join を差し替え
#   例）(' '.join(f'{m}{result_marks[m]}' for m in ['◎','〇','▲','△','×','α'] if m in result_marks))
#   の直後などに「無」を追加
#   例）
#   ('無　' + (' '.join(map(str, no_mark_ids)) if no_mark_ids else '—'))
#
# ・以降のロジックでは「β」への参照を残さないこと（Noneチェック含め全削除OK）
#   もし `if i != result_marks.get("β")` のような行が残っていたら、単に削除してください。


if "α" not in result_marks:
    used_now = set(result_marks.values())
    pool = [i for i in USED_IDS if i not in used_now]
    if pool:
        alpha_pick = pool[-1]
        result_marks["α"] = alpha_pick
        reasons[alpha_pick] = reasons.get(alpha_pick, "α（フォールバック：禁止条件全滅→最弱を採用）")




# =========================
#  Tesla369｜出力統合・最終ブロック（安定版・重複なし / 3車ライン厚め対応）
# =========================
import re, json, hashlib, math
from typing import List, Dict, Any, Optional

# ---------- 基本ヘルパ ----------
def _t369_norm(s) -> str:
    return (str(s) if s is not None else "").replace("　", " ").strip()

def _t369_safe_mean(xs, default: float = 0.0) -> float:
    try:
        return sum(xs) / len(xs) if xs else default
    except Exception:
        return default

def _t369_sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-2.0 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ---------- 文脈→ライン/印/スコア復元 ----------
def _t369_parse_lines_from_context() -> List[List[int]]:
    # _groups 優先
    try:
        _gs = globals().get("_groups") or []
        if _gs:
            out: List[List[int]] = []
            for g in _gs:
                ln = [int(x) for x in g if str(x).strip()]
                if ln: out.append(ln)
            if out: return out
    except Exception:
        pass
    # line_inputs（例："16","524","37"...）
    try:
        arr = [_t369_norm(x) for x in (globals().get("line_inputs") or []) if _t369_norm(x)]
        out: List[List[int]] = []
        for s in arr:
            nums = [int(ch) for ch in s if ch.isdigit()]
            if nums: out.append(nums)
        return out
    except Exception:
        return []

def _t369_lines_str(lines: List[List[int]]) -> str:
    return " ".join("".join(str(n) for n in ln) for ln in lines)

def _t369_buckets(lines: List[List[int]]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    lid = 0
    for ln in lines:
        if len(ln) == 1:
            m[ln[0]] = f"S{ln[0]}"
        else:
            lid += 1
            for n in ln: m[n] = f"L{lid}"
    return m

# ライン
_lines_list: List[List[int]] = _t369_parse_lines_from_context()
lines_str: str = globals().get("lines_str") or _t369_lines_str(_lines_list)

# 印（result_marks → {"◎":3,...}）
_result_marks_raw = (globals().get("result_marks", {}) or {})
marks: Dict[str, int] = {}
for k, v in _result_marks_raw.items():
    m = re.search(r"\d+", str(v))
    if m:
        try: marks[str(k)] = int(m.group(0))
        except Exception: pass

# スコア（race_t / USED_IDS）
race_t   = dict(globals().get("race_t", {}) or {})
USED_IDS = list(globals().get("USED_IDS", []) or [])

def _t369_num(v) -> float:
    try: return float(v)
    except Exception:
        try: return float(str(v).replace("%","").strip())
        except Exception: return 0.0

def _t369_get_score_from_entry(e: Any) -> float:
    if isinstance(e, (int, float)): return float(e)
    if isinstance(e, dict):
        for k in ("偏差値","hensachi","dev","score","sc","S","s","val","value"):
            if k in e: return _t369_num(e[k])
    return 0.0

scores: Dict[int, float] = {}
ids_source = USED_IDS[:] or [n for ln in _lines_list for n in ln]
for n in ids_source:
    e = race_t.get(n, race_t.get(int(n), race_t.get(str(n), {})))
    scores[int(n)] = _t369_get_score_from_entry(e)
for n in [x for ln in _lines_list for x in ln]:
    scores.setdefault(int(n), 0.0)

# ---------- 流れ指標（簡潔・安定版） ----------
# ---------- 流れ指標（簡潔・安定版） ----------
def compute_flow_indicators(lines_str, marks, scores):
    parts = [_t369_norm(p) for p in str(lines_str).split() if _t369_norm(p)]
    lines = [[int(ch) for ch in p if ch.isdigit()] for p in parts if any(ch.isdigit() for ch in p)]
    if not lines:
        return {
            "VTX": 0.0, "FR": 0.0, "U": 0.0,
            "note": "【流れ未循環】ラインなし → ケン",
            "waves": {}, "vtx_bid": "", "lines": [], "dbg": {},
            "FR_line": [], "VTX_line": [], "U_line": []
        }

    buckets = _t369_buckets(lines)
    bucket_to_members = {buckets[ln[0]]: ln for ln in lines}

    def mean(xs, d=0.0):
        try:
            return sum(xs) / len(xs) if xs else d
        except Exception:
            return d

    def avg_score(mem):
        return mean([scores.get(n, 50.0) for n in mem], 50.0)

    muA = mean([avg_score(ln) for ln in lines], 50.0) / 100.0
    star_id = marks.get("◎", -999)
    none_id = marks.get("無", -999)

    def est(mem):
        A = max(10.0, min(avg_score(mem), 90.0)) / 100.0
        if star_id in mem:
            phi0, d = -0.8, +1
        elif none_id in mem:
            phi0, d = +0.8, -1
        else:
            phi0, d = +0.2, +1
        phi = phi0 + 1.2 * (A - muA)
        return A, phi, d

    def S_end(A, phi, t=0.9, f=0.9, gamma=0.12):
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    waves = {}
    for bid, mem in bucket_to_members.items():
        A, phi, d = est(mem)
        waves[bid] = {"A": A, "phi": phi, "d": d, "S": S_end(A, phi, t=0.9)}

    def I(bi, bj):
        if not bi or not bj or bi not in waves or bj not in waves:
            return 0.0
        return math.cos(waves[bi]["phi"] - waves[bj]["phi"])

    # ★順流/逆流：ライン強さ（スコア合計）で決める
    def line_strength(bid: str) -> float:
        mem = bucket_to_members.get(bid, [])
        return float(sum(scores.get(n, 50.0) for n in mem))

    all_buckets = list(bucket_to_members.keys())
    b_star = max(all_buckets, key=lambda bid: (line_strength(bid), bid))
    cand_buckets = [bid for bid in all_buckets if bid != b_star]
    b_none = min(cand_buckets, key=lambda bid: (line_strength(bid), bid)) if cand_buckets else ""

    # --- VTX ---
    vtx_list = []
    for bid, mem in bucket_to_members.items():
        if bid in (b_star, b_none):
            continue
        if waves.get(bid, {}).get("S", -1e9) < -0.02:
            continue
        wA = 0.5 + 0.5 * waves[bid]["A"]
        v = (0.6 * abs(I(bid, b_star)) + 0.4 * abs(I(bid, b_none))) * wA
        vtx_list.append((v, bid))
    vtx_list.sort(reverse=True, key=lambda x: x[0])
    VTX = vtx_list[0][0] if vtx_list else 0.0
    VTX_bid = vtx_list[0][1] if vtx_list else ""

    # --- FR ---
    ws, wn = waves.get(b_star, {}), waves.get(b_none, {})

    def S_point(w, t=0.95, f=0.9, gamma=0.12):
        if not w:
            return 0.0
        A, phi = w.get("A", 0.0), w.get("phi", 0.0)
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    blend_star = 0.6 * S_point(ws) + 0.4 * ws.get("S", 0.0)
    blend_none = 0.6 * S_point(wn) + 0.4 * wn.get("S", 0.0)

    def sig(x, k=3.0):
        try:
            return 1.0 / (1.0 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    sd_raw = (sig(-blend_star, 3.0) - 0.5) * 2.0
    nu_raw = (sig(blend_none, 3.0) - 0.5) * 2.0
    sd = max(0.0, sd_raw)
    nu = max(0.05, nu_raw)
    FR = sd * nu

    # --- U ---
    vtx_vals = [v for v, _ in vtx_list] or [0.0]
    vtx_mu = _t369_safe_mean(vtx_vals, 0.0)
    vtx_sd = (_t369_safe_mean([(x - vtx_mu) ** 2 for x in vtx_vals], 0.0)) ** 0.5
    vtx_hi = max(0.60, vtx_mu + 0.35 * vtx_sd)
    VTX_high = 1.0 if VTX >= vtx_hi else 0.0

    S_max = max(1e-6, max(abs(w["S"]) for w in waves.values()))
    S_noneN = max(0.0, wn.get("S", 0.0)) / S_max
    U_raw = sig(I(b_none, b_star), k=2.0)
    U = max(0.05, (0.6 * U_raw + 0.4 * S_noneN) * (1.0 if VTX_high > 0 else 0.8))

    def label(bid):
        mem = bucket_to_members.get(bid, [])
        return "".join(map(str, mem)) if mem else "—"

    note = "\n".join([
        f"【順流】◎ライン {label(b_star)}：失速危険 {'高' if FR >= 0.15 else ('中' if FR >= 0.05 else '低')}",
        f"【渦】候補ライン：{label(VTX_bid)}（VTX={VTX:.2f}）",
        f"【逆流】無ライン {label(b_none)}：U={U:.2f}（※判定基準内）",
    ])

    dbg = {"blend_star": blend_star, "blend_none": blend_none, "sd": sd, "nu": nu, "vtx_hi": vtx_hi}

    # ★パッチ2：内部で使ったラインを返す
    def members_of(bid: str) -> list[int]:
        return list(bucket_to_members.get(bid, []) or [])

    FR_line = members_of(b_star)
    VTX_line = members_of(VTX_bid)
    U_line = members_of(b_none)

    return {
        "VTX": VTX,
        "FR": FR,
        "U": U,
        "note": note,
        "waves": waves,
        "vtx_bid": VTX_bid,
        "lines": lines,
        "dbg": dbg,
        "FR_line": FR_line,
        "VTX_line": VTX_line,
        "U_line": U_line,
    }


# === v2.3: 相手4枠ロジック（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚まで許容）===

import re
from typing import List, Dict, Optional

def _t369p_parse_groups(lines_str: str) -> List[List[int]]:
    parts = re.findall(r'[0-9]+', str(lines_str or ""))
    groups: List[List[int]] = []
    for p in parts:
        g = [int(ch) for ch in p]
        if g:
            groups.append(g)
    return groups

def _t369p_find_line_of(num: int, groups: List[List[int]]) -> List[int]:
    for g in groups:
        if num in g:
            return g
    return []

def _t369p_line_avg(g: List[int], hens: Dict[int, float]) -> float:
    if not g:
        return -1e9
    return sum(hens.get(x, 0.0) for x in g) / len(g)

def _t369p_best_in_group(
    g: List[int],
    hens: Dict[int, float],
    exclude: Optional[int] = None
) -> Optional[int]:
    cand = [x for x in (g or []) if x != exclude]
    if not cand:
        return None
    return max(cand, key=lambda x: hens.get(x, 0.0), default=None)

def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: Dict[int, float],              # 偏差値/スコアのマップ
    vtx: float,                          # 渦の強さ（0〜1）
    u: float,                            # 逆流の強さ（0〜1）
    marks: Dict[str, int],               # 印（{'◎':5, ...}）
    shissoku_label: str = "中",          # ◎ラインの「失速危険」：'低'/'中'/'高'
    vtx_line_str: Optional[str] = None,  # 渦候補ライン（例 '375'）
    u_line_str: Optional[str] = None,    # 逆流ライン（例 '63'）
    n_opps: int = 4,
    fr_v: float | None = None,           # レースFR（帯判定用）
) -> List[int]:

    # しきい値/ブースト
    U_HIGH       = 0.90
    THIRD_BOOST  = 0.18
    THICK_BASE   = 0.25
    AXIS_LINE_2P = 0.35

    # 3番手保証（FR帯）
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0
    _FRv = float(fr_v or 0.0)

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # FRライン（◎のライン。なければ平均最大ライン）
    g_star  = marks.get("◎")
    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if not FR_line and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    thick_groups = [g for g in groups if len(g) >= 3]  # 3車(以上)ライン
    thick_others = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # 必須候補
    picks_must: List[int] = []

    # ① 軸相方（番手）を強採用
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ラインの代表）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ）。※3車u_groupは最大2枚まで許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # ④ スコアリング
    scores_local: Dict[int, float] = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方強化＋同ライン控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表を加点
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋“2枚目抑制（3車なら許容2まで）”
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local:
            scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ラインは厚め（基礎加点）
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += THICK_BASE

    # 軸が3車(以上)なら同ライン2枚体制を厚め
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += AXIS_LINE_2P

    # 渦/FRが3車(以上)なら中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30

    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # 3列目ブースト（“3番手”を軽く押す：ライン並びの3番手がいる前提）
    if axis_line and len(axis_line) >= 3:
        third = axis_line[2]
        if third in scores_local:
            scores_local[third] += THIRD_BOOST

    # まずは必須枠を採用（順序維持）
    def _unique_keep_order(xs: List[int]) -> List[int]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア高い順。ただしU高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a: int, b: int, group: List[int]) -> bool:
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)なら、相手4枠に同ライン2枚（相方＋もう1枚）を必ず確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members and x != axis_partner]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸ライン以外で“最厚”の3車(以上)ラインは、相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))
        while len(have) < need and len(picks) > 0:
            cand = max(
                [x for x in best_thick_other if x not in picks and x != axis],
                key=lambda x: hens.get(x, 0.0),
                default=None
            )
            if cand is None:
                break
            drop_cands = [x for x in picks if x not in best_thick_other and x != axis_partner]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分があれば偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        for x in sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True):
            picks.append(x)
            if len(picks) >= n_opps:
                break

    # ==== 3番手保証（FR帯 0.25〜0.65 限定）====
    if BAND_LO <= _FRv <= BAND_HI:
        target = axis_line if (axis_line and len(axis_line) >= 3) else (
            best_thick_other if (best_thick_other and len(best_thick_other) >= 3) else None
        )
        if target:
            g_sorted = sorted(target, key=lambda x: hens.get(x, 0.0), reverse=True)
            if len(g_sorted) >= 3:
                third = g_sorted[2]
                if (third not in picks) and (hens.get(third, 0.0) >= THIRD_MIN) and (third != axis):
                    drop_cands = [x for x in picks if (x not in target) and (x != axis_partner)]
                    if drop_cands:
                        worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                        if worst != third:
                            picks = [x for x in picks if x != worst] + [third]

    # --- 二車軸ロック（相方は絶対保持） ---
    if (axis_partner is not None) and (axis_partner not in picks):
        drop_cands = [x for x in picks if x != axis_partner]
        if drop_cands:
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            picks = [x for x in picks if x != worst] + [axis_partner]
        else:
            picks.append(axis_partner)

    # --- ユニーク＆サイズ調整（相方保護） ---
    seen = set()
    picks = [x for x in picks if not (x in seen or seen.add(x))]

    if len(picks) > n_opps:
        to_drop = len(picks) - n_opps
        cand = [x for x in picks if x != axis_partner]
        cand_sorted = sorted(cand, key=lambda x: scores_local.get(x, -1e9))
        for i in range(min(to_drop, len(cand_sorted))):
            if cand_sorted[i] in picks:
                picks.remove(cand_sorted[i])

    return picks

# === /v2.3 ===




def format_tri_1x4(axis: int, opps: List[int]) -> str:
    opps_sorted = ''.join(str(x) for x in sorted(opps))
    return f"{axis}-{opps_sorted}-{opps_sorted}"

# === PATCH（generate_tesla_bets の直前に挿入）==============================
# 前提：ファイル上部に import re があるならここでは不要（無ければ追加）
# 前提：typing を上で import 済みならここでは不要（無ければ追加）

# 軸選定用（generate_tesla_bets から呼ばれる）
def _topk(line, k, scores):
    line = list(line or [])
    return sorted(line, key=lambda x: (scores.get(x, -1.0), -int(x)), reverse=True)[:k]

def _t369p_parse_groups(lines_str: str):
    parts = re.findall(r"[0-9]+", str(lines_str or ""))
    groups = []
    for p in parts:
        g = [int(ch) for ch in p]
        if g:
            groups.append(g)
    return groups

def _t369p_find_line_of(num: int, groups):
    for g in groups:
        if num in g:
            return g
    return []

def _t369p_line_avg(g, hens):
    if not g:
        return -1e9
    return sum(hens.get(x, 0.0) for x in g) / len(g)

def _t369p_best_in_group(g, hens, exclude=None):
    cand = [x for x in (g or []) if x != exclude]
    if not cand:
        return None
    return max(cand, key=lambda x: hens.get(x, 0.0), default=None)


# ---- 相手4枠ロジック v2.3（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚許容）----
def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: dict,              # {車番:int -> 偏差値/スコア:float}
    vtx: float,              # 渦の強さ（0〜1）
    u: float,                # 逆流の強さ（0〜1）
    marks: dict,             # {印:車番} or {車番:印} が来るので両対応
    shissoku_label: str = "中",
    vtx_line_str=None,
    u_line_str=None,
    n_opps: int = 4,
    fr_v: float | None = None,   # レースFR（帯判定用）
):
    # しきい値/ブースト
    U_HIGH       = 0.90
    THIRD_BOOST  = 0.18
    THICK_BASE   = 0.25
    AXIS_LINE_2P = 0.35

    # 3番手保証（FR帯）
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0
    _FRv = float(fr_v or 0.0)

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # --- ◎車番を marks から取得（{印:車番} / {車番:印} 両対応）---
    g_star = None
    if marks:
        # {印:車番} の可能性
        if all(isinstance(v, int) for v in marks.values()):
            g_star = marks.get("◎", None)
        else:
            # {車番:印} の可能性
            for cid, sym in marks.items():
                try:
                    if sym == "◎":
                        g_star = int(cid)
                        break
                except Exception:
                    pass

    # FRライン（◎のライン。なければ平均最大ライン）
    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if (not FR_line) and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    # 3車(以上)ライン群と「軸以外の最厚」
    thick_groups     = [g for g in groups if len(g) >= 3]
    thick_others     = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # --- 必須枠 ---
    picks_must = []

    # ① 軸相方（番手）
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ラインの代表）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ）。※u_group が3車以上なら最大2枚許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # --- スコアリング ---
    scores_local = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方強化、同ライン他は控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表の底上げ
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋2枚目抑制（3車以上は緩め）
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local:
            scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ライン厚め：基礎加点＋“3列目”ブースト（各ラインの3番手）
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += THICK_BASE
        g_sorted = sorted(g3, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            third = g_sorted[2]
            if third != axis and third in scores_local:
                scores_local[third] += THIRD_BOOST

    # 軸が3車(以上)：同ライン2枚体制を強化
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += AXIS_LINE_2P

    # 渦/FRが3車(以上)：中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30
    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # 必須（順序維持）
    def _unique_keep_order(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア順。U高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a, b, group):
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)→相手4枠に同ライン2枚（相方＋もう1枚）を確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members and x != axis_partner]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸以外で“最厚”の3車(以上)ライン→相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))
        while len(have) < need and len(picks) > 0:
            cand = max([x for x in best_thick_other if x not in picks and x != axis],
                       key=lambda x: hens.get(x, 0.0), default=None)
            if cand is None:
                break
            drop_cands = [x for x in picks if x not in best_thick_other and x != axis_partner]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分を偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        for x in sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True):
            picks.append(x)
            if len(picks) >= n_opps:
                break

    # ===== 3番手保証（FR帯 0.25〜0.65）=====
    if (BAND_LO <= _FRv <= BAND_HI) and axis_line and len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]
            if (axis_third not in picks) and (hens.get(axis_third, 0.0) >= THIRD_MIN) and (axis_third != axis):
                drop_cands = [x for x in picks if (x not in axis_line) and (x != axis_partner)]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [axis_third]

    # --- ユニーク＆サイズ調整（相方を落とさない） ---
    seen = set()
    uniq = []
    for x in picks:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    picks = uniq

    if len(picks) > n_opps:
        # 相方は保護して、残りから低スコアを落とす
        protect = set([axis_partner]) if axis_partner is not None else set()
        drop_pool = [x for x in picks if x not in protect]
        drop_pool_sorted = sorted(drop_pool, key=lambda x: scores_local.get(x, -1e9))
        while len(picks) > n_opps and drop_pool_sorted:
            picks.remove(drop_pool_sorted.pop(0))

    return picks


def _format_tri_axis_partner_rest(axis: int, opps: list, axis_line: list,
                                  hens: dict, lines: list) -> str:
    """
    出力形式： 軸・相方 － 残り3枠 － 残り3枠
    並び規則：対抗ラインの2名（番号昇順）→ 軸ラインの3番手（存在時）→ 残りをスコア順で充填
    """
    if not isinstance(axis, int) or axis <= 0 or not isinstance(opps, list):
        return "—"

    hens = {int(k): float(v) for k, v in (hens or {}).items() if str(k).isdigit()}
    axis_line = list(axis_line or [])

    # 相方（軸ライン内の最上位・軸以外）
    partner = None
    if axis in axis_line:
        cands = [x for x in axis_line if x != axis]
        if cands:
            partner = max(cands, key=lambda x: (hens.get(x, 0.0), -int(x)))

    # フォールバック：相方不在なら通常 1-XXXX-XXXX
    if partner is None:
        rest = "".join(str(x) for x in sorted(opps))
        return f"{axis}-{rest}-{rest}"

    # 軸3番手（スコア順の3番手）
    axis_third = None
    if len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]

    # 対抗ライン（＝軸ライン以外で平均偏差最大）
    def _line_avg(g):
        return sum(hens.get(x, 0.0) for x in g) / len(g) if g else -1e9
    other_lines = [g for g in (lines or []) if g != axis_line]
    opp_line = max(other_lines, key=_line_avg) if other_lines else []

    # 残り3枠（相方を除く）
    pool = [x for x in opps if x != partner]

    # まず対抗ラインの2名（昇順で最大2名）
    opp_two = sorted([x for x in pool if x in (opp_line or [])])[:2]

    rest_three = []
    rest_three.extend(opp_two)

    # 軸3番手を追加（まだ入っておらず、poolに居るなら）
    if axis_third is not None and axis_third in pool and axis_third not in rest_three:
        rest_three.append(axis_third)

    # 不足充填：スコア降順→番号昇順で埋める
    if len(rest_three) < 3:
        remain = [x for x in pool if x not in rest_three]
        remain_sorted = sorted(remain, key=lambda x: (hens.get(x, 0.0), -int(x)), reverse=True)
        rest_three.extend(remain_sorted[: (3 - len(rest_three))])

    rest_three = rest_three[:3]

    # 表示は「対抗(昇順) → それ以外」の順
    in_opp = [x for x in rest_three if x in (opp_line or [])]
    not_opp = [x for x in rest_three if x not in (opp_line or [])]
    rest_str = "".join(str(x) for x in (sorted(in_opp) + not_opp))

    return f"{axis}・{partner} － {rest_str} － {rest_str}"

# === /PATCH ==============================================================


# ======================= T369｜FREE-ONLY 完全置換ブロック（精簡版） =======================

# ---- 小ヘルパ（ローカル名で衝突回避） -----------------------------------------
def _free_fmt_nums(arr):
    if isinstance(arr, list):
        return "".join(str(x) for x in arr) if arr else "—"
    return "—"

def _free_norm_marks(marks_any):
    marks_any = dict(marks_any or {})
    if not marks_any:
        return {}
    # 値が全部 int → {印:車番} と判断し反転
    if all(isinstance(v, int) for v in marks_any.values()):
        out = {}
        for k, v in marks_any.items():
            try:
                out[int(v)] = str(k)
            except Exception:
                pass
        return out
    # それ以外は {車番:印}
    out = {}
    for k, v in marks_any.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out

def _free_fmt_marks_line(raw_marks: dict, used_ids: list) -> tuple[str, str]:
    """
    raw_marks: {車番:int -> '◎'} または { '◎' -> 車番:int } の両方に対応
    used_ids:  表示対象の車番リスト（スコア順など）
    戻り値: ("◎5 〇3 ▲1 △2 ×6 α7", "を除く未指名：...") のタプル
    """
    used_ids = [int(x) for x in (used_ids or [])]
    marks = _free_norm_marks(raw_marks)
    prio = ["◎", "〇", "▲", "△", "×", "α"]
    parts = []
    for s in prio:
        ids = [cid for cid, sym in marks.items() if sym == s]
        ids_sorted = sorted(ids, key=lambda c: (used_ids.index(c) if c in used_ids else 10**9, c))
        parts.extend([f"{s}{cid}" for cid in ids_sorted])
    marks_str = " ".join(parts)
    un = [cid for cid in used_ids if cid not in marks]
    no_str = ("を除く未指名：" + " ".join(map(str, un))) if un else ""
    return marks_str, no_str

# --- 3区分バンド（短評で使うなら残す） ---
def _band3_fr(fr: float) -> str:
    if fr >= 0.61: return "不利域"
    if fr >= 0.46: return "標準域"
    return "有利域"

def _band3_vtx(v: float) -> str:
    if v > 0.60:  return "不利域"
    if v >= 0.52: return "標準域"
    return "有利域"

def _band3_u(u: float) -> str:
    if u > 0.65:  return "不利域"
    if u >= 0.55: return "標準域"
    return "有利域"

# --- 優位/互角/混戦 判定（必要なら残す） ---
def infer_eval_with_share(fr_v: float, vtx_v: float, u_v: float, share_pct: float | None) -> str:
    fr_low, fr_high = 0.40, 0.60
    vtx_strong, u_strong = 0.60, 0.65
    share_lo, share_hi = 25.0, 33.0  # %
    if (fr_v > fr_high) and (vtx_v <= vtx_strong) and (u_v <= u_strong) and (share_pct is not None and share_pct >= share_hi):
        return "優位"
    if (fr_v < fr_low) or ((vtx_v > vtx_strong) and (u_v > u_strong)) or (share_pct is not None and share_pct <= share_lo):
        return "混戦"
    return "互角"

# ============================================================
# /T369｜FREE-ONLY 出力一括ブロック（券種コード完全撤去 + 0.000連発対策 + KO統一）
# ============================================================

def _normalize_lines(_lines):
    """
    入力 lines を必ず [[2,4],[5,7,1]...] の形にする
    - "24" / 24 / [24] / [2,4] どれでもOK（数字だけ抜いて桁分解）
    """
    out = []
    for ln in (_lines or []):
        if ln is None:
            continue
        s = "".join(ch for ch in str(ln) if ch.isdigit())
        if not s:
            continue
        out.append([int(ch) for ch in s])
    return out

# --- line_fr_map が無い/空でも出せる保険（本体は既存 _build_line_fr_map を優先） ---
if "_build_line_fr_map" not in globals():
    def _build_line_fr_map(lines, scores_map, FRv,
                           SINGLETON_FR_SCALE=0.70,
                           MIN_LINE_SHARE=0.00,
                           MAX_SINGLETON_SHARE=0.45):
        lines = _normalize_lines(lines)
        scores_map = {int(k): float(v) for k, v in (scores_map or {}).items() if str(k).strip().isdigit()}
        FRv = float(FRv or 0.0)
        if not lines:
            return {}

        line_sums = []
        for ln in lines:
            s = sum(scores_map.get(int(x), 0.0) for x in ln)
            if len(ln) == 1:
                s *= float(SINGLETON_FR_SCALE)
            line_sums.append((ln, s))

        total = sum(s for _, s in line_sums)
        sum_target = FRv if FRv > 0.0 else 1.0

        if total <= 0.0:
            eq = 1.0 / len(lines)
            return {"".join(map(str, ln)): eq for ln, _ in line_sums}

        return {"".join(map(str, ln)): sum_target * (s / total) for ln, s in line_sums}

# ---------- 3) 安全ラッパ（券種なし：flowだけ） ----------
def _safe_flow(lines_str, marks, scores):
    try:
        fr = compute_flow_indicators(lines_str, marks, scores)
        return fr if isinstance(fr, dict) else {}
    except Exception:
        return {}

# ===================== 4) 出力本体（券種なし・一括置換） =====================
try:
    import math

    # --- note_sections を必ず用意 ---
    ns = globals().get("note_sections", None)
    if not isinstance(ns, list):
        ns = []
        globals()["note_sections"] = ns
    note_sections = ns

    # ---- flow 作成 ----
    _flow = _safe_flow(
        globals().get("lines_str", ""),
        globals().get("marks", {}),
        globals().get("scores", {}),
    )
    globals()["_flow"] = _flow  # 後段参照用に保持

    # ---- 値の確定 ----
    FRv = float(_flow.get("FR", 0.0) or 0.0)
    VTXv = float(_flow.get("VTX", 0.0) or 0.0)
    Uv = float(_flow.get("U", 0.0) or 0.0)

    all_lines = list(_flow.get("lines") or [])
    all_lines = _normalize_lines(all_lines)  # ここで必ず正規化
    globals()["all_lines"] = all_lines

    # ---- レース名 ----
    venue = str(globals().get("track") or globals().get("place") or "").strip()
    race_no = str(globals().get("race_no") or "").strip()
    if venue or race_no:
        _rn = race_no if (race_no.endswith("R") or race_no == "") else f"{race_no}R"
        note_sections.append(f"{venue}{_rn}")
        note_sections.append("")

    # =========================================================
    # KO母集団スコア（v_final > v_wo > scores）で統一
    # =========================================================
    def _as_int_float_map(m):
        out = {}
        if not isinstance(m, dict):
            return out
        for k, v in m.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    out[kk] = vv
            except Exception:
                pass
        return out

    v_final_map = _as_int_float_map(globals().get("v_final"))
    v_wo_map = _as_int_float_map(globals().get("v_wo"))
    scores_map = _as_int_float_map(globals().get("scores"))

    score_map = dict(v_final_map or v_wo_map or scores_map or {})

    # active_cars を必ず含める（欠けを防ぐ）
    active_cars = [int(x) for x in (globals().get("active_cars") or []) if str(x).isdigit()]
    for n in active_cars:
        score_map.setdefault(int(n), 0.0)

   

        # =========================================================
    # KO母集団スコア補正：ライン3番手以降・H0/B0の過大評価抑制
    # ※脚質名に依存しない版。「追」ではなく「マーク」扱いでも効く。
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})

        for _n in list(score_map.keys()):
            _car = int(_n)

            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            _h_val = float(_H.get(_car, _H.get(str(_car), 0)) or 0)
            _b_val = float(_B.get(_car, _B.get(str(_car), 0)) or 0)

            # 例：364 の 4番 = thirdplus、H0、B0 → 必ず減点
            if _role == "thirdplus":
                if _h_val == 0 and _b_val == 0:
                    score_map[_n] = float(score_map[_n]) - 0.15
                else:
                    score_map[_n] = float(score_map[_n]) - 0.08

    except Exception as _e:
        note_sections.append(f"※KO母集団補正エラー：{_e}")

    score_map_before_last_half = dict(score_map)
    globals()["score_map_before_last_half"] = dict(score_map_before_last_half)

    # =========================================================
    # ラスト半周補正：自力粘り・番手差し
    # ※既存のKO母集団スコアに後付けする
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})
        _kaku = globals().get("kaku", {})
        _tenscore = globals().get("tenscore", globals().get("tenscores", {}))

        # 競走得点の取り出し
        def _get_num_from_map(_mp, _car, _default=0.0):
            try:
                if isinstance(_mp, dict):
                    return float(_mp.get(int(_car), _mp.get(str(_car), _default)) or _default)
            except Exception:
                pass
            return float(_default)

        _race_scores = []
        for _n in active_cars:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            if _v > 0:
                _race_scores.append(_v)

        _race_avg_tenscore = float(np.mean(_race_scores)) if _race_scores else 0.0
        _last_half_bonus_map = {}
        _last_half_reason_map = {}
        
                # -------------------------------------------------
        # ラスト半周補正用：レース内順位マップ
        # 上位1/3判定用。7車なら3位以内。
        # -------------------------------------------------
        _active_list = [int(x) for x in active_cars]
        _top_third_limit = int(math.ceil(len(_active_list) / 3.0)) if _active_list else 3
        _top_third_limit = max(1, _top_third_limit)

        # 競走得点順位
        _race_score_rank_map = {}
        _ten_pairs = []
        for _n in _active_list:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            _ten_pairs.append((int(_n), float(_v)))

        _ten_pairs_sorted = sorted(_ten_pairs, key=lambda x: (-x[1], x[0]))
        for _idx, (_car2, _v2) in enumerate(_ten_pairs_sorted, start=1):
            _race_score_rank_map[int(_car2)] = int(_idx)

        # KO順位・展開順位
        # この時点の score_map_before_last_half は「ラスト半周補正前」のスコア
        _ko_score_rank_map = {}
        _ko_pairs_sorted = sorted(
            [(int(k), float(v)) for k, v in score_map_before_last_half.items()],
            key=lambda x: (-x[1], x[0])
        )
        for _idx, (_car2, _v2) in enumerate(_ko_pairs_sorted, start=1):
            _ko_score_rank_map[int(_car2)] = int(_idx)

        _tenkai_score_rank_map = dict(_ko_score_rank_map)

        # 順流・渦・逆流の複数上位は次段階用
        _scenario_top_count_map = globals().get("scenario_top_count_map", {})
        if not isinstance(_scenario_top_count_map, dict):
            _scenario_top_count_map = {}

        for _n in list(score_map.keys()):
            _car = int(_n)

            # ライン内の役割
            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            # 同ライン先頭の競走得点
            _leader = _car
            try:
                if isinstance(_line_def, dict):
                    for _gid, _mem in _line_def.items():
                        _mem2 = [int(x) for x in _mem]
                        if _car in _mem2 and _mem2:
                            _leader = int(_mem2[0])
                            break
            except Exception:
                _leader = _car

            _car_ten = _get_num_from_map(_tenscore, _car, 0.0)
            _leader_ten = _get_num_from_map(_tenscore, _leader, _car_ten)

            _h_val = _get_num_from_map(_H, _car, 0.0)
            _b_val = _get_num_from_map(_B, _car, 0.0)

            # kakuは現在の入力仕様では使わない。
            # 関数互換用に空文字で渡す。
            _style = ""

            # H主導ラインの3番手以降かどうか
            _is_h_lead_thirdplus = False
            try:
                _h_members = []
                if home_top_gid is not None and isinstance(_line_def, dict):
                    _h_members = [int(x) for x in _line_def.get(home_top_gid, [])]

                if (
                    len(_h_members) >= 3
                    and _role == "thirdplus"
                    and _car in _h_members[2:]
                ):
                    _is_h_lead_thirdplus = True

            except Exception:
                _is_h_lead_thirdplus = False

            # ---------------------------------------------
            # ラスト半周用：個人成績率
            # x1 / x2 / x3 / x_out から
            # 1着率・2着内率・3着内率を作る
            # ---------------------------------------------
            _p1_rate = None
            _p2_rate = None
            _p3_rate = None

            try:
                _x1 = globals().get("x1", {})
                _x2 = globals().get("x2", {})
                _x3 = globals().get("x3", {})
                _xo = globals().get("x_out", {})

                _n1 = float(_x1.get(_car, _x1.get(str(_car), 0)) or 0)
                _n2 = float(_x2.get(_car, _x2.get(str(_car), 0)) or 0)
                _n3 = float(_x3.get(_car, _x3.get(str(_car), 0)) or 0)
                _no = float(_xo.get(_car, _xo.get(str(_car), 0)) or 0)

                _total = _n1 + _n2 + _n3 + _no

                if _total > 0:
                    _p1_rate = _n1 / _total
                    _p2_rate = (_n1 + _n2) / _total
                    _p3_rate = (_n1 + _n2 + _n3) / _total

            except Exception:
                _p1_rate = None
                _p2_rate = None
                _p3_rate = None

            _bonus, _reasons = calc_last_half_role_bonus(
                role=_role,
                kaku=_style,
                tenscore=_car_ten,
                leader_tenscore=_leader_ten,
                race_avg_tenscore=_race_avg_tenscore,
                h_count=_h_val,
                b_count=_b_val,
                race_score_rank=_race_score_rank_map.get(_car),
                ko_score_rank=_ko_score_rank_map.get(_car),
                tenkai_score_rank=_tenkai_score_rank_map.get(_car),
                top_third_limit=_top_third_limit,
                scenario_top_count=int(_scenario_top_count_map.get(_car, 0) or 0),
                p1_rate=_p1_rate,
                p2_rate=_p2_rate,
                p3_rate=_p3_rate,
            )

            _last_half_bonus_map[_car] = float(_bonus)
            _last_half_reason_map[_car] = list(_reasons)

            score_map[_car] = float(score_map.get(_car, 0.0)) + float(_bonus)

    


            # -------------------------------------------------
        # H主導ライン3番手以降：3着内率40%以上なら最低4番手評価まで床上げ
        # -------------------------------------------------
        THIRDPLUS_TOP3_RATE_MIN = 0.40
        THIRDPLUS_FLOOR_RANK = 4
        THIRDPLUS_FLOOR_EPS = 0.001

        def _normalize_rate_0to1(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        def _get_top3_rate_for_car(_car_no):
            """
            車番ごとの3着内率を取得する。
            変数名が多少違っても拾えるように、候補名とglobals内のdictを探す。
            値は 0.40 / 40.0 のどちらでも対応。
            """
            _car_no = int(_car_no)

            # よくありそうな名前を優先
            _candidate_names = [
                "top3_rate_map",
                "in3_rate_map",
                "pTop3_map",
                "ptop3_map",
                "car_top3_rate_map",
                "car_in3_rate_map",
                "top3_map",
                "in3_map",
                "P_TOP3_MAP",
                "IN3_RATE_MAP",
            ]

            for _name in _candidate_names:
                _obj = globals().get(_name, None)
                if isinstance(_obj, dict):
                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r

            # 名前が違う場合の保険：globals内の「top3 / in3 / 3着」系dictを探索
            try:
                for _name, _obj in globals().items():
                    _lname = str(_name).lower()
                    if not isinstance(_obj, dict):
                        continue

                    if not (
                        "top3" in _lname
                        or "in3" in _lname
                        or "p_top3" in _lname
                        or "3着" in str(_name)
                        or "三着" in str(_name)
                    ):
                        continue

                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r
            except Exception:
                pass

            return None



        globals()["last_half_bonus_map"] = _last_half_bonus_map
        globals()["last_half_reason_map"] = _last_half_reason_map
        globals()["score_map_last_half_applied"] = dict(score_map)

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正エラー：{_e}")

        # 0/None/NaN の床値補完
    vals_pos = [
        float(v) for v in score_map.values()
        if isinstance(v, (int, float)) and float(v) > 0.0 and math.isfinite(float(v))
    ]

    _floor = min(vals_pos) if vals_pos else 1e-6

    for k in list(score_map.keys()):
        try:
            v = float(score_map[k])
            if (not math.isfinite(v)) or v <= 0.0:
                score_map[k] = float(_floor)
        except Exception:
            score_map[k] = float(_floor)

    globals()["score_map"] = score_map  # 後段参照用に保持

    # =========================================================
    # line_fr_map を確定（_lfr 未定義事故対策）
    # =========================================================
    line_fr_map = globals().get("line_fr_map")
    need_rebuild = (not isinstance(line_fr_map, dict)) or (len(line_fr_map) == 0)

    # 既存があればキー正規化（tuple/listキー → "571"）
    if (not need_rebuild) and isinstance(line_fr_map, dict):
        _lfm2 = {}
        for k, v in line_fr_map.items():
            try:
                if isinstance(k, (list, tuple, set)):
                    kk = "".join(str(x) for x in k if str(x).isdigit())
                else:
                    kk = "".join(ch for ch in str(k) if ch.isdigit())

                if kk:
                    _lfm2[kk] = float(v or 0.0)
            except Exception:
                continue

        line_fr_map = _lfm2
        need_rebuild = (len(line_fr_map) == 0)

    # 空なら作り直し
    if need_rebuild:
        try:
            line_fr_map = _build_line_fr_map(
                all_lines,
                score_map,
                FRv if FRv > 0.0 else 1.0
            )
        except Exception:
            line_fr_map = {}

    globals()["line_fr_map"] = line_fr_map

    def _line_key(ln):
        try:
            if not ln:
                return ""
            return "".join(str(int(x)) for x in ln if str(x).isdigit())
        except Exception:
            return "".join(ch for ch in str(ln) if ch.isdigit())

    def _lfr(ln):
        try:
            return float(line_fr_map.get(_line_key(ln), 0.0) or 0.0)
        except Exception:
            return 0.0
    # =========================================================
    # 展開評価（share_pct は「順流ライン」基準）
    # =========================================================
    FR_line = _flow.get("FR_line") or []
    VTX_line = _flow.get("VTX_line") or []
    U_line = _flow.get("U_line") or []

    FR_line = _normalize_lines([FR_line])[0] if FR_line else []
    VTX_line = _normalize_lines([VTX_line])[0] if VTX_line else []
    U_line = _normalize_lines([U_line])[0] if U_line else []

    globals()["FR_line"] = FR_line
    globals()["VTX_line"] = VTX_line
    globals()["U_line"] = U_line

    # =========================================================
    # 渦ラインを必ず埋める（空なら自動選定）
    # ルール：FR_line / U_line 以外で、想定FRが最大のラインを渦にする
    # =========================================================
    if (not VTX_line) or (_lfr(VTX_line) <= 0.0):
        _cand = []
        for ln in (all_lines or []):
            if not ln:
                continue
            if ln == FR_line or ln == U_line:
                continue
            _cand.append(ln)
        if _cand:
            VTX_line = max(_cand, key=lambda x: _lfr(x))
            globals()["VTX_line"] = VTX_line

    axis_line = FR_line if FR_line else (all_lines[0] if all_lines else [])
    axis_line_fr = float(line_fr_map.get(_line_key(axis_line), 0.0) or 0.0)
    total_fr = sum(float(v or 0.0) for v in line_fr_map.values()) if isinstance(line_fr_map, dict) else 0.0
    share_pct = (axis_line_fr / total_fr * 100.0) if (total_fr > 1e-12 and axis_line) else None

    note_sections.append(f"展開評価：{infer_eval_with_share(FRv, VTXv, Uv, share_pct)}")
    note_sections.append("")

    # ---- 時刻・クラス ----
    race_time = str(globals().get("race_time", "") or "")
    race_class = str(globals().get("race_class", "") or "")
    hdr = f"{race_time}　{race_class}".strip()
    if hdr:
        note_sections.append(hdr)

        # ---- ライン表示 ----
    line_inputs = globals().get("line_inputs", [])
    if isinstance(line_inputs, list) and any(str(x).strip() for x in line_inputs):
        _lines = [str(x).strip() for x in line_inputs if str(x).strip()]
        note_sections.append("ライン　" + "　".join(_lines))

        # H：最終ホーム想定ライン
        try:
            note_sections.append(f"最終ホーム想定　{home_line_text}")
            note_sections.append(f"H主導ライン　{home_top_line}")
        except Exception:
            pass

    note_sections.append("")

    # =========================================================
    # ライン想定FR（順流/渦/逆流 + その他）表示  ※区分け復活
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

        # =========================================================
    # ライン評価グループ（順流域／渦域／逆流域）
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

    def _same_line(a, b):
        return tuple(int(x) for x in (a or [])) == tuple(int(x) for x in (b or []))

    try:
        h_line_members = line_def.get(home_top_gid, []) if home_top_gid is not None else []
    except Exception:
        h_line_members = []

    valid_lines = [ln for ln in (all_lines or []) if ln]
    line_items = []

    for ln in valid_lines:
        fr = float(_lfr(ln))
        line_items.append({
            "line": ln,
            "fr": fr,
            "is_fr": _same_line(ln, FR_line),
            "is_vtx": _same_line(ln, VTX_line),
            "is_u": _same_line(ln, U_line),
            "is_h": _same_line(ln, h_line_members),
        })

    line_items = sorted(line_items, key=lambda x: (-x["fr"], _fmt_line(x["line"])))

    if line_items:
        top_fr = float(line_items[0]["fr"])

        # FR差による範囲判定
        # 7車以下はやや狭め、8・9車は広め
        if int(n_cars) >= 8:
            upper_gap = 0.080
            middle_ratio = 0.45
            h_gap = 0.150
        else:
            upper_gap = 0.050
            middle_ratio = 0.45
            h_gap = 0.090

        zones = {
            "順流域": [],
            "渦域": [],
            "逆流域": [],
        }

        for item in line_items:
            ln = item["line"]
            fr = float(item["fr"])
            gap = top_fr - fr
            ratio = (fr / top_fr) if top_fr > 1e-12 else 0.0

            tags = []
            if item["is_fr"]:
                tags.append("◎")
            if item["is_h"]:
                tags.append("H主導")
            if item["is_vtx"]:
                tags.append("旧渦")
            if item["is_u"]:
                tags.append("旧逆流")

            # 順流域：
            # FRトップ、またはFRトップとの差が小さいライン
            if item["is_fr"] or gap <= upper_gap:
                zone = "順流域"

            # H主導ラインは、FR2位級なら実質上位へ寄せる
            elif item["is_h"] and (gap <= h_gap or ratio >= 0.55):
                zone = "順流域"
                tags.append("実質上位")

            # 中位以上の別線は渦域
            elif ratio >= middle_ratio:
                zone = "渦域"

            # 低FR・単騎・押上げ側は逆流域
            else:
                zone = "逆流域"

            sort_score = fr + (0.030 if item["is_h"] else 0.0)

            zones[zone].append({
                "line": ln,
                "fr": fr,
                "tags": tags,
                "sort_score": sort_score,
            })

        for z in zones:
            zones[z] = sorted(
                zones[z],
                key=lambda x: (-x["sort_score"], -x["fr"], _fmt_line(x["line"]))
            )

                # =====================================================
        # 全ラインが順流域に吸収された場合の強制分割
        # 目的：順流・渦・逆流メインが全部同じになるのを防ぐ
        # =====================================================
        try:
            if (
                len(zones.get("順流域", [])) >= 3
                and len(zones.get("渦域", [])) == 0
                and len(zones.get("逆流域", [])) == 0
            ):
                all_top_items = list(zones["順流域"])

                # まずFR順で並べる
                all_top_items = sorted(
                    all_top_items,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                # ◎ラインは順流域に残す
                fr_items = [x for x in all_top_items if "◎" in x.get("tags", [])]

                if fr_items:
                    keep_jun = fr_items[0]
                else:
                    keep_jun = all_top_items[0]

                rest = [x for x in all_top_items if x is not keep_jun]

                # 残りの中でFR最上位を渦域へ
                rest = sorted(
                    rest,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                keep_vtx = rest[0] if rest else None
                rest2 = [x for x in rest if x is not keep_vtx]

                zones["順流域"] = [keep_jun]
                zones["渦域"] = [keep_vtx] if keep_vtx is not None else []
                zones["逆流域"] = rest2

        except Exception:
            pass

        

        # KO隊列用：ラインごとの新ゾーン分類を保存
        _LINE_ZONE_MAP = {}

        _zone_to_short = {
            "順流域": "順流",
            "渦域": "渦",
            "逆流域": "逆流",
        }

        for zone_name, items in zones.items():
            short_zone = _zone_to_short.get(zone_name, "その他")
            for item in items:
                try:
                   key = "".join(ch for ch in str(item["line"]) if ch.isdigit())
                   if key:
                       _LINE_ZONE_MAP[key] = short_zone
                except Exception:
                   pass

        globals()["LINE_ZONE_MAP"] = _LINE_ZONE_MAP

        # st.write("DEBUG LINE_ZONE_MAP", _LINE_ZONE_MAP)

        note_sections.append("【ライン評価グループ】")

        for zone_name in ["順流域", "渦域", "逆流域"]:
            items = zones.get(zone_name, [])
            if not items:
                note_sections.append(f"{zone_name}：—")
                continue
            parts = []
            for item in items:
                tag_txt = ""
                if item["tags"]:
                    tag_txt = "・" + "・".join(item["tags"])

                parts.append(
                    f"{_fmt_line(item['line'])}［FR={item['fr']:.3f}{tag_txt}］"
                )

            note_sections.append(f"{zone_name}：" + "／".join(parts))

    else:
        note_sections.append("【ライン評価グループ】")
        note_sections.append("順流域：—")
        note_sections.append("渦域：—")
        note_sections.append("逆流域：—")

    note_sections.append("")

        # =========================================================
    # ラスト半周補正 表示
    # =========================================================
    try:
        _lh_bonus_map = globals().get("last_half_bonus_map", {})
        _lh_reason_map = globals().get("last_half_reason_map", {})
        _before_map = globals().get("score_map_before_last_half", {})
        _after_map = globals().get("score_map_last_half_applied", {})

        if isinstance(_lh_bonus_map, dict) and _lh_bonus_map:
            note_sections.append("【ラスト半周補正】")

            _lh_pairs = sorted(
                [(int(k), float(v)) for k, v in _lh_bonus_map.items()],
                key=lambda t: t[0]
            )

            for _car, _bonus in _lh_pairs:
                _before = float(_before_map.get(_car, 0.0) or 0.0)
                _after = float(_after_map.get(_car, _before + _bonus) or 0.0)

                _reasons = _lh_reason_map.get(_car, [])
                if not isinstance(_reasons, list):
                    _reasons = [_reasons]

                _reason_txt = "／".join(str(x) for x in _reasons if str(x).strip())
                if not _reason_txt:
                    _reason_txt = "補正なし"

                note_sections.append(
                    f"{_car}：展開={_before:.6f} ／ 補正={_bonus:+.3f} ／ 最終={_after:.6f}［{_reason_txt}］"
                )

            note_sections.append("")

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正表示エラー：{_e}")
        note_sections.append("")
    # =========================================================
    # KO使用スコア（降順）
    # =========================================================
    _sc_pairs = sorted(
        [(int(k), float(v)) for k, v in (score_map or {}).items()],
        key=lambda t: (-t[1], t[0])
    )

    note_sections.append("【KO使用スコア（降順）】")

    
    if _sc_pairs:
        for i, (n, sc) in enumerate(_sc_pairs, start=1):
            note_sections.append(f"{i}位：{n} (スコア={sc:.6f})")
    else:
        note_sections.append("—")
    note_sections.append("")

    # =========================================================
    # 最終ジャン想定隊列 → KO（6パターン）
    #   ワープ禁止：全体再ソート禁止
    #   距離：隣同士の交換のみ + 交換コスト
    #   重要：1パス中に同一車が何回も抜けない
    # =========================================================
    def _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr):
        def _digits_of_line(ln):
            s = "".join(ch for ch in str(ln) if ch.isdigit())
            return [int(ch) for ch in s] if s else []

        def _norm_line(ln):
            return "".join(ch for ch in str(ln) if ch.isdigit())

        _PATTERNS = [
            ("順流→渦→逆流", ["順流", "渦", "逆流"]),
            ("順流→逆流→渦", ["順流", "逆流", "渦"]),
            ("渦→順流→逆流", ["渦", "順流", "逆流"]),
            ("渦→逆流→順流", ["渦", "逆流", "順流"]),
            ("逆流→順流→渦", ["逆流", "順流", "渦"]),
            ("逆流→渦→順流", ["逆流", "渦", "順流"]),
        ]

        def _infer_line_zone(ln):
            s = _norm_line(ln)

            # 新方式：ライン評価グループを優先
            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and s in zmap:
                    return zmap.get(s, "その他")
            except Exception:
                pass

            # 保険：旧方式
            if s and FR_line and s == _norm_line(FR_line):
                return "順流"
            if VTX_line and s == _norm_line(VTX_line):
                return "渦"
            if s and U_line and s == _norm_line(U_line):
                return "逆流"

            return "その他"

        def _queue_for_pattern(lines, svr_order):
            lines = list(lines or [])
            bucket = {"順流": [], "渦": [], "逆流": [], "その他": []}
            for ln in lines:
                bucket[_infer_line_zone(ln)].append(ln)

            queue = []
            for z in (svr_order or ["順流", "渦", "逆流"]):
                xs = sorted(bucket.get(z, []), key=lambda x: _lfr(x), reverse=True)
                for ln in xs:
                    queue.extend(_digits_of_line(ln))

            xs = sorted(bucket.get("その他", []), key=lambda x: _lfr(x), reverse=True)
            for ln in xs:
                queue.extend(_digits_of_line(ln))

            if not queue:
                for ln in lines:
                    queue.extend(_digits_of_line(ln))
            return queue

        def _build_car_zone_map(lines):
            m = {}
            for ln in (lines or []):
                z = _infer_line_zone(ln)
                for c in _digits_of_line(ln):
                    m[int(c)] = z
            return m

        _car_zone_map = _build_car_zone_map(all_lines)

        _car_line_size = {}
        _car_line_pos = {}

        for ln in (all_lines or []):
            ds = _digits_of_line(ln)
            sz = len(ds)

            for idx, c in enumerate(ds):
                _car_line_size[int(c)] = sz if sz > 0 else 1
                _car_line_pos[int(c)] = int(idx)

        def _pos_adj_for_car(car):
            """
            位置補正は隊列全体の何番目かではなく、
            その車が所属ライン内で何番手かを見る。
            単騎は番手利を与えない。
            """
            car = int(car)
            sz = int(_car_line_size.get(car, 1) or 1)
            pos = int(_car_line_pos.get(car, 0) or 0)

            # 単騎は位置補正なし
            if sz <= 1:
                return 0.0

            # ライン先頭
            if pos == 0:
                return -0.040

            # ライン2番手
            if pos == 1:
                return +0.020

            # 3番手以降
            return 0.0

        _FR_K_MAIN = 0.18
        _FR_K_SUB = 0.06
        _FR_BONUS_CAP = 0.06

        def _fr_bonus_for_car(car, main_zone):
            z = _car_zone_map.get(int(car), "その他")
            z_fr = {
                "順流": float(_lfr(FR_line) if FR_line else 0.0),
                "渦":   float(_lfr(VTX_line) if VTX_line else 0.0),
                "逆流": float(_lfr(U_line) if U_line else 0.0),
            }.get(z, 0.0)

            k = _FR_K_MAIN if z == main_zone else _FR_K_SUB
            sz = float(_car_line_size.get(int(car), 1) or 1.0)

            bonus = (k * z_fr) / sz
            if bonus > _FR_BONUS_CAP:
                bonus = _FR_BONUS_CAP
            if bonus < 0.0:
                bonus = 0.0
            return bonus

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)


            tail = [int(c) for c in score_map.keys() if int(c) not in seen]
            tail.sort(key=lambda c: float(score_map.get(int(c), 0.0)), reverse=True)
            order.extend(tail)

            straight_m = float(globals().get("straight_length", 60.0) or 60.0)
            style = float(globals().get("style", 0.0) or 0.0)
            wind_ms = float(globals().get("wind_speed", 0.0) or 0.0)
            race_class = str(globals().get("race_class", "Ａ級") or "Ａ級")

            CLASS_SPREAD = {"Ｓ級": 1.00, "Ａ級": 0.90, "Ａ級チャレンジ": 0.80, "ガールズ": 0.85}
            spread = float(CLASS_SPREAD.get(race_class, 0.90))

            def _final_at(car, i):
                base = float(score_map.get(int(car), 0.0))
                return base + _pos_adj_for_car(int(car)) + _fr_bonus_for_car(int(car), main_zone)
            
            # ====== PATCH: venue-aware pass_m / available_m + speed-based MAX_PASSES ======
            pass_m = 14.0 + 0.35 * straight_m
            pass_m *= (1.0 + 0.25 * max(0.0, style))
            pass_m *= (1.0 + 0.03 * max(0.0, wind_ms - 3))

            # 会場カント（薄く：外回しロス増）
            bank_angle = float(globals().get("bank_angle", 30.0) or 30.0)
            pass_m *= (1.0 + 0.10 * max(0.0, (bank_angle - 30.0) / 10.0))  # 36°で+6%程度

            # クリップ
            if pass_m < 18.0:
                pass_m = 18.0
            if pass_m > 55.0:
                pass_m = 55.0

            # ---- available_m: bank_len を “差分だけ” 反映して飽和を減らす ----
            bank_len = float(globals().get("bank_length", 400.0) or 400.0)
            base_bank = 400.0
            # bank_len差分の反映を少し強める（500が1回に張り付くのを緩和）
            bank_term = 0.20 * base_bank + 0.30 * (bank_len - base_bank)
            available_m = float(straight_m) + bank_term

            # ---- スコア分布（sigma）----
            vals = [float(score_map.get(int(c), 0.0)) for c in order]
            if len(vals) >= 2:
                mu = sum(vals) / float(len(vals))
                var = sum((v - mu) ** 2 for v in vals) / float(len(vals))
                sigma = max(var ** 0.5, 1e-6)
            else:
                mu = (vals[0] if vals else 0.0)
                sigma = 1e-6

            # ---- クラス別の代表速度（終盤の代表値）----
            VREF_KMH = {"Ｓ級": 67.0, "Ａ級": 64.0, "Ａ級チャレンジ": 62.0, "ガールズ": 63.0}
            v_ref = float(VREF_KMH.get(race_class, 64.0)) / 3.6  # m/s

            # ---- スコア→速度：zで圧縮（暴走防止）----
            # 333/335は終盤時間が短く gain_m が出にくいので少し強める
            if bank_len <= 335:
                k_speed = float(globals().get("ko_k_speed_333", 0.014) or 0.014)
            elif bank_len >= 500:
                k_speed = float(globals().get("ko_k_speed_500", 0.012) or 0.012)
            else:
                k_speed = float(globals().get("ko_k_speed", 0.011) or 0.011)
            def _v_from_score(sc: float) -> float:
                z = (float(sc) - float(mu)) / float(sigma)
                if z > 2.0:
                    z = 2.0
                if z < -2.0:
                    z = -2.0
                return float(v_ref) * (1.0 + float(k_speed) * z)

            # ---- 終盤時間 & 相対距離（抜ける回数の根拠）----
            t_final = float(available_m) / max(float(v_ref), 1e-6)

            top_scores = sorted(vals, reverse=True)
            if len(top_scores) >= 3:
                v_fast = _v_from_score(top_scores[0])
                v_mid  = _v_from_score(top_scores[2])
            else:
                v_fast = _v_from_score(mu + sigma)
                v_mid  = _v_from_score(mu)

            gain_m = max(0.0, (float(v_fast) - float(v_mid)) * float(t_final))

            MAX_PASSES = int(gain_m // max(pass_m, 1e-9))
            if MAX_PASSES < 1:
                MAX_PASSES = 1

            # 333/335は最大2、その他は最大3（過剰シャッフル防止）
            cap = 2 if bank_len <= 335 else 3
            if MAX_PASSES > cap:
                MAX_PASSES = cap

            # ---- PASS_DELTAの正規化：available_m依存を弱めて安定化 ----
            base_k = float(globals().get("ko_base_k", 0.040) or 0.040)  # 0.025〜0.060
            score_per_m = base_k * sigma * (1.0 / max(spread, 1e-6)) / max(pass_m, 1e-6)

            PASS_DELTA = score_per_m * pass_m
            cross_mul = 0.35 if bank_len <= 335 else 0.30
            CROSS_DELTA = score_per_m * (cross_mul * pass_m)
            fatigue_delta = 0.35 * PASS_DELTA
            # ====== /PATCH ======

            overtake_cnt = {int(c): 0 for c in order}

            for _ in range(MAX_PASSES):
                swapped = False
                n = len(order)
                moved_this_pass = set()

                for i in range(n - 1):
                    a = order[i]
                    b = order[i + 1]

                    if b in moved_this_pass:
                        continue

                    sa = _final_at(a, i)
                    sb = _final_at(b, i + 1)

                    need = PASS_DELTA + fatigue_delta * float(overtake_cnt.get(b, 0))

                    za = _car_zone_map.get(int(a), "その他")
                    zb = _car_zone_map.get(int(b), "その他")
                    if za != zb:
                        need += CROSS_DELTA

                    if sb >= sa + need:
                        order[i], order[i + 1] = b, a
                        overtake_cnt[b] = overtake_cnt.get(b, 0) + 1
                        moved_this_pass.add(b)
                        swapped = True

                if not swapped:
                    break

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            # 任意：調整が速くなるデバッグ（欲しければ d 表示にも足せる）
            globals()["_overtake_gain_m"] = float(gain_m)
            globals()["_overtake_t_final"] = float(t_final)
            globals()["_overtake_v_ref"] = float(v_ref)

            return order

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            return order

        outs = {}
        for pname, svr in _PATTERNS:
            q = _queue_for_pattern(all_lines, svr)
            main_zone = (svr[0] if (svr and len(svr) >= 1) else "順流")
            outs[pname] = _run_ko(q, main_zone)

        def _fmt_seq(seq, max_n=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if max_n is None:
                max_n = int(globals().get("n_cars", len(xs)))
            xs = xs[:max_n]
            return " → ".join(str(x) for x in xs) if xs else "（なし）"

        out_j = outs.get("順流→渦→逆流") or []
        out_v = outs.get("渦→順流→逆流") or []
        out_u = outs.get("逆流→順流→渦") or []

        

                       # ======================================================
        # 表示用ガード：
        # 1) KO隊列結果がスコア下位を頭に置きすぎる場合だけ補正
        # 2) 主戦ライン先頭が同ライン低スコア車より後ろに落ちるのを防ぐ
        # ※ _run_ko本体は触らない
        # ======================================================
        def _digits_line(x):
            return [int(ch) for ch in str(x) if ch.isdigit()]

        def _display_score_guard(seq, main_line=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if not xs:
                return xs

            score_order = sorted(
                [int(k) for k in score_map.keys()],
                key=lambda c: (-float(score_map.get(c, 0.0)), c)
            )
            score_rank = {c: i + 1 for i, c in enumerate(score_order)}

            # 1) 先頭ガード
            # 先頭がKOスコア5位以下なら、スコア上位3台のうち
            # 元の隊列内で一番前にいる車を先頭へ上げる
            head = xs[0]
            if score_rank.get(head, 99) >= 5:
                candidates = [c for c in score_order[:3] if c in xs]
                if candidates:
                    best = min(candidates, key=lambda c: xs.index(c))
                    xs.remove(best)
                    xs.insert(0, best)

                        # 2) 主戦ライン先頭ガード
            # 例：364なら3がライン先頭。
            # 3よりスコアが低い同ライン車（例：6）が3より前にいるなら、
            # 3をその車の前まで戻す。
            line_members = _digits_line(main_line)
            if len(line_members) >= 2:
                line_head = line_members[0]

                if line_head in xs:
                    line_head_score = float(score_map.get(line_head, 0.0))
                    line_head_idx = xs.index(line_head)

                    lower_mates_before = []
                    for m in line_members[1:]:
                        if m in xs:
                            m_score = float(score_map.get(m, 0.0))
                            if m_score < line_head_score and xs.index(m) < line_head_idx:
                                lower_mates_before.append(m)

                    if lower_mates_before:
                        target_idx = min(xs.index(m) for m in lower_mates_before)
                        xs.remove(line_head)
                        xs.insert(target_idx, line_head)

                        # 3) 最下位スコア車の早出しガード
            # KOスコア最下位の車が3番手以内に残るのを防ぐ
            n_score = len(score_order)

            for bad in list(xs):
                if score_rank.get(bad, 99) == n_score and xs.index(bad) <= 2:
                    xs.remove(bad)

                    # スコア5位以内の車が並んだ最後の直後へ送る
                    insert_pos = 0
                    for i, c in enumerate(xs):
                        if score_rank.get(c, 99) <= 5:
                            insert_pos = i + 1

                    xs.insert(insert_pos, bad)

                        # 4) KO上位車の沈みすぎガード
            # KOスコア上位3車が沈みすぎるのを防ぐ
            # 1位は頭候補、2〜3位は3番手以内を目安に戻す
            for good in score_order[:3]:
                if good not in xs:
                    continue

                r = score_rank.get(good, 99)

                # KO2〜3位が4番手以下なら、3番手以内へ戻す
                if r in (2, 3) and xs.index(good) >= 3:
                    xs.remove(good)
                    target_pos = min(2, len(xs))
                    xs.insert(target_pos, good)

                # KO1位が3番手以下なら、2番手以内へ戻す
                elif r == 1 and xs.index(good) >= 2:
                    xs.remove(good)
                    target_pos = min(1, len(xs))
                    xs.insert(target_pos, good)

            return xs

        out_j = _display_score_guard(out_j, FR_line)
        out_v = _display_score_guard(out_v, VTX_line)
        out_u = _display_score_guard(out_u, U_line)

        # ======================================================
        # H主導ライン3番手以降：
        # 3着内率40%以上なら、
        # 「その戦法の表示1着候補ライン」と同じ場合だけ4番手以内へ移動
        # ======================================================
        try:
            def _display_promote_gid(_car_no):
                try:
                    _car_no = int(_car_no)
                    if isinstance(line_def, dict):
                        for _gid, _mem in line_def.items():
                            _mem2 = [int(x) for x in _mem]
                            if _car_no in _mem2:
                                return _gid
                except Exception:
                    pass
                return None

            def _display_promote_top3_rate(_car_no):
                try:
                    _car_no = int(_car_no)

                    _x1 = globals().get("x1", {})
                    _x2 = globals().get("x2", {})
                    _x3 = globals().get("x3", {})
                    _xo = globals().get("x_out", {})

                    n1 = float(_x1.get(_car_no, _x1.get(str(_car_no), 0)) or 0)
                    n2 = float(_x2.get(_car_no, _x2.get(str(_car_no), 0)) or 0)
                    n3 = float(_x3.get(_car_no, _x3.get(str(_car_no), 0)) or 0)
                    no = float(_xo.get(_car_no, _xo.get(str(_car_no), 0)) or 0)

                    total = n1 + n2 + n3 + no
                    if total <= 0:
                        return None

                    return float((n1 + n2 + n3) / total)

                except Exception:
                    return None

            def _display_promote_to_top4(_seq, _target_car):
                try:
                    _target_car = int(_target_car)
                    _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]

                    if _target_car not in _xs:
                        return _xs

                    _idx = _xs.index(_target_car)

                    # すでに4番手以内なら何もしない
                    if _idx <= 3:
                        return _xs

                    _xs.pop(_idx)
                    _xs.insert(3, _target_car)

                    return _xs

                except Exception:
                    return _seq

            # H主導ラインの3番手以降で、3着内率40%以上の車だけ対象
            _promote_targets = []

            if home_top_gid is not None and isinstance(line_def, dict):
                _h_members = [int(x) for x in line_def.get(home_top_gid, [])]

                if len(_h_members) >= 3:
                    for _car3 in _h_members[2:]:
                        _p3 = _display_promote_top3_rate(_car3)

                        if _p3 is not None and float(_p3) >= 0.40:
                            _promote_targets.append(int(_car3))

            # 各戦法の「表示上の1着候補ライン」と同じ場合だけ、4番手以内へ移動
            for _car3 in _promote_targets:
                _target_gid = _display_promote_gid(_car3)

                if _target_gid is None:
                    continue

                # 順流
                if out_j:
                    _jun_head = int(out_j[0])
                    _jun_gid = _display_promote_gid(_jun_head)
                    if _target_gid == _jun_gid:
                        out_j = _display_promote_to_top4(out_j, _car3)

                # 渦
                if out_v:
                    _vtx_head = int(out_v[0])
                    _vtx_gid = _display_promote_gid(_vtx_head)
                    if _target_gid == _vtx_gid:
                        out_v = _display_promote_to_top4(out_v, _car3)

                # 逆流
                if out_u:
                    _u_head = int(out_u[0])
                    _u_gid = _display_promote_gid(_u_head)
                    if _target_gid == _u_gid:
                        out_u = _display_promote_to_top4(out_u, _car3)

        except Exception as _e:
            note_sections.append(f"※H主導3番手以降・戦法別4番手以内補正エラー：{_e}")
            note_sections.append("")

        # ======================================================
        # 戦法別評価順を保存
        # 後段の「戦法別想定決着率」「2車複候補」で使う
        # ======================================================
        globals()["STYLE_SEQ_MAP"] = {
            "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
            "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
            "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
        }

        # ======================================================
        # 戦法別着順予想を全表示
        # ※ここでは推奨戦法がまだ確定していないため、強調はしない。
        #   後段で「推奨戦法＋コピー用」を別枠表示する。
        # ======================================================
        try:
            def _fmt_seq_full(_seq):
                _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]
                return " → ".join(str(x) for x in _xs) if _xs else "—"

            note_sections.append("【順流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_j))
            note_sections.append("")
            note_sections.append("【渦メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_v))
            note_sections.append("")
            note_sections.append("【逆流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_u))
            note_sections.append("")
        except Exception as _e:
            note_sections.append(f"※戦法別着順予想表示エラー：{_e}")
            note_sections.append("")


    _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr)
    # ここまでで note_sections を確実に保持

        # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）※完全tryゼロ
    # =========================================================
    lines_out = ["＜短評＞"]

    # レースFR：flowのFR（過去出力と同じ定義）
    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:  # NaN
        raceFR = 0.0

    # flowが0なら「混戦度」= 1 - 最大取り分（line_fr_mapがあれば）
    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            if raceFR < 0.0:
                raceFR = 0.0
            if raceFR > 1.0:
                raceFR = 1.0

    # レースFR表示
    lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    # VTX/U はラインFR（ズレ防止）
    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

    

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    # 内訳要約（flow dbg）
    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}

    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 0.0) or 0.0)

        star_txt = "先頭負担:強" if bs <= -0.60 else (
                   "先頭負担:中" if bs <= -0.30 else
                   "先頭負担:小")

        none_txt = "無印押上げ:強" if bn >= 1.20 else (
                   "無印押上げ:中" if bn >= 0.60 else
                   "無印押上げ:小")

        sd_txt = "ライン偏差:大" if sd >= 0.60 else (
                 "ライン偏差:中" if sd >= 0.30 else
                 "ライン偏差:小")

        nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

        lines_out.append(
            f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}"
        )

    # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）
    # =========================================================
    lines_out = ["＜短評＞"]

    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:
        raceFR = 0.0

    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            raceFR = max(0.0, min(1.0, raceFR))

        lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # レースレベル表示
    try:
        lines_out.append(
            f"・レースレベル：{race_level_label}［平均得点={race_level_avg:.2f}／得点差={race_level_spread:.2f}］"
        )
    except Exception:
        pass

    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

        # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    bs = 0.0
    bn = 0.0
    sd = 0.0
    nu = 1.0

    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}
    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 1.0) or 1.0)

    star_txt = "先頭負担:強" if bs <= -0.60 else ("先頭負担:中" if bs <= -0.30 else "先頭負担:小")
    none_txt = "無印押上げ:強" if bn >= 1.20 else ("無印押上げ:中" if bn >= 0.60 else "無印押上げ:小")
    sd_txt = "ライン偏差:大" if sd >= 0.60 else ("ライン偏差:中" if sd >= 0.30 else "ライン偏差:小")
    nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

    lines_out.append(f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}")

    # =========================================================
    # 推奨戦法（優先順位固定・上書き禁止）
    # =========================================================

    try:
        recommend_style = None
        recommend_reason = []
        confidence = "C"

        tenkai_txt = str(
            globals().get("展開評価", "")
            or globals().get("tenkai_eval", "")
            or ""
        ).strip()

        fr_diff = abs(_vtx_fr - _u_fr)

                # =====================================================
        # 現在のライン評価グループでH主導ラインを判定する
        #   旧FR_line / 旧VTX_line / 旧U_line ではなく、
        #   LINE_ZONE_MAP を優先する
        # =====================================================

        def _norm_line_key_for_recommend(ln):
            try:
                if isinstance(ln, (list, tuple)):
                    return "".join(str(int(x)) for x in ln if str(x).isdigit())
            except Exception:
                pass
            return "".join(ch for ch in str(ln) if ch.isdigit())

        def _current_zone_for_line(ln):
            key = _norm_line_key_for_recommend(ln)

            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and key in zmap:
                    return zmap.get(key, "その他")
            except Exception:
                pass

            # 保険：LINE_ZONE_MAPが無い場合だけ旧方式へフォールバック
            if key and key == _norm_line_key_for_recommend(FR_line):
                return "順流"
            if key and key == _norm_line_key_for_recommend(VTX_line):
                return "渦"
            if key and key == _norm_line_key_for_recommend(U_line):
                return "逆流"

            return "その他"

        def _style_fr_for_recommend(style_name):
            if style_name == "順流":
                return float(_lfr(FR_line) if FR_line else 0.0)
            if style_name == "渦":
                return float(_lfr(VTX_line) if VTX_line else 0.0)
            if style_name == "逆流":
                return float(_lfr(U_line) if U_line else 0.0)
            return 0.0

        # =====================================================
        # 1. 展開評価（最優先）
        # =====================================================

        if "混戦" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=混戦"]

        elif "差し" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=差し寄り"]

        elif "先行" in tenkai_txt or "逃げ" in tenkai_txt:
            recommend_style = "順流"
            recommend_reason = ["展開=先行寄り"]

        # =====================================================
        # 2. 短評（ここで確定させる）
        # =====================================================

        if recommend_style is None:

            if bn >= 0.50:
                recommend_style = "渦"
                recommend_reason = ["無印押上げ=中以上"]

            elif sd >= 0.60:
                recommend_style = "順流"
                recommend_reason = ["ライン偏差=大"]

            elif bs <= -0.60 and bn >= 0.50:
                recommend_style = "逆流"
                recommend_reason = ["先頭負担強＋押上げ中以上"]

        # =====================================================
        # 3. FR差（ここは最後）
        # =====================================================

        if recommend_style is None:

            if fr_diff >= 0.02:

                if _u_fr > _vtx_fr:
                    recommend_style = "逆流"
                    recommend_reason = ["逆流FR優勢"]

                else:
                    recommend_style = "順流"
                    recommend_reason = ["VTX優勢"]

        # =====================================================
        # 4. 最終安全側
        # =====================================================

        if recommend_style is None:
            recommend_style = "渦"
            recommend_reason = ["標準判定"]

               
                
                # =====================================================
        # H：推奨理由への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        # =====================================================
        try:
            if home_top_line == "主導なし":
                recommend_reason.append("H主導ラインなし")
            else:
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    recommend_reason.append(f"H主導={h_zone}ライン")
                else:
                    recommend_reason.append("H主導=その他ライン")
        except Exception:
            pass

                
               
                # =====================================================
        # 信頼度
        # =====================================================
        if bn >= 0.50:
            confidence = "B"

        elif fr_diff >= 0.02:
            confidence = "A"

        elif fr_diff >= 0.01:
            confidence = "B"

        else:
            confidence = "C"

                # =====================================================
        # H：低信頼時の推奨戦法切り替え
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で戦法を切り替えない
        # =====================================================
        h_style = None
        h_changed = False

        try:
            if home_top_line != "主導なし":
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    h_style = h_zone
                    h_fr = float(_lfr(h_line) if h_line else 0.0)
                else:
                    h_style = None
                    h_fr = 0.0

                cur_fr = _style_fr_for_recommend(recommend_style)

                if not is_girls_like:
                    if (
                        h_style is not None
                        and h_style != recommend_style
                        and confidence in ("B", "C")
                        and h_fr >= cur_fr - 0.01
                    ):
                        recommend_reason.append(f"H主導により{h_style}寄せ")
                        recommend_style = h_style
                        h_changed = True
                        confidence = "B"
                else:
                    recommend_reason.append("ガールズ/アドバンスのためH主導による戦法変更なし")
        except Exception:
            pass

        # =====================================================
        # H：信頼度への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で信頼度も上下させない
        # =====================================================
        try:
            if not is_girls_like:
                if home_top_line != "主導なし":
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    h_zone = _current_zone_for_line(h_line)

                    h_match = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone == recommend_style
                    )

                    h_conflict = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone != recommend_style
                    )

                    if h_match:
                        if confidence == "C":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "A"

                    elif h_conflict:
                        if confidence == "A":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "C"

        except Exception:
            pass

        # Hで戦法変更した場合は、過信防止で信頼度AをBに抑える
        try:
            if h_changed and confidence == "A":
                confidence = "B"
        except Exception:
            pass

        # H反映チェック表示
        try:
            if h_style is not None:
                if h_changed:
                    recommend_reason.append("H反映=戦法変更あり")
                else:
                    recommend_reason.append("H反映=戦法変更なし")
        except Exception:
            pass

                # =====================================================
        # ガールズ補正
        #   ガールズはライン戦ではないため、
        #   無印押上げだけで渦に寄せすぎない
        # =====================================================
        try:
            if is_girls_like and recommend_style == "渦":
                recommend_style = "順流"
                recommend_reason.append("ガールズ/アドバンスのため渦寄せを順流扱いに補正")
        except Exception:
            pass

                # =====================================================
        # 信頼度の最終補正：展開評価・順当度・上位差を統合
        # =====================================================
        try:
            compact_label = str(globals().get("race_compact_label", ""))
            compact_gap = globals().get("race_compact_gap", None)

            def _down_conf(conf):
                if conf == "A":
                    return "B"
                if conf == "B":
                    return "C"
                return "C"

            conf_down_reasons = []

            # 波乱気味＋上位差小は、信頼度を1段階下げる
            if "波乱気味" in compact_label and compact_gap is not None:
                if float(compact_gap) < 1.0:
                    old_conf = confidence
                    confidence = _down_conf(confidence)
                    if confidence != old_conf:
                        conf_down_reasons.append(
                            f"波乱気味＋上位差小={float(compact_gap):.2f}"
                        )

            # 混戦＋波乱気味はB以上を出しすぎない
            if "混戦" in tenkai_txt and "波乱気味" in compact_label:
                if confidence in ("A", "B"):
                    old_conf = confidence
                    confidence = "C"
                    if confidence != old_conf:
                        conf_down_reasons.append("混戦＋波乱気味")

            # レースFRが不利域なら、AはBへ落とす
            if raceFR >= 0.65 and confidence == "A":
                confidence = "B"
                conf_down_reasons.append(f"レースFR不利域={raceFR:.3f}")

            # ライン偏差大なら、B以上を1段階下げる
            if sd >= 0.60:
                old_conf = confidence
                confidence = _down_conf(confidence)
                if confidence != old_conf:
                    conf_down_reasons.append("ライン偏差大")

            if conf_down_reasons:
                recommend_reason.append(
                    "信頼度補正：" + "／".join(conf_down_reasons)
                )

        except Exception:
            pass

               # =====================================================
        # 推奨戦法を＜短評＞の上に表示
        # =====================================================
        recommend_lines = []
        recommend_lines.append(
            f"推奨戦法：{recommend_style}"
        )

        # =====================================================
        # 買い間違い防止：推奨戦法の着順予想だけを強調表示
        #   目視用：7 → 1 → 5 ...
        #   コピー用：7152364
        # =====================================================
        try:
            _style_seq_map_for_display = globals().get("STYLE_SEQ_MAP", {})
            _recommended_seq = _style_seq_map_for_display.get(recommend_style, [])

            if not _recommended_seq:
                # 保険：STYLE_SEQ_MAPが空の場合は、直前で作った各戦法順から拾う
                _fallback_map = {
                    "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
                    "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
                    "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
                }
                _recommended_seq = _fallback_map.get(recommend_style, [])

            if _recommended_seq:
                _display_seq = " → ".join(str(int(x)) for x in _recommended_seq if str(x).isdigit())
                _copy_seq = "".join(str(int(x)) for x in _recommended_seq if str(x).isdigit())

                recommend_lines.append("")
                recommend_lines.append(f"✅ 推奨戦法：{recommend_style}")
                recommend_lines.append("")
                recommend_lines.append(f"【{recommend_style}メイン着順予想】")
                recommend_lines.append(_display_seq)
                recommend_lines.append("")
                recommend_lines.append(f"コピー用：{_copy_seq}")
                recommend_lines.append("")

                # st.markdown表示時に強調しやすいよう、HTML版も保持しておく
                globals()["RECOMMENDED_STYLE"] = recommend_style
                globals()["RECOMMENDED_STYLE_SEQ"] = _recommended_seq
                globals()["RECOMMENDED_STYLE_COPY"] = _copy_seq
        except Exception as _e:
            recommend_lines.append(f"推奨戦法表示生成不可（{_e}）")
            recommend_lines.append("")

        # =====================================================
        # 2車複 評価1軸候補
        # 推奨戦法の評価順だけを使う
        # =====================================================
        try:
            import math

            def _axis_rank(p):
                if p >= 0.40:
                    return "A", "買い候補強"
                if p >= 0.30:
                    return "B", "買い候補"
                if p >= 0.20:
                    return "C", "オッズ条件付き"
                if p >= 0.10:
                    return "D", "高配当条件"
                return "E", "ケン寄り"

            def _safe_odds_from_prob(p):
                if p <= 1e-12:
                    return None
                return 1.0 / float(p)

            def _style_axis_pairs(seq, score_map):
                """
                評価順seqの1位を軸にした2車複候補を作る。
                推定率はPlackett-Luce型の上位2着内ペア近似。
                """
                xs = []
                seen = set()

                for x in (seq or []):
                    if str(x).isdigit():
                        c = int(x)
                        if c not in seen:
                            seen.add(c)
                            xs.append(c)

                if len(xs) < 2:
                    return None, [], 0.0

                vals = []
                for c in xs:
                    vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                mu = sum(vals) / max(len(vals), 1)
                var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                sdv = var ** 0.5
                if sdv <= 1e-9:
                    sdv = 1.0

                # 温度：小さいほどスコア差を強く見る
                temp = 1.65

                weights = {}
                for c, v in zip(xs, vals):
                    z = (v - mu) / (sdv * temp)
                    z = max(-6.0, min(6.0, z))
                    weights[int(c)] = math.exp(z)

                total_w = sum(weights.values())
                if total_w <= 1e-12:
                    return xs[0], [], 0.0

                axis = xs[0]
                wa = float(weights.get(axis, 0.0))

                pair_rows = []
                for opp in xs[1:]:
                    wb = float(weights.get(int(opp), 0.0))

                    # 無順序2車複：P(axis→opp) + P(opp→axis)
                    p1 = (wa / total_w) * (wb / max(total_w - wa, 1e-12))
                    p2 = (wb / total_w) * (wa / max(total_w - wb, 1e-12))
                    p_pair = max(0.0, p1 + p2)

                    pair_rows.append((axis, int(opp), p_pair))

                # 評価1軸の想定2着内率
                axis_rate = sum(p for _, _, p in pair_rows)

                return axis, pair_rows, axis_rate

            style_seq_map = globals().get("STYLE_SEQ_MAP", {})

            # 推奨戦法に応じた評価順を採用
            selected_style = recommend_style
            selected_seq = style_seq_map.get(selected_style, [])

            # 保険：推奨戦法のseqが空なら順流を使う
            if not selected_seq:
                selected_style = "順流"
                selected_seq = style_seq_map.get("順流", [])

            axis, pair_rows, axis_rate = _style_axis_pairs(selected_seq, score_map)
            axis_rank, axis_label = _axis_rank(axis_rate)

            # 他戦法で軸率が高いものを参考表示する
            ref_msgs = []
            selected_fr = _style_fr_for_recommend(selected_style)

            for other_style in ["順流", "渦", "逆流"]:
                if other_style == selected_style:
                    continue

                other_seq = style_seq_map.get(other_style, [])
                other_axis, _, other_rate = _style_axis_pairs(other_seq, score_map)

                if other_axis is None:
                    continue

                                # 推奨戦法より3%以上高い場合だけ参考表示
                if other_rate >= axis_rate + 0.03:
                    other_fr = _style_fr_for_recommend(other_style)

                    if other_fr < selected_fr - 0.05:
                        ref_msgs.append(
                            f"{other_style}評価1位の{int(other_axis)}は軸率高め。ただし{other_style}FR低めのため参考扱い。"
                        )
                    else:
                        ref_msgs.append(
                            f"{other_style}評価1位の{int(other_axis)}も軸候補。{other_style}警戒。"
                        )

            if axis is not None and pair_rows:
                recommend_lines.append(
                    f"軸評価：{axis_rank}［{axis_label}］"
                    f"（軸想定2着内率 {axis_rate*100:.0f}%）"
                )
                globals()["AXIS_EVAL_TOP_LINE"] = (
                    f"軸評価：{axis_rank}［{axis_label}］"
                    f"（軸想定2着内率 {axis_rate*100:.0f}%）"
                )

                try:
                    _compact_label_for_buy = str(
                        globals().get("race_compact_label", "未判定")
                    )
                    _compact_gap_for_buy = globals().get("race_compact_gap", None)

                    if _compact_gap_for_buy is not None:
                        recommend_lines.append(
                            f"順当度：{_compact_label_for_buy}［上位差={float(_compact_gap_for_buy):.2f}］"
                        )
                    else:
                        recommend_lines.append(
                            f"順当度：{_compact_label_for_buy}"
                        )

                except Exception:
                    pass

                recommend_lines.append("")

                recommend_lines.append("【2車複 評価軸候補】")
                recommend_lines.append(f"基準：{selected_style}メイン")
                recommend_lines.append("2車複想定軸：評価1・評価2")

                # =====================================================
                # 2車複候補一覧＋絞り推奨買目
                # 評価1・評価2を軸にする
                # 候補一覧：重複を削らない
                # 絞り推奨：推定率10%以上、かつ重複削除
                # =====================================================
                SHIBORI_MIN_PROB = 0.10
                shibori_items = []
                shibori_seen = set()

                def _format_nifuku_line(a, b, p):
                    odds = _safe_odds_from_prob(p)
                    if odds is None:
                        return f"{int(a)}-{int(b)}　推定率 0.0% ／ 足切り —"
                    return f"{int(a)}-{int(b)}　推定率 {p*100:.1f}% ／ 足切り {odds:.1f}倍以上"

                def _make_axis_pair_rows(seq, score_map, axis_index=0):
                    """
                    評価順seqの axis_index 番目を軸にした2車複候補を作る。
                    axis_index=0 → 評価1軸
                    axis_index=1 → 評価2軸
                    """
                    xs = []
                    seen = set()

                    for x in (seq or []):
                        if str(x).isdigit():
                            c = int(x)
                            if c not in seen:
                                seen.add(c)
                                xs.append(c)

                    if len(xs) < 2 or axis_index >= len(xs):
                        return None, [], 0.0

                    vals = []
                    for c in xs:
                        vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                    mu = sum(vals) / max(len(vals), 1)
                    var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                    sdv = var ** 0.5
                    if sdv <= 1e-9:
                        sdv = 1.0

                    temp = 1.65

                    weights = {}
                    for c, v in zip(xs, vals):
                        z = (v - mu) / (sdv * temp)
                        z = max(-6.0, min(6.0, z))
                        weights[int(c)] = math.exp(z)

                    total_w = sum(weights.values())
                    if total_w <= 1e-12:
                        return xs[axis_index], [], 0.0

                    axis2 = int(xs[axis_index])
                    wa = float(weights.get(axis2, 0.0))

                    rows = []
                    for opp in xs:
                        opp = int(opp)
                        if opp == axis2:
                            continue

                        wb = float(weights.get(opp, 0.0))

                        # 無順序2車複：P(axis→opp) + P(opp→axis)
                        p1 = (wa / total_w) * (wb / max(total_w - wa, 1e-12))
                        p2 = (wb / total_w) * (wa / max(total_w - wb, 1e-12))
                        p_pair = max(0.0, p1 + p2)

                        rows.append((axis2, opp, p_pair))

                    rate = sum(p for _, _, p in rows)
                    return axis2, rows, rate

                for _axis_index, _label in [(0, "評価1軸"), (1, "評価2軸")]:
                    _axis_car, _rows, _rate = _make_axis_pair_rows(
                        selected_seq,
                        score_map,
                        axis_index=_axis_index
                    )

                    if _axis_car is None or not _rows:
                        continue

                    recommend_lines.append("")
                    recommend_lines.append(f"{_label}：{int(_axis_car)}")

                    _rows_sorted = sorted(
                        _rows,
                        key=lambda t: int(t[1])
                    )

                    for a, b, p in _rows_sorted:
                        line = _format_nifuku_line(a, b, p)
                        recommend_lines.append(line)

                        if float(p) >= SHIBORI_MIN_PROB:
                            # 絞り推奨だけは2車複なので重複削除
                            k = tuple(sorted((int(a), int(b))))
                            if k not in shibori_seen:
                                shibori_seen.add(k)
                                shibori_items.append((a, b, p, line))

                if ref_msgs:
                    recommend_lines.append("参考：" + "／".join(ref_msgs))

                # 絞り推奨買目を別枠で表示
                if shibori_items:
                    recommend_lines.append("")
                    recommend_lines.append("**【絞り推奨買目】（推定率10％以上が基準／重複削除）**")

                    for a, b, p, line in shibori_items:
                        recommend_lines.append(f"**{line}**")


                # =====================================================
                # 3連複：評価1軸・相手3車フォーメーション
                # 思想：123A BOXではなく、評価1を軸にして相手3車から2車を拾う。
                # 点数：3点（例：1-23A-23A = 1-2-3 / 1-2-A / 1-3-A）
                # メインコード側では集計アプリの「安定差表」を持たないため、
                # ここでは推奨戦法の評価順から、評価1を除く上位3車を相手3車にする。
                # =====================================================
                try:
                    def _trio_sorted_key3(a, b, c, sep="-"):
                        vals = sorted([int(a), int(b), int(c)])
                        return sep.join(str(x) for x in vals)

                    _seq_for_trio = _unique_seq(selected_seq)

                    if len(_seq_for_trio) >= 4:
                        _trio_axis = int(_seq_for_trio[0])
                        _trio_opps = [int(x) for x in _seq_for_trio[1:4]]

                        _trio_keys = []
                        _trio_copy_keys = []
                        for _a, _b in combinations(_trio_opps, 2):
                            _trio_keys.append(_trio_sorted_key3(_trio_axis, _a, _b, sep="-"))
                            _trio_copy_keys.append(_trio_sorted_key3(_trio_axis, _a, _b, sep=""))

                        recommend_lines.append("")
                        recommend_lines.append("【3連複 1軸相手3車フォメ】")
                        recommend_lines.append(f"基準：{selected_style}メイン")
                        recommend_lines.append(f"評価1軸：{_trio_axis}")
                        recommend_lines.append("相手3車：" + "・".join(str(x) for x in _trio_opps))
                        recommend_lines.append("買い目：" + " / ".join(_trio_keys))
                        recommend_lines.append("コピー用：" + " / ".join(_trio_copy_keys))
                        recommend_lines.append("考え方：評価1を軸固定し、相手3車から2車を拾う3点。123A BOXから評価1なしの目を削る。")

                except Exception as _e:
                    recommend_lines.append("")
                    recommend_lines.append(f"3連複1軸相手3車フォメ生成不可（{_e}）")

                                # =====================================================
                # 仮想単勝：2車単 軸→全
                # 競輪には単勝がないため、2車単「軸→全」を仮想単勝として扱う
                # 評価1軸・評価2軸を表示
                # =====================================================
                try:
                    def _axis_win_prob(seq, score_map, axis_car):
                        xs = []
                        seen = set()

                        for x in (seq or []):
                            if str(x).isdigit():
                                c = int(x)
                                if c not in seen:
                                    seen.add(c)
                                    xs.append(c)

                        if not xs or int(axis_car) not in xs:
                            return 0.0

                        vals = []
                        for c in xs:
                            vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                        mu = sum(vals) / max(len(vals), 1)
                        var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                        sdv = var ** 0.5
                        if sdv <= 1e-9:
                            sdv = 1.0

                        # 2車複推定率と同じ温度を使用
                        temp = 1.65

                        weights = {}
                        for c, v in zip(xs, vals):
                            z = (v - mu) / (sdv * temp)
                            z = max(-6.0, min(6.0, z))
                            weights[int(c)] = math.exp(z)

                        total_w = sum(weights.values())
                        if total_w <= 1e-12:
                            return 0.0

                        return float(weights.get(int(axis_car), 0.0)) / total_w

                    def _unique_seq(seq):
                        xs = []
                        seen = set()
                        for x in (seq or []):
                            if str(x).isdigit():
                                c = int(x)
                                if c not in seen:
                                    seen.add(c)
                                    xs.append(c)
                        return xs

                    _seq_unique = _unique_seq(selected_seq)

                    recommend_lines.append("")
                    recommend_lines.append("【仮想単勝：2車単 軸→全】")

                    for _axis_index, _label in [(0, "評価1軸"), (1, "評価2軸")]:
                        if _axis_index >= len(_seq_unique):
                            continue

                        _axis_car = int(_seq_unique[_axis_index])
                        axis_win_rate = _axis_win_prob(
                            selected_seq,
                            score_map,
                            _axis_car
                        )

                        if axis_win_rate <= 1e-12:
                            continue

                        theoretical_odds = 1.0 / axis_win_rate

                        # 軸→全の点数。7車なら6点、5車なら4点、9車なら8点。
                        n_tansho_points = max(len(_seq_unique) - 1, 1)

                        # 2車単「軸→全」は、最安目ではなく合成オッズで見る
                        required_composite_odds = theoretical_odds

                        # 実戦では推定誤差を考えて少し上乗せ
                        practical_composite_odds = theoretical_odds * 1.10

                        recommend_lines.append("")
                        recommend_lines.append(f"{_label}：{int(_axis_car)}")
                        recommend_lines.append(
                            f"軸1着推定率：{axis_win_rate*100:.1f}%"
                        )
                        
                        recommend_lines.append(
                            f"2車単 軸→全 必要合成オッズ：{required_composite_odds:.1f}倍以上"
                        )
                        recommend_lines.append(
                            f"実戦目安：合成{practical_composite_odds:.1f}倍以上なら検討"
                        )
                        recommend_lines.append(
                            f"参考：均等買いのトリガミ回避は各目{float(n_tansho_points):.1f}倍以上"
                        )
                except Exception:
                    pass

                recommend_lines.append("")

        except Exception as _e:
            recommend_lines.append(
                f"【2車複 評価1軸候補】生成不可（{_e}）"
            )
            recommend_lines.append("")

        # 推奨理由は短評内に残す
        lines_out.append(
            f"・推奨理由：{'／'.join(recommend_reason)}"
        )

    except Exception as _e:
        recommend_lines = []
        recommend_lines.append(
            f"推奨戦法：判定不可（{_e}）"
        )
        recommend_lines.append("")

    # =====================================================
    # 冒頭表示用：展開評価の直後に軸評価を1行だけ差し込む
    # =====================================================
    try:
        _axis_top_line = globals().get("AXIS_EVAL_TOP_LINE", "")

        if _axis_top_line:
            for _i, _s in enumerate(note_sections):
                if str(_s).startswith("展開評価："):
                    if (
                        _i + 1 >= len(note_sections)
                        or str(note_sections[_i + 1]) != _axis_top_line
                    ):
                        note_sections.insert(_i + 1, _axis_top_line)
                    break

    except Exception:
        pass

    note_sections.extend(recommend_lines)
    note_sections.extend(lines_out)
    note_sections.append("")
    globals()["note_sections"] = note_sections

    globals()["note_sections"] = note_sections

except Exception as _e:
    try:
        ns = globals().get("note_sections", None)
        if not isinstance(ns, list):
            ns = []
            globals()["note_sections"] = ns

        ns.append("")
        ns.append("＜短評＞")
        ns.append(f"・出力生成中に例外が発生しました: {_e}")
        ns.append("判定：混戦")

    except Exception:
        pass

# =========================
note_text = "\n".join(note_sections)
st.markdown("### 📋 note用（コピーエリア）")
st.text_area("ここを選択してコピー", note_text, height=560)
# =========================


# =========================
#  一括置換ブロック ここまで
# =========================
