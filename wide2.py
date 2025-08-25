# -*- coding: utf-8 -*-
# 汎用 期待順位ベース買い目ジェネレーター（競輪/競馬/競艇）5〜18対応
# - オッズ不使用。確率(p_win, p2, p3)と相性係数でスコアリング
# - 複勝はデフォルトで連対率p2を採用（p3切替オプションあり）
# - アンカー（参考予想補助）: 固定(force)/加点(boost)/無視(none)
# - 券種: 単勝/複勝/連複(=2車複/馬連)/ワイド(=拡連複相当)/連単(=馬単/2連単)/三連複/三連単
# - 競技ごとの既定: 競輪/競馬/競艇でON/OFFの初期値を自動切替
# - 競輪拡張（任意トグル）: ヴェロビ評価順→確率の自動算出（5〜9車対応）／ライン入力／脚質入力 → 係数反映
# 依存: pip install streamlit pandas

from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="期待順位ベース買い目（5〜18・参考予想補助）", layout="wide")

# -------------------------- Utils --------------------------

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x >= 1.0:
        return 0.999999
    return x

def odds_ratio(x: float) -> float:
    # 単勝確率→強さ指標(Plackett–Luce用)
    x = clamp01(x)
    return x / max(1e-9, 1.0 - x)

# -------------------------- 競輪拡張: 係数・プロファイル --------------------------
# 7車の実績プロファイル（順位→確率）
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}  # 3着内率
DEFAULT_P1_RATIO = 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 1e-4) for r in range(1,8)}  # 1着率近似

# 日別係数（7点のプロファイル）
K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
K1 = {"初日":{1:0.766,2:1.282,3:1.106,4:1.000,5:0.746,6:1.110,7:0.728}, "2日目":{}, "最終日":{}}
for r in range(1,8):
    K1["2日目"][r]  = K12["2日目"][r]
    K1["最終日"][r] = K12["最終日"][r]
K3 = {
    "初日":   {1:0.9749518304, 2:1.2857142857, 3:1.1207627119, 4:0.9196428571, 5:0.8010471204, 6:1.0502793296, 7:0.8043478261},
    "2日目":  {1:1.1156069364, 2:0.7457983193, 3:0.8368644068, 4:0.9687500000, 5:1.1020942408, 6:1.0279329609, 7:1.1428571429},
    "最終日": {1:0.8689788054, 2:0.9054621849, 3:1.0381355932, 4:1.1808035714, 5:1.1806282723, 6:0.8770949721, 7:1.1180124224},
}

# ライン/脚質係数
LINE_COEF = {"初日":{"adj":1.35,"same":1.10,"diff":1.00}, "2日目":{"adj":1.20,"same":1.05,"diff":1.00}, "最終日":{"adj":1.25,"same":1.05,"diff":1.00}}

def style_factor_same_line(pos_a:int, pos_b:int)->float:
    a,b = sorted([pos_a,pos_b])
    if (a,b)==(1,2): return 1.15
    if (a,b)==(2,3): return 1.08
    if (a,b)==(1,3): return 1.03
    return 1.00

STYLE_COEF_DIFF = {("逃","逃"):0.90, ("両","逃"):0.95, ("逃","両"):0.95,
                   ("追","追"):1.00, ("両","追"):1.00, ("追","両"):1.00,
                   ("逃","追"):1.00, ("追","逃"):1.00, ("両","両"):1.00}

# 7点→N点(5〜9)へ滑らかに拡張する補間（順位位置を0〜1に正規化して線形補間）

def _interp_profile(base7: Dict[int,float], N: int) -> Dict[int,float]:
    xs = [(i-1)/(7-1) for i in range(1,8)]
    ys = [base7[i] for i in range(1,8)]
    def interp(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for a in range(6):
            if xs[a] <= x <= xs[a+1]:
                t = (x - xs[a]) / (xs[a+1]-xs[a])
                return ys[a]*(1-t) + ys[a+1]*t
        return ys[-1]
    out = {}
    for r in range(1, N+1):
        xr = 0.0 if N==1 else (r-1)/(N-1)
        out[r] = float(interp(xr))
    return out

# ヴェロビ評価順→確率（競輪・5〜9車対応）

def verovi_rank_to_probs_keirin(rank_map: Dict[int,int], N:int, day:str):
    # 7点プロファイルをN点に拡張
    bp1  = _interp_profile(BASE_P1,   N)
    bp12 = _interp_profile(BASE_P12,  N)
    bp3  = _interp_profile(BASE_PIN3, N)
    k1   = _interp_profile(K1[day],   N)
    k12  = _interp_profile(K12[day],  N)
    k3   = _interp_profile(K3[day],   N)
    p_win, p2, p3 = {}, {}, {}
    for no, rk in rank_map.items():
        p_win[no] = clamp01(bp1[rk]  * k1[rk])
        p2[no]    = clamp01(bp12[rk] * k12[rk])
        p3[no]    = clamp01(bp3[rk]  * k3[rk])
    return p_win, p2, p3

# 文字列からヴェロビ順位（1..Nの並び）を取得：空白/カンマ/ハイフン or 連続1桁(<=9)に対応

def parse_verovi_order_generic(s: str, N: int) -> List[int]:
    if not s:
        return []
    s_clean = re.sub(r"\s+", "", s)
    if N <= 9 and re.fullmatch(r"[1-9]{%d}" % N, s_clean):  # 例: "3142576"
        order = [int(ch) for ch in s_clean]
    else:
        toks = [int(x) for x in re.findall(r"\d+", s)]
        if len(toks) != N:
            return []
        order = toks
    return order if sorted(order) == list(range(1, N+1)) else []

# ラインパターン→(line_id, pos_in_line)

def parse_line_pattern(pattern: str, N: int):
    groups_raw = [g for g in re.split(r"[\s\|]+", pattern.strip()) if g]
    id_map: Dict[int,str] = {}
    pos_map: Dict[int,int] = {}
    used = set(); gid = 0
    for g in groups_raw:
        toks = [int(x) for x in re.findall(r"\d+", g)]
        # 1桁まとめ記法 "123" → [1,2,3]
        if len(toks) <= 1 and g.isdigit() and len(g) > 1 and max(int(c) for c in g) <= 9:
            toks = [int(c) for c in g]
        if not toks:
            continue
        lid = chr(ord('A')+gid)
        for idx, n in enumerate(toks, start=1):
            if 1 <= n <= N and n not in used:
                id_map[n] = lid
                pos_map[n] = idx
                used.add(n)
        gid += 1
    for n in range(1, N+1):
        if n not in used:
            lid = chr(ord('A')+gid)
            id_map[n]  = lid
            pos_map[n] = 1
            gid += 1
    return id_map, pos_map

# 車番列パーサ

def parse_car_list(s: str, N: int) -> List[int]:
    if not s: return []
    t = (s.replace("、"," ").replace(","," ").replace("　"," ")
         .replace("・"," ").replace("/"," ").replace("-"," "))
    out = []
    for tok in t.split():
        tok = re.sub(r"[^\d]", "", tok)
        if tok.isdigit() and 1 <= int(tok) <= N:
            out.append(int(tok))
    return out

# -------------------------- Core（共通） --------------------------

def build_multi_sport_selection(
    N: int,
    entrants: List[int],
    anchor: Optional[int] = None,
    p_win: Optional[Dict[int, float]] = None,
    p2: Optional[Dict[int, float]] = None,
    p3: Optional[Dict[int, float]] = None,
    pair_coef: Optional[Dict[frozenset, float]] = None,
    triple_coef: Optional[Dict[frozenset, float]] = None,
    order_coef: Optional[Dict[Tuple[int, int], float]] = None,
    # 出力点数上限
    top_tan: int = 1,
    top_fuku: int = 1,
    top_ren: int = 3,
    top_wide: int = 3,
    top_triofuku: int = 12,
    top_exacta: int = 6,
    top_trifecta: int = 12,
    # 3連単の計算対象頭数（重いので上位のみ）
    trifecta_pool_limit: int = 8,
    # アンカー運用
    anchor_policy: str = "force",  # "force" / "boost" / "none"
    # 複勝の基準
    fuku_basis: str = "p2",  # "p2"(既定) / "p3"
    # 単・複をforce時に◎限定にするか
    restrict_anchor_singles: bool = False,
):
    p_win = p_win or {}
    p2 = p2 or {}
    p3 = p3 or {}
    pair_coef = pair_coef or {}
    triple_coef = triple_coef or {}
    order_coef = order_coef or {}

    def c2(i, j):
        return pair_coef.get(frozenset({i, j}), 1.0)

    def c3(i, j, k):
        base = c2(i, j) * c2(i, k) * c2(j, k)
        return base * triple_coef.get(frozenset({i, j, k}), 1.0)

    def ocoef(i, j):
        return order_coef.get((i, j), 1.0)

    cand = list(entrants)

    # 単勝
    tan_pool = [anchor] if (restrict_anchor_singles and anchor_policy == "force" and anchor in cand) else cand
    tan = sorted(
        [{"券種": "単勝", "脚": (i,), "score": clamp01(p_win.get(i, 0.0))} for i in tan_pool],
        key=lambda x: (-x["score"], x["脚"]) 
    )[:max(0, top_tan)]

    # 複勝（デフォルト: p2）
    def fuku_score(i: int) -> float:
        return clamp01(p2.get(i, 0.0)) if fuku_basis == "p2" else clamp01(p3.get(i, 0.0))

    fuku_pool = [anchor] if (restrict_anchor_singles and anchor_policy == "force" and anchor in cand) else cand
    fuku = sorted(
        [{"券種": "複勝", "脚": (i,), "score": fuku_score(i)} for i in fuku_pool],
        key=lambda x: (-x["score"], x["脚"]) 
    )[:max(0, top_fuku)]

    # 連複（2車複/馬連）
    pool_pair = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy == "force" and anchor in cand) else cand
    ren_pairs = []
    for i, j in combinations(pool_pair, 2):
        if anchor_policy == "force" and anchor not in (i, j):
            continue
        s = clamp01(p2.get(i, 0.0)) * clamp01(p2.get(j, 0.0)) * c2(i, j)
        ren_pairs.append({"券種": "連複", "脚": tuple(sorted((i, j))), "score": s})
    ren = sorted(ren_pairs, key=lambda x: (-x["score"], x["脚"]))[:max(0, top_ren)]

    # ワイド（拡連複相当）
    wide_pairs = []
    for i, j in combinations(pool_pair, 2):
        if anchor_policy == "force" and anchor not in (i, j):
            continue
        s = clamp01(p3.get(i, 0.0)) * clamp01(p3.get(j, 0.0)) * c2(i, j)
        wide_pairs.append({"券種": "ワイド", "脚": tuple(sorted((i, j))), "score": s})
    wide = sorted(wide_pairs, key=lambda x: (-x["score"], x["脚"]))[:max(0, top_wide)]

    # 三連複（順序なし）
    triofuku_list = []
    pool_triofuku = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy == "force" and anchor in cand) else cand
    for i, j, k in combinations(pool_triofuku, 3):
        if anchor_policy == "force" and anchor not in (i, j, k):
            continue
        s = clamp01(p3.get(i, 0.0)) * clamp01(p3.get(j, 0.0)) * clamp01(p3.get(k, 0.0)) * c3(i, j, k)
        triofuku_list.append({"券種": "三連複", "脚": tuple(sorted((i, j, k))), "score": s})
    triofuku = sorted(triofuku_list, key=lambda x: (-x["score"], x["脚"]))[:max(0, top_triofuku)]

    # 連単（馬単/2連単）
    s_strength = {i: max(1e-9, odds_ratio(p_win.get(i, 0.0))) for i in cand}
    total_s = sum(s_strength.values()) if s_strength else 1.0

    ex_list = []
    pool_order = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy == "force" and anchor in cand) else cand
    for i, j in permutations(pool_order, 2):
        if anchor_policy == "force" and anchor not in (i, j):
            continue
        denom1 = total_s
        denom2 = denom1 - s_strength[i]
        if denom2 <= 0:
            continue
        p_ij = (s_strength[i] / denom1) * (s_strength[j] / denom2) * ocoef(i, j)
        ex_list.append({"券種": "連単", "脚": (i, j), "score": p_ij})
    exacta = sorted(ex_list, key=lambda x: (-x["score"], x["脚"]))[:max(0, top_exacta)]

    # 三連単（順序あり）
    pool_trifecta = sorted(cand, key=lambda i: -s_strength[i])[:min(trifecta_pool_limit, len(cand))]
    if anchor_policy == "force" and anchor in pool_trifecta:
        pool_trifecta = [anchor] + [x for x in pool_trifecta if x != anchor]

    tri_list = []
    sum_s_all = sum(s_strength.get(i, 0.0) for i in pool_trifecta) if pool_trifecta else 1.0
    for i, j, k in permutations(pool_trifecta, 3):
        if anchor_policy == "force" and anchor not in (i, j, k):
            continue
        denom1 = sum_s_all
        denom2 = denom1 - s_strength[i]
        denom3 = denom2 - s_strength[j]
        if denom2 <= 0 or denom3 <= 0:
            continue
        p_ijk = (s_strength[i] / denom1) * (s_strength[j] / denom2) * (s_strength[k] / denom3) \
                * ocoef(i, j) * ocoef(i, k) * ocoef(j, k)
        tri_list.append({"券種": "三連単", "脚": (i, j, k), "score": p_ijk})
    trifecta = sorted(tri_list, key=lambda x: (-x["score"], x["脚"]))[:max(0, top_trifecta)]

    return {
        "単勝": tan,
        "複勝": fuku,
        "連複": ren,      # 競馬=馬連 / 競輪=2車複
        "ワイド": wide,    # 競馬=ワイド / 競艇=拡連複相当
        "三連複": triofuku,
        "連単": exacta,    # 競馬=馬単 / 競輪=2車単 / 競艇=2連単
        "三連単": trifecta
    }

# -------------------------- UI --------------------------

st.title("期待順位ベース買い目（オッズ非使用／5〜18）")
st.caption("参考予想補助")

with st.sidebar:
    sport = st.selectbox("競技", ["競輪", "競馬", "競艇"], index=0)
    default_N = 6 if sport == "競艇" else (9 if sport == "競輪" else 12)
    N = st.number_input("頭数/艇数", min_value=5, max_value=18, value=default_N, step=1)

    # 券種ON/OFFの既定
    if sport == "競輪":
        use_tan, use_fuku = False, False
        use_ren, use_wide, use_triofuku, use_exacta, use_trifecta = True, True, True, True, True
    elif sport == "競艇":
        use_tan, use_fuku = False, False
        use_ren, use_wide, use_triofuku, use_exacta, use_trifecta = False, True, False, True, True
    else:  # 競馬
        use_tan, use_fuku = True, True
        use_ren, use_wide, use_triofuku, use_exacta, use_trifecta = True, True, True, True, True

    st.markdown("### アンカー（参考予想補助）")
    anchor_policy = st.radio("アンカーの扱い", ["force", "boost", "none"], index=0, horizontal=True)
    restrict_anchor_singles = st.checkbox("単勝/複勝も◎限定（force時）", value=False)

    st.markdown("### 券種 ON/OFF")
    use_tan = st.checkbox("単勝", value=use_tan)
    use_fuku = st.checkbox("複勝（p2）", value=use_fuku)
    use_ren = st.checkbox("連複(2車複/馬連)", value=use_ren)
    use_wide = st.checkbox("ワイド(拡連複相当)", value=use_wide)
    use_triofuku = st.checkbox("三連複", value=use_triofuku)
    use_exacta = st.checkbox("連単(馬単/2連単)", value=use_exacta)
    use_trifecta = st.checkbox("三連単", value=use_trifecta)

    st.markdown("### 点数上限")
    top_tan = st.number_input("単勝 上位", 0, 10, 1)
    top_fuku = st.number_input("複勝 上位", 0, 10, 1)
    top_ren = st.number_input("連複 上位", 0, 60, 3)
    top_wide = st.number_input("ワイド 上位", 0, 60, 3)
    top_triofuku = st.number_input("三連複 上位", 0, 240, 12)
    top_exacta = st.number_input("連単 上位", 0, 120, 6)
    top_trifecta = st.number_input("三連単 上位", 0, 240, 12)
    trifecta_pool_limit = st.slider("三連単 計算対象頭数(上位強さで切替)", 3, 18, 8)

    # ーーーー 競輪拡張（任意） ーーーー
    keirin_ext = False
    day_sel = "初日"; vorder_str = ""; pattern = ""; raw_nige=""; raw_ryo=""; raw_tsu=""; default_style = "追"
    if sport == "競輪":
        with st.expander("競輪拡張（任意: ヴェロビ自動(5〜9車)／ライン／脚質）", expanded=False):
            keirin_ext = st.checkbox("競輪拡張を有効にする", value=False)
            day_sel = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)
            st.caption("拡張OFFなら従来どおり“表の値だけ”で計算")
            st.markdown("**ヴェロビ評価順（5〜9車で有効）**")
            vorder_str = st.text_input("1行（例: 3 1 4 2 5 7 6 / 3142576）", value="")
            st.markdown("**ライン入力** 例: 123 45 6 7 / 1-2-3 | 4-5 | 6 | 7")
            pattern = st.text_input("ラインパターン", value="")
            st.markdown("**脚質入力（車番を空白区切り）**")
            c1,c2,c3 = st.columns(3)
            with c1: raw_nige = st.text_input("逃", value="")
            with c2: raw_ryo  = st.text_input("両", value="")
            with c3: raw_tsu  = st.text_input("追", value="")
            default_style = st.radio("未指定の既定脚質", ["追","両","逃"], index=0, horizontal=True)

# 入力テーブル
st.subheader("確率とグループ入力")
nums = list(range(1, int(N) + 1))
init = pd.DataFrame({
    "番号": nums,
    "p_win": [0.0] * len(nums),
    "p2": [0.0] * len(nums),
    "p3": [0.0] * len(nums),
    "group": [1] * len(nums),  # ライン/枠/チーム等の同一グループ番号
})

edited = st.data_editor(
    init,
    num_rows="fixed",
    use_container_width=True,
    key="prob_input",
    column_config={
        "番号": st.column_config.NumberColumn(disabled=True),
        "p_win": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
        "p2": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
        "p3": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
        "group": st.column_config.NumberColumn(format="%d", min_value=1, max_value=20),
    },
)

entrants = edited["番号"].astype(int).tolist()
anchor_choice = st.selectbox(
    "アンカー番号（なし可）",
    options=["なし"] + entrants,
    index=0,
)
anchor = None if anchor_choice == "なし" else int(anchor_choice)

# 係数設定
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### ペア係数（自動）")
    pair_same_group_bonus = st.slider("同グループ加点(×)", 0.0, 0.5, 0.10, 0.01)
    anchor_pair_boost = st.slider("アンカー絡み加点(×) [boost時]", 1.00, 1.50, 1.05, 0.01)
with col2:
    st.markdown("#### 順序係数（任意）")
    st.caption("既定は1.0。必要なら将来拡張で個別設定可（現状は一律1.0）。")

# 辞書へ整形（手入力ベース）
p_win = {int(r.番号): clamp01(r.p_win) for r in edited.itertuples()}
_p2   = {int(r.番号): clamp01(r.p2)    for r in edited.itertuples()}
# p3は未入力なら p3 := max(p2, p_win)
p_3 = {}
for r in edited.itertuples():
    i = int(r.番号)
    val3 = clamp01(r.p3)
    if val3 == 0.0:
        val3 = max(_p2.get(i, 0.0), p_win.get(i, 0.0))
    _p3 = clamp01(val3)
    p_3[i] = _p3

# ーーーー 競輪拡張の適用（任意） ーーーー
id_map = {i: chr(ord('A')+i-1) for i in range(1,int(N)+1)}
pos_map = {i: 1 for i in range(1,int(N)+1)}
style_map = {i: "追" for i in range(1,int(N)+1)}

if 'keirin_ext' in locals() and keirin_ext:
    # ヴェロビ→確率（5〜9車）。表は上書きせず内部辞書を置き換え
    if sport == "競輪" and 5 <= int(N) <= 9 and vorder_str.strip():
        order = parse_verovi_order_generic(vorder_str, int(N))
        if order:
            rank_map = {car: i+1 for i,car in enumerate(order)}
            auto_win, auto_p2, auto_p3 = verovi_rank_to_probs_keirin(rank_map, int(N), day_sel)
            p_win.update(auto_win); _p2.update(auto_p2); p_3.update(auto_p3)
            with st.expander("自動算出された確率（プレビュー）", expanded=False):
                keys = sorted(auto_win.keys())
                st.dataframe(pd.DataFrame({
                    "番号": keys,
                    "p_win": [auto_win[k] for k in keys],
                    "p2":    [auto_p2[k]   for k in keys],
                    "p3":    [auto_p3[k]   for k in keys],
                }), use_container_width=True)
    # ライン/脚質
    if pattern.strip():
        id_map, pos_map = parse_line_pattern(pattern, int(N))
    list_nige = parse_car_list(raw_nige, int(N))
    list_ryo  = parse_car_list(raw_ryo,  int(N))
    list_tsu  = parse_car_list(raw_tsu,  int(N))
    dup = (set(list_nige)&set(list_ryo)) | (set(list_nige)&set(list_tsu)) | (set(list_ryo)&set(list_tsu))
    if dup:
        st.error(f"脚質の重複指定: {sorted(dup)}")
    style_map = {i: ("逃" if i in list_nige else ("両" if i in list_ryo else ("追" if i in list_tsu else default_style))) for i in range(1,int(N)+1)}
else:
    day_sel = "初日"  # 係数既定

# ペア係数（同グループ/アンカーboost/競輪拡張のライン×脚質）
pair_coef = {}
for i, j in combinations(entrants, 2):
    gi = int(edited.loc[edited["番号"] == i, "group"].iloc[0])
    gj = int(edited.loc[edited["番号"] == j, "group"].iloc[0])
    coef = 1.0
    if gi == gj:
        coef *= (1.0 + pair_same_group_bonus)
    if anchor_policy == "boost" and (anchor in (i, j) if anchor is not None else False):
        coef *= anchor_pair_boost
    # 競輪拡張（ライン×脚質係数）
    if 'keirin_ext' in locals() and keirin_ext:
        li, lj = id_map.get(i), id_map.get(j)
        pi, pj = pos_map.get(i,1), pos_map.get(j,1)
        if li == lj:
            base = LINE_COEF[day_sel]["adj"] if abs(pi-pj)==1 else LINE_COEF[day_sel]["same"]
            style = style_factor_same_line(pi,pj)
        else:
            base = LINE_COEF[day_sel]["diff"]
            style = STYLE_COEF_DIFF.get((style_map.get(i,"追"), style_map.get(j,"追")),
                                        STYLE_COEF_DIFF.get((style_map.get(j,"追"), style_map.get(i,"追")), 1.0))
        coef *= (base * style)
    pair_coef[frozenset({i, j})] = coef

# 順序係数は現状デフォルト=1.0（空dict）
order_coef: Dict[Tuple[int, int], float] = {}

# 出力点数の既定: サイドバー値を採用
fuku_basis = "p2"  # 既定固定（UIで切替不要だが、必要なら簡単に追加可）

# 実行
if st.button("計算／更新", type="primary"):
    res = build_multi_sport_selection(
        N=int(N),
        entrants=entrants,
        anchor=anchor,
        p_win=p_win,
        p2=_p2,
        p3=p_3,
        pair_coef=pair_coef,
        triple_coef={},
        order_coef=order_coef,
        top_tan=int(top_tan),
        top_fuku=int(top_fuku),
        top_ren=int(top_ren),
        top_wide=int(top_wide),
        top_triofuku=int(top_triofuku),
        top_exacta=int(top_exacta),
        top_trifecta=int(top_trifecta),
        trifecta_pool_limit=int(trifecta_pool_limit),
        anchor_policy=anchor_policy,
        fuku_basis=fuku_basis,
        restrict_anchor_singles=restrict_anchor_singles,
    )

    # 競技に応じて表示を限定
    tabs = []
    if use_tan: tabs.append("単勝")
    if use_fuku: tabs.append("複勝")
    if use_ren: tabs.append("連複")
    if use_wide: tabs.append("ワイド")
    if use_triofuku: tabs.append("三連複")
    if use_exacta: tabs.append("連単")
    if use_trifecta: tabs.append("三連単")

    if not tabs:
        st.warning("表示する券種が選択されていません。")
    else:
        t_objs = st.tabs(tabs)
        for t, key in zip(t_objs, tabs):
            with t:
                data = res.get(key, [])
                if not data:
                    st.info("候補なし")
                else:
                    df_out = pd.DataFrame(data)
                    df_out["脚"] = df_out["脚"].apply(lambda x: "-".join(map(str, x)))
                    df_out["score"] = df_out["score"].astype(float)
                    st.dataframe(df_out, use_container_width=True, height=360)
else:
    st.info("左の設定→（任意で競輪拡張）→上の表を入力して『計算／更新』を押してください。")

st.markdown("---")
st.caption("これは娯楽としてのものであり、結果については一切の責任を負いません。")
