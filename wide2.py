# -*- coding: utf-8 -*-
# 汎用 期待順位ベース買い目ジェネレーター（競輪/競馬/競艇）5〜18対応
# - オッズ不使用。確率(p_win, p2, p3)と相性係数でスコアリング
# - 複勝はデフォルトで連対率p2を採用（p3切替オプションあり）
# - アンカー（参考予想補助）: 固定(force)/加点(boost)/無視(none)
# - 券種: 単勝/複勝/連複(=2車複/馬連)/ワイド(=拡連複相当)/三連複/連単(=馬単/2連単)/三連単
# - 競技ごとの既定: 競輪/競馬/競艇でON/OFFの初期値を自動切替
# - 競輪7車時は 連対率データ（BASE/K係数）を使った“rank→確率”自動算出 を選択可
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

# -------------------------- 競輪7車用：連対率データ（BASE/K） --------------------------
# 元の7車アプリのテーブルを移植（ユーザー提供の値）
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}  # 3着内率
DEFAULT_P1_RATIO = 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 1e-4) for r in range(1,8)}
K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
K1 = {
    "初日":   {1:0.766, 2:1.282, 3:1.106, 4:1.000, 5:0.746, 6:1.110, 7:0.728},
    "2日目":  {},
    "最終日": {},
}
for r in range(1,8):
    K1["2日目"][r]  = K12["2日目"][r]
    K1["最終日"][r] = K12["最終日"][r]
K3 = {
    "初日":   {1:0.9749518304, 2:1.2857142857, 3:1.1207627119, 4:0.9196428571, 5:0.8010471204, 6:1.0502793296, 7:0.8043478261},
    "2日目":  {1:1.1156069364, 2:0.7457983193, 3:0.8368644068, 4:0.9687500000, 5:1.1020942408, 6:1.0279329609, 7:1.1428571429},
    "最終日": {1:0.8689788054, 2:0.9054621849, 3:1.0381355932, 4:1.1808035714, 5:1.1806282723, 6:0.8770949721, 7:1.1180124224},
}

def parse_verovi_order(s: str) -> list[int]:
    if not s: return []
    t = re.sub(r"[^\d]", "", s)
    if len(t) != 7: return []
    order = [int(ch) for ch in t]
    if sorted(order) != list(range(1,8)): return []
    return order

# rank→p辞書（競輪7車×日別）

def keirin7_rank_to_probs(rank_map: Dict[int,int], day: str):
    p_win, p2, p3 = {}, {}, {}
    for no, rk in rank_map.items():
        p_win[no] = clamp01(BASE_P1[rk]   * K1[day][rk])
        p2[no]    = clamp01(BASE_P12[rk]  * K12[day][rk])
        p3[no]    = clamp01(BASE_PIN3[rk] * K3[day][rk])
    return p_win, p2, p3

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

# 競輪7車の“rank→確率”自動モード
keirin_auto = False
rank_map: Dict[int,int] = {}
sel_day = "初日"
if sport == "競輪":
    st.subheader("競輪専用（7車）: ヴェロビ評価順→確率 自動算出")
    sel_day = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)
    if int(N) == 7:
        keirin_auto = st.checkbox("7車時は 連対率データ（BASE/K）で p_win/p2/p3 を自動算出する", value=True)
        if keirin_auto:
            vorder_str = st.text_input("ヴェロビ評価順（例: 3142576 や 3 1 4 2 5 7 6）", value="")
            vorder = parse_verovi_order(vorder_str)
            if vorder:
                rank_map = {car: i+1 for i,car in enumerate(vorder)}
            else:
                st.info("7桁・重複なしで 1〜7 を並べてください。無効なら下の表入力を使用します。")

# 入力テーブル（手動モード／他競技）
st.subheader("確率とグループ入力（手動または自動の補完）")
nums = list(range(1, int(N) + 1))
init = pd.DataFrame({
    "番号": nums,
    "p_win": [0.0] * len(nums),
    "p2": [0.0] * len(nums),
    "p3": [0.0] * len(nums),
    "group": [1] * len(nums),  # ライン/枠/チーム等
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

# p辞書の決定（自動優先、なければ手動）
if keirin_auto and rank_map:
    p_win_dict, p2_dict, p3_dict = keirin7_rank_to_probs(rank_map, sel_day)
else:
    p_win_dict = {int(r.番号): clamp01(r.p_win) for r in edited.itertuples()}
    p2_dict   = {int(r.番号): clamp01(r.p2) for r in edited.itertuples()}
    # p3は未入力なら p3 := max(p2, p_win)
    p3_dict = {}
    for r in edited.itertuples():
        i = int(r.番号)
        val3 = clamp01(r.p3)
        if val3 == 0.0:
            val3 = max(p2_dict.get(i, 0.0), p_win_dict.get(i, 0.0))
        p3_dict[i] = clamp01(val3)

# ペア係数（同グループ/アンカーboost）
pair_coef = {}
for i, j in combinations(entrants, 2):
    gi = int(edited.loc[edited["番号"] == i, "group"].iloc[0])
    gj = int(edited.loc[edited["番号"] == j, "group"].iloc[0])
    coef = 1.0
    if gi == gj:
        coef *= (1.0 + pair_same_group_bonus)
    if anchor_policy == "boost" and (anchor in (i, j) if anchor is not None else False):
        coef *= anchor_pair_boost
    pair_coef[frozenset({i, j})] = coef

# 順序係数は現状デフォルト=1.0（空dict）
order_coef: Dict[Tuple[int, int], float] = {}

# 出力点数の既定: サイドバー値を採用
fuku_basis = "p2"

# 実行
if st.button("計算／更新", type="primary"):
    res = build_multi_sport_selection(
        N=int(N),
        entrants=entrants,
        anchor=anchor,
        p_win=p_win_dict,
        p2=p2_dict,
        p3=p3_dict,
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
    st.info("左の設定と上の表を入力して『計算／更新』を押してください。")

st.markdown("---")
st.caption("これは娯楽としてのものであり、結果については一切の責任を負いません。")
