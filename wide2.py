# -*- coding: utf-8 -*-
# 期待順位ベース買い目（競輪/競馬/競艇）5〜18対応  — V4: ライン／評価順／脚質 復活
# - オッズ非使用。p_win/p2/p3 を入力 or（競輪7車）ヴェロビ評価順→BASE/Kで自動算出
# - ライン文字列（例: "123 45 6 7" ほか 1-2-3 / 1,2,3 / 1 2 3 でもOK）
# - 脚質は「逃/両/追」を車番で列挙（未指定は既定で補完）
# - ペア係数 = ライン係数 × 脚質係数 ×（任意: 同グループ加点）
# - 券種: 単勝/複勝/連複/ワイド/三連複/連単/三連単（ON/OFF可）
# - アンカー（参考予想補助）: force / boost / none（単・複を◎限定オプション）
# 依存: pip install streamlit pandas

from __future__ import annotations
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
    x = clamp01(x)
    return x / max(1e-9, 1.0 - x)

# -------------------------- 競輪7車用：連対率データ（BASE/K） --------------------------
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}
DEFAULT_P1_RATIO = 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 1e-4) for r in range(1,8)}
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

def parse_verovi_order_generic(s: str, N: int) -> list[int]:
    if not s: return []
    toks = [int(x) for x in re.findall(r"\d+", s)]
    if len(toks) != N: return []
    if sorted(toks) != list(range(1, N+1)): return []
    return toks

# rank→p辞書（競輪7車×日別）

def keirin7_rank_to_probs(rank_map: Dict[int,int], day: str):
    p_win, p2, p3 = {}, {}, {}
    for no, rk in rank_map.items():
        p_win[no] = clamp01(BASE_P1[rk]   * K1[day][rk])
        p2[no]    = clamp01(BASE_P12[rk]  * K12[day][rk])
        p3[no]    = clamp01(BASE_PIN3[rk] * K3[day][rk])
    return p_win, p2, p3

# -------------------------- ライン/脚質 係数 --------------------------
LINE_COEF = {
    "初日":   {"adj":1.35, "same":1.10, "diff":1.00},
    "2日目":  {"adj":1.20, "same":1.05, "diff":1.00},
    "最終日": {"adj":1.25, "same":1.05, "diff":1.00},
}

def style_factor_same_line(pos_a:int, pos_b:int)->float:
    a,b = sorted([pos_a,pos_b])
    if (a,b)==(1,2): return 1.15
    if (a,b)==(2,3): return 1.08
    if (a,b)==(1,3): return 1.03
    return 1.00

STYLE_COEF_DIFF = {
    ("逃","逃"):0.90, ("両","逃"):0.95, ("逃","両"):0.95,
    ("追","追"):1.00, ("両","追"):1.00, ("追","両"):1.00,
    ("逃","追"):1.00, ("追","逃"):1.00, ("両","両"):1.00,
}

def parse_line_pattern(pattern: str, N: int):
    """パターン例:
    - "123 45 6 7"（1桁番号のみの時）
    - "1-2-3  4-5  6  7" / "1,2,3 | 4,5 | 6 | 7"
    - 2桁番号は必ず区切って書く（例: "10 11 12"）
    戻り値: line_id(番号→グループID[文字]) と pos_in_line(番号→1..)
    """
    groups_raw = [g for g in re.split(r"[\s\|]+", pattern.strip()) if g]
    id_map: Dict[int,str] = {}
    pos_map: Dict[int,int] = {}
    used = set()
    gid = 0
    for g in groups_raw:
        # 数字の塊を拾う（例: "1-2-3"→[1,2,3] / "123"→[1,2,3] とみなす）
        toks = [int(x) for x in re.findall(r"\d+", g)]
        if len(toks) <= 1 and all(ch.isdigit() for ch in g) and len(g) > 1 and max(int(c) for c in g) <= 9:
            toks = [int(c) for c in g]  # "123"→[1,2,3]
        if not toks:
            continue
        lid = chr(ord('A')+gid)
        for idx, n in enumerate(toks, start=1):
            if 1 <= n <= N and n not in used:
                id_map[n] = lid
                pos_map[n] = idx
                used.add(n)
        gid += 1
    # 未指定は単独ライン扱い
    for n in range(1, N+1):
        if n not in used:
            lid = chr(ord('A')+gid)
            id_map[n]  = lid
            pos_map[n] = 1
            gid += 1
    return id_map, pos_map

def parse_car_list(s: str, N: int) -> List[int]:
    if not s: return []
    t = (s.replace("、"," ").replace(","," ").replace("　"," ")
         .replace("・"," ").replace("/"," ").replace("-"," "))
    out = []
    for tok in t.split():
        tok = re.sub(r"[^\d]", "", tok)
        if not tok: continue
        try:
            v = int(tok)
            if 1 <= v <= N:
                out.append(v)
        except:  # noqa
            pass
    return out

# -------------------------- スコアエンジン --------------------------

def build_multi_sport_selection(
    N: int,
    entrants: List[int],
    anchor: Optional[int] = None,
    p_win: Optional[Dict[int, float]] = None,
    p2: Optional[Dict[int, float]] = None,
    p3: Optional[Dict[int, float]] = None,
    # ライン/脚質係数
    id_map: Optional[Dict[int,str]] = None,
    pos_map: Optional[Dict[int,int]] = None,
    style_map: Optional[Dict[int,str]] = None,
    day: str = "初日",
    pair_same_group_bonus: float = 0.0,
    anchor_policy: str = "force",
    restrict_anchor_singles: bool = False,
    # 出力点数上限
    top_tan: int = 1,
    top_fuku: int = 1,
    top_ren: int = 3,
    top_wide: int = 3,
    top_triofuku: int = 12,
    top_exacta: int = 6,
    top_trifecta: int = 12,
    trifecta_pool_limit: int = 8,
):
    p_win = p_win or {}
    p2 = p2 or {}
    p3 = p3 or {}
    id_map = id_map or {i: chr(ord('A')+i-1) for i in entrants}
    pos_map = pos_map or {i: 1 for i in entrants}
    style_map = style_map or {i: "追" for i in entrants}

    def line_factor(i:int,j:int)->float:
        li, lj = id_map[i], id_map[j]
        pi, pj = pos_map[i], pos_map[j]
        if li == lj:
            base = LINE_COEF[day]["adj"] if abs(pi-pj)==1 else LINE_COEF[day]["same"]
            style = style_factor_same_line(pi,pj)
        else:
            base = LINE_COEF[day]["diff"]
            style = STYLE_COEF_DIFF.get((style_map[i], style_map[j]),
                                        STYLE_COEF_DIFF.get((style_map[j], style_map[i]), 1.0))
        return base * style

    def pair_coef(i:int,j:int)->float:
        g_bonus = 1.0 + pair_same_group_bonus if id_map[i]==id_map[j] and pair_same_group_bonus>0 else 1.0
        return line_factor(i,j) * g_bonus

    def triple_coef(i:int,j:int,k:int)->float:
        # 幾何平均
        L12 = pair_coef(i,j); L13 = pair_coef(i,k); L23 = pair_coef(j,k)
        return (L12*L13*L23) ** (1/3)

    def sstrength(i:int)->float:
        return max(1e-9, odds_ratio(p_win.get(i,0.0)))

    cand = list(entrants)

    # 単勝
    tan_pool = [anchor] if (restrict_anchor_singles and anchor_policy=="force" and anchor in cand) else cand
    tan = sorted(({"券種":"単勝","脚":(i,),"score":clamp01(p_win.get(i,0.0))} for i in tan_pool),
                 key=lambda x:(-x["score"], x["脚"]))[:max(0, top_tan)]

    # 複勝（p2既定）
    fuku = sorted(({"券種":"複勝","脚":(i,),"score":clamp01(p2.get(i,0.0))} for i in tan_pool),
                  key=lambda x:(-x["score"], x["脚"]))[:max(0, top_fuku)]

    # 連複
    pool_pair = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy=="force" and anchor in cand) else cand
    ren_pairs = []
    for i,j in combinations(pool_pair,2):
        if anchor_policy=="force" and anchor not in (i,j):
            continue
        s = clamp01(p2.get(i,0.0))*clamp01(p2.get(j,0.0))*pair_coef(i,j)
        ren_pairs.append({"券種":"連複","脚":tuple(sorted((i,j))),"score":s})
    ren = sorted(ren_pairs, key=lambda x:(-x["score"], x["脚"]))[:max(0, top_ren)]

    # ワイド
    wide_pairs = []
    for i,j in combinations(pool_pair,2):
        if anchor_policy=="force" and anchor not in (i,j):
            continue
        s = clamp01(p3.get(i,0.0))*clamp01(p3.get(j,0.0))*pair_coef(i,j)
        wide_pairs.append({"券種":"ワイド","脚":tuple(sorted((i,j))),"score":s})
    wide = sorted(wide_pairs, key=lambda x:(-x["score"], x["脚"]))[:max(0, top_wide)]

    # 三連複
    trio_list = []
    pool_trio = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy=="force" and anchor in cand) else cand
    for i,j,k in combinations(pool_trio,3):
        if anchor_policy=="force" and anchor not in (i,j,k):
            continue
        s = clamp01(p3.get(i,0.0))*clamp01(p3.get(j,0.0))*clamp01(p3.get(k,0.0))*triple_coef(i,j,k)
        trio_list.append({"券種":"三連複","脚":tuple(sorted((i,j,k))),"score":s})
    triofuku = sorted(trio_list, key=lambda x:(-x["score"], x["脚"]))[:max(0, top_triofuku)]

    # 連単（Plackett–Luce 近似）
    s_strength = {i:sstrength(i) for i in cand}
    total_s = sum(s_strength.values()) if s_strength else 1.0
    ex_list = []
    pool_order = ([anchor] + [x for x in cand if x != anchor]) if (anchor_policy=="force" and anchor in cand) else cand
    for i,j in permutations(pool_order,2):
        if anchor_policy=="force" and anchor not in (i,j):
            continue
        d1 = total_s; d2 = d1 - s_strength[i]
        if d2<=0: continue
        p_ij = (s_strength[i]/d1)*(s_strength[j]/d2)
        ex_list.append({"券種":"連単","脚":(i,j),"score":p_ij})
    exacta = sorted(ex_list, key=lambda x:(-x["score"], x["脚"]))[:max(0, top_exacta)]

    # 三連単
    pool_tri = sorted(cand, key=lambda i:-s_strength[i])[:min(trifecta_pool_limit, len(cand))]
    if anchor_policy=="force" and anchor in pool_tri:
        pool_tri = [anchor] + [x for x in pool_tri if x!=anchor]
    tri_list2 = []
    sum_s_all = sum(s_strength.get(i,0.0) for i in pool_tri) if pool_tri else 1.0
    for i,j,k in permutations(pool_tri,3):
        if anchor_policy=="force" and anchor not in (i,j,k):
            continue
        d1 = sum_s_all; d2 = d1 - s_strength[i]; d3 = d2 - s_strength[j]
        if d2<=0 or d3<=0: continue
        p_ijk = (s_strength[i]/d1)*(s_strength[j]/d2)*(s_strength[k]/d3)
        tri_list2.append({"券種":"三連単","脚":(i,j,k),"score":p_ijk})
    trifecta = sorted(tri_list2, key=lambda x:(-x["score"], x["脚"]))[:max(0, top_trifecta)]

    return {"単勝":tan, "複勝":fuku, "連複":ren, "ワイド":wide, "三連複":triofuku, "連単":exacta, "三連単":trifecta}

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
    else:
        use_tan, use_fuku = True, True
        use_ren, use_wide, use_triofuku, use_exacta, use_trifecta = True, True, True, True, True

    st.markdown("### アンカー（参考予想補助）")
    anchor_policy = st.radio("アンカーの扱い", ["force", "boost", "none"], index=0, horizontal=True)
    restrict_anchor_singles = st.checkbox("単勝/複勝も◎限定（force時）", value=False)

    st.markdown("### 点数上限")
    top_tan = st.number_input("単勝 上位", 0, 10, 1)
    top_fuku = st.number_input("複勝 上位", 0, 10, 1)
    top_ren = st.number_input("連複 上位", 0, 60, 3)
    top_wide = st.number_input("ワイド 上位", 0, 60, 3)
    top_triofuku = st.number_input("三連複 上位", 0, 240, 12)
    top_exacta = st.number_input("連単 上位", 0, 120, 6)
    top_trifecta = st.number_input("三連単 上位", 0, 240, 12)
    trifecta_pool_limit = st.slider("三連単 計算対象頭数(上位強さで切替)", 3, 18, 8)

# 競輪・開催日（係数切替用）
if sport == "競輪":
    day = st.selectbox("開催日（ライン係数に反映）", ["初日","2日目","最終日"], index=0)
else:
    day = "初日"

# --- ライン入力 ---
st.subheader("ライン入力（例：123 45 6 7  /  1-2-3 | 4-5 | 6 | 7）")
pattern = st.text_input("ラインパターン", value="", help="2桁番号は必ず区切る：例 '10 11 12'。'123'は1,2,3の意味。")
id_map, pos_map = parse_line_pattern(pattern, int(N)) if pattern.strip() else ({i: chr(ord('A')+i-1) for i in range(1,int(N)+1)}, {i:1 for i in range(1,int(N)+1)})

# --- 脚質入力 ---
st.subheader("脚質入力（車番を空白区切りで列挙）")
c1,c2,c3 = st.columns(3)
with c1:
    raw_nige = st.text_input("逃", value="")
with c2:
    raw_ryo  = st.text_input("両", value="")
with c3:
    raw_tsu  = st.text_input("追", value="")
list_nige = parse_car_list(raw_nige, int(N))
list_ryo  = parse_car_list(raw_ryo,  int(N))
list_tsu  = parse_car_list(raw_tsu,  int(N))
dup = (set(list_nige)&set(list_ryo)) | (set(list_nige)&set(list_tsu)) | (set(list_ryo)&set(list_tsu))
if dup:
    st.error(f"脚質の重複指定: {sorted(dup)}")

default_style = st.radio("未指定の既定脚質", options=["追","両","逃"], index=0, horizontal=True)
style_map = {i: ("逃" if i in list_nige else ("両" if i in list_ryo else ("追" if i in list_tsu else default_style))) for i in range(1,int(N)+1)}

# --- 評価順位（ヴェロビ並び）→ 競輪7車のみ 自動p化 ---
st.subheader("評価順位（ヴェロビ並び） — 競輪7車のみ自動算出対応")
verovi_str = st.text_input("ヴェロビ評価順（1行、例: 3 1 4 2 5 7 6）", value="")
rank_map: Dict[int,int] = {}
if verovi_str.strip():
    order = parse_verovi_order_generic(verovi_str, int(N))
    if order:
        rank_map = {car: i+1 for i,car in enumerate(order)}
    else:
        st.info(f"1..{int(N)} を重複なく並べてください。2桁番号は空白等で区切る。")

# --- 確率テーブル（手動入力） ---
st.subheader("確率とグループ（手動入力）")
nums = list(range(1, int(N)+1))
init = pd.DataFrame({"番号":nums, "p_win":[0.0]*len(nums), "p2":[0.0]*len(nums), "p3":[0.0]*len(nums), "group":[1]*len(nums)})
edited = st.data_editor(init, num_rows="fixed", use_container_width=True,
    column_config={"番号": st.column_config.NumberColumn(disabled=True),
                   "p_win": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
                   "p2": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
                   "p3": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
                   "group": st.column_config.NumberColumn(format="%d", min_value=1, max_value=20)})

entrants = edited["番号"].astype(int).tolist()

# --- 競輪7車：rank→確率 の自動化 ---
auto_keirin = (sport=="競輪" and int(N)==7 and bool(rank_map))
if auto_keirin:
    p_win_dict, p2_dict, p3_dict = keirin7_rank_to_probs(rank_map, day)
else:
    p_win_dict = {int(r.番号): clamp01(r.p_win) for r in edited.itertuples()}
    p2_dict   = {int(r.番号): clamp01(r.p2) for r in edited.itertuples()}
    p3_dict   = {}
    for r in edited.itertuples():
        i = int(r.番号)
        v3 = clamp01(r.p3)
        if v3 == 0.0:
            v3 = max(p2_dict.get(i,0.0), p_win_dict.get(i,0.0))
        p3_dict[i] = clamp01(v3)

# --- アンカー & ペア係数設定 ---
colA,colB = st.columns(2)
with colA:
    anchor_choice = st.selectbox("アンカー番号（なし可）", options=["なし"] + entrants, index=0)
    anchor = None if anchor_choice=="なし" else int(anchor_choice)
with colB:
    pair_same_group_bonus = st.slider("同グループ加点(×)", 0.0, 0.5, 0.10, 0.01, help="ラインIDが同一のとき追加で掛ける係数。0で無効。")

# --- 実行ボタン ---
if st.button("計算／更新", type="primary"):
    if dup:
        st.error("脚質の重複指定を解消してください。")
    else:
        res = build_multi_sport_selection(
            N=int(N), entrants=entrants, anchor=anchor,
            p_win=p_win_dict, p2=p2_dict, p3=p3_dict,
            id_map=id_map, pos_map=pos_map, style_map=style_map, day=day,
            pair_same_group_bonus=pair_same_group_bonus,
            anchor_policy=anchor_policy, restrict_anchor_singles=restrict_anchor_singles,
            top_tan=int(top_tan), top_fuku=int(top_fuku), top_ren=int(top_ren), top_wide=int(top_wide),
            top_triofuku=int(top_triofuku), top_exacta=int(top_exacta), top_trifecta=int(top_trifecta),
            trifecta_pool_limit=int(trifecta_pool_limit),
        )

        tabs = []
        if (sport!="競輪" and sport!="競艇") or True:  # 券種ON/OFF（競技既定はサイドバーで）
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
    st.info("左の設定→ライン/脚質→（必要ならヴェロビ順位）→確率表→『計算／更新』の順に操作してください。")

st.markdown("---")
st.caption("これは娯楽としてのものであり、結果については一切の責任を負いません。")


