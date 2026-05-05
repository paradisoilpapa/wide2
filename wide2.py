import streamlit as st
import pandas as pd

st.title("復習ヴェロビ｜評価軸別 二車複・二車単検証")

st.caption("保存なし。36Rまで入力して、評価1〜7を軸に総流しした場合の的中率・回収率を確認します。")

# =====================================================
# 設定
# =====================================================

MAX_RACES = 36
EVAL_COUNT = 7
UNIT = 100

# =====================================================
# 補助関数
# =====================================================

def to_int_or_none(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return int(x)
    except Exception:
        return None


def pct(n, d):
    if d == 0:
        return 0.0
    return n / d * 100


# =====================================================
# 入力
# =====================================================

st.subheader("入力")

rows = []

for r in range(1, MAX_RACES + 1):
    with st.expander(f"{r}R", expanded=(r <= 3)):

        st.markdown("#### ヴェロビ評価順")

        eval_cols = st.columns(EVAL_COUNT)
        eval_cars = {}

        for ev in range(1, EVAL_COUNT + 1):
            with eval_cols[ev - 1]:
                eval_cars[ev] = to_int_or_none(
                    st.text_input(
                        f"評価{ev}",
                        key=f"race_{r}_eval_{ev}",
                        placeholder=str(ev)
                    )
                )

        st.markdown("#### 結果・払戻")

        result_cols = st.columns(4)

        with result_cols[0]:
            first_car = to_int_or_none(
                st.text_input("1着", key=f"race_{r}_first")
            )

        with result_cols[1]:
            second_car = to_int_or_none(
                st.text_input("2着", key=f"race_{r}_second")
            )

        with result_cols[2]:
            pair_pay = to_int_or_none(
                st.text_input("二車複払戻", key=f"race_{r}_pair_pay", placeholder="円")
            )

        with result_cols[3]:
            exacta_pay = to_int_or_none(
                st.text_input("二車単払戻", key=f"race_{r}_exacta_pay", placeholder="円")
            )

        # 車番 → 評価番号
        car_to_eval = {}
        for ev, car in eval_cars.items():
            if car is not None:
                car_to_eval[car] = ev

        first_eval = car_to_eval.get(first_car)
        second_eval = car_to_eval.get(second_car)

        valid = first_eval is not None and second_eval is not None

        if valid:
            st.info(f"結果評価：1着=評価{first_eval} ／ 2着=評価{second_eval}")

        rows.append({
            "R": r,
            "評価1": eval_cars.get(1),
            "評価2": eval_cars.get(2),
            "評価3": eval_cars.get(3),
            "評価4": eval_cars.get(4),
            "評価5": eval_cars.get(5),
            "評価6": eval_cars.get(6),
            "評価7": eval_cars.get(7),
            "1着車番": first_car,
            "2着車番": second_car,
            "1着評価": first_eval,
            "2着評価": second_eval,
            "二車複払戻": pair_pay,
            "二車単払戻": exacta_pay,
            "有効": valid,
        })

df = pd.DataFrame(rows)
valid_df = df[df["有効"] == True].copy()

# =====================================================
# 集計
# =====================================================

st.divider()
st.subheader("集計結果")

race_count = len(valid_df)
st.write(f"有効入力レース数：**{race_count}R**")

if race_count == 0:
    st.warning("評価順と1着・2着を入力すると集計されます。")
    st.stop()

# =====================================================
# 評価別 1着・2着・2着内率
# =====================================================

rank_rows = []

for ev in range(1, EVAL_COUNT + 1):
    first_count = int((valid_df["1着評価"] == ev).sum())
    second_count = int((valid_df["2着評価"] == ev).sum())
    top2_count = first_count + second_count

    rank_rows.append({
        "評価": f"評価{ev}",
        "1着回数": first_count,
        "1着率": pct(first_count, race_count),
        "2着回数": second_count,
        "2着率": pct(second_count, race_count),
        "2着内回数": top2_count,
        "2着内率": pct(top2_count, race_count),
    })

rank_df = pd.DataFrame(rank_rows)

display_rank = rank_df.copy()
for col in ["1着率", "2着率", "2着内率"]:
    display_rank[col] = display_rank[col].map(lambda x: f"{x:.1f}%")

st.markdown("### 評価別成績")
st.dataframe(display_rank, use_container_width=True)

# =====================================================
# 評価軸別 二車複・二車単 総流し
# =====================================================

axis_rows = []

for ev in range(1, EVAL_COUNT + 1):

    pair_hit = 0
    exacta_hit = 0

    pair_return = 0
    exacta_return = 0

    pair_invest = 0
    exacta_invest = 0

    valid_axis_races = 0

    for _, row in valid_df.iterrows():

        # そのレースで入力されている評価数
        entered_evals = []
        for i in range(1, EVAL_COUNT + 1):
            if pd.notna(row.get(f"評価{i}")):
                entered_evals.append(i)

        if ev not in entered_evals:
            continue

        n_cars = len(entered_evals)

        if n_cars <= 1:
            continue

        points = n_cars - 1
        invest = points * UNIT

        valid_axis_races += 1

        first_eval = row["1着評価"]
        second_eval = row["2着評価"]

        pair_pay = row["二車複払戻"] if pd.notna(row["二車複払戻"]) else 0
        exacta_pay = row["二車単払戻"] if pd.notna(row["二車単払戻"]) else 0

        # 二車複：評価ev - 全
        pair_invest += invest
        if ev == first_eval or ev == second_eval:
            pair_hit += 1
            pair_return += int(pair_pay)

        # 二車単：評価ev → 全
        exacta_invest += invest
        if ev == first_eval:
            exacta_hit += 1
            exacta_return += int(exacta_pay)

    axis_rows.append({
        "評価軸": f"評価{ev}",
        "対象R": valid_axis_races,

        "二車複点数/R": "総流し",
        "二車複的中": pair_hit,
        "二車複的中率": pct(pair_hit, valid_axis_races),
        "二車複投資": pair_invest,
        "二車複払戻": pair_return,
        "二車複回収率": pct(pair_return, pair_invest),
        "二車複収支": pair_return - pair_invest,

        "二車単点数/R": "軸→全",
        "二車単的中": exacta_hit,
        "二車単的中率": pct(exacta_hit, valid_axis_races),
        "二車単投資": exacta_invest,
        "二車単払戻": exacta_return,
        "二車単回収率": pct(exacta_return, exacta_invest),
        "二車単収支": exacta_return - exacta_invest,
    })

axis_df = pd.DataFrame(axis_rows)

display_axis = axis_df.copy()
for col in ["二車複的中率", "二車複回収率", "二車単的中率", "二車単回収率"]:
    display_axis[col] = display_axis[col].map(lambda x: f"{x:.1f}%")

st.markdown("### 評価軸別｜二車複・二車単 総流し成績")
st.dataframe(display_axis, use_container_width=True)

# =====================================================
# 1着評価 → 2着評価 分布
# =====================================================

matrix = pd.DataFrame(
    0,
    index=[f"評価{i}" for i in range(1, EVAL_COUNT + 1)],
    columns=[f"評価{i}" for i in range(1, EVAL_COUNT + 1)]
)

for _, row in valid_df.iterrows():
    a = row["1着評価"]
    b = row["2着評価"]
    if pd.notna(a) and pd.notna(b):
        matrix.loc[f"評価{int(a)}", f"評価{int(b)}"] += 1

st.markdown("### 1着評価 → 2着評価 分布")
st.dataframe(matrix, use_container_width=True)

# =====================================================
# レース別確認
# =====================================================

st.markdown("### レース別確認")

check_df = valid_df[[
    "R",
    "1着車番",
    "2着車番",
    "1着評価",
    "2着評価",
    "二車複払戻",
    "二車単払戻",
]].copy()

check_df["結果評価"] = check_df.apply(
    lambda x: f"{int(x['1着評価'])}-{int(x['2着評価'])}",
    axis=1
)

check_df = check_df[[
    "R",
    "結果評価",
    "1着車番",
    "2着車番",
    "二車複払戻",
    "二車単払戻",
]]

st.dataframe(check_df, use_container_width=True)
