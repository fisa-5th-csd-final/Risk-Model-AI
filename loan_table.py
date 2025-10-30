import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------- 설정 ----------
INPUT_PATH = "card.csv"
OUTPUT_PATH = "loan_table_from_card_no_salary.csv"
TARGET_ROWS = 892_997
RNG = np.random.default_rng(42)   # 결정론적 난수
# -------------------------

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def main():
    # 1) load
    # customer_id는 문자열 유지, 나머지는 이후에 수치 변환
    card = pd.read_csv(INPUT_PATH, dtype={"customer_id": "string"})
    n_src = len(card)
    print(f"[INFO] loaded rows: {n_src:,}")

    # 2) 원하는 행 수로 맞추기 (card.csv와 무관하게 정확히 892,997개 생성)
    #    - 원본이 많으면: 비복원 샘플링
    #    - 원본이 적으면: 복원 샘플링(재사용)로 증폭
    if n_src >= TARGET_ROWS:
        base = card.sample(n=TARGET_ROWS, random_state=42, replace=False).reset_index(drop=True)
    else:
        # 복원 샘플링으로 증폭
        idx = RNG.integers(low=0, high=n_src, size=TARGET_ROWS)
        base = card.iloc[idx].reset_index(drop=True)

    n = len(base)
    print(f"[INFO] working rows: {n:,} (target {TARGET_ROWS:,})")

    # 3) 필요한 컬럼 숫자형 변환 및 결측 보정
    for col in ["AGE", "MBR_RK", "TOT_USE_AM"]:
        if col in base.columns:
            base[col] = to_num(base[col])

    # AGE: 결측은 중앙값(없으면 40), 범위 20~69
    if "AGE" not in base.columns:
        base["AGE"] = 40
    age_med = to_num(base["AGE"]).dropna().median()
    if pd.isna(age_med): age_med = 40
    base["AGE"] = to_num(base["AGE"]).fillna(age_med).clip(lower=20, upper=69)

    # MBR_RK: 결측 3, 1~5로 클립(정수)
    if "MBR_RK" not in base.columns:
        base["MBR_RK"] = 3
    mbr_med = to_num(base["MBR_RK"]).dropna().median()
    if pd.isna(mbr_med): mbr_med = 3
    base["MBR_RK"] = to_num(base["MBR_RK"]).fillna(mbr_med).round().clip(1, 5)

    # TOT_USE_AM: 결측 0, 음수 방지
    if "TOT_USE_AM" not in base.columns:
        base["TOT_USE_AM"] = 0
    base["TOT_USE_AM"] = to_num(base["TOT_USE_AM"]).fillna(0).clip(lower=0)

    # 4) balance (② 등급↑ → 잔액↑)
    balance = base["MBR_RK"] * 4_000_000 + RNG.uniform(0, 3_000_000, n)

    # 5) delinquency_count (④,⑤ 소비 많고 나이 낮을수록 ↑)
    base_tot_mean = float(base["TOT_USE_AM"].mean()) if base["TOT_USE_AM"].mean() > 0 else 1.0
    raw_delq = (base["TOT_USE_AM"] / base_tot_mean) * (1.2 + (69 - base["AGE"]) * 0.01)
    raw_delq += RNG.random(n) * 1.2
    delinquency_count = np.clip(raw_delq, 0, 8)

    # 6) interest_rate (⑥ 등급↑ → 금리↓, ⑦ 연체↑ → 금리↑)
    interest_rate = (
        0.12
        - (base["MBR_RK"] - 1) * 0.016
        + delinquency_count * 0.004
        + (RNG.random(n) - 0.5) * 0.004   # ±0.2%p 노이즈
    )

    # 7) principal_amount (⑥,⑨ 등급/잔액 높을수록 ↓, ⑤ 젊을수록 약간 ↑)
    principal = (
        60_000_000
        - base["MBR_RK"] * 7_000_000
        - balance * 0.25
        + (70 - base["AGE"]) * 200_000
    )
    principal_amount = np.maximum(5_000_000, principal)

    # 8) loan_type (⑧ 급여 대신 고등급일수록 담보 비중↑로 대체)
    prob_collateral = np.where(base["MBR_RK"] >= 4, 0.75, 0.35)
    loan_type = np.where(RNG.random(n) < prob_collateral, "1", "0")

    # 9) remaining_principal (⑩ 금리↑ → 잔존비율↑)
    remain_ratio = np.clip(0.4 + (interest_rate - 0.03) * 5, 0.4, 0.95)
    remaining_principal = principal_amount * remain_ratio

    # 10) 금리 유형, 상환 방법, 상환일
    interest_type = RNG.choice(["0", "1"], size=n, p=[0.6, 0.4])
    repayment_method = RNG.choice(["0", "1", "2"], size=n, p=[0.5, 0.3, 0.2])
    start_date = datetime(2025, 1, 1)
    offs = RNG.integers(low=0, high=365 * 5, size=n)
    repayment_date = [start_date + timedelta(days=int(x)) for x in offs]

    # 11) 최종 테이블
    loan_table = pd.DataFrame({
        "customer_id": base.get("customer_id", pd.Series([None]*n, dtype="string")),
        "loan_type": loan_type,
        "principal_amount": principal_amount.round(2),
        "remaining_principal": remaining_principal.round(2),
        "interest_rate": interest_rate.round(5),
        "interest_type": interest_type,
        "repayment_method": repayment_method,
        "repayment_date": [d.strftime("%Y-%m-%d") for d in repayment_date],
    })

    # 12) 검증 및 저장
    assert len(loan_table) == TARGET_ROWS
    loan_table.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {OUTPUT_PATH} ({len(loan_table):,} rows)")

if __name__ == "__main__":
    main()
