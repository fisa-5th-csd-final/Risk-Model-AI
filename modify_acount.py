import pandas as pd
import numpy as np

CARD_PATH   = "card.csv"
ACCOUNT_IN  = "account_salary_added.csv"
ACCOUNT_OUT = "account_table_updated.csv"
CLIP_SALARY_RANGE = (200, 2000)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_quarter_from_BAS_YH(bas):
    s = bas.astype(str).str.replace(r"\D", "", regex=True).str.strip()
    s6 = s.str.slice(0, 6).str.pad(6, fillchar="0")
    y = to_num(s6.str[:4]).fillna(2000).astype(int)
    m = to_num(s6.str[4:6]).fillna(1).astype(int).clip(1, 12)
    q = ((m - 1) // 3 + 1).astype(int)
    return q

def build_salary_balance(card):
    num_cols = ["AGE","MBR_RK","TOT_USE_AM","CRDSL_USE_AM","CNF_USE_AM","INTERIOR_AM","INSUHOS_AM",
                "OFFEDU_AM","TRVLEC_AM","FSBZ_AM","SVCARC_AM","DIST_AM","PLSANIT_AM","CLOTHGDS_AM","AUTO_AM"]
    for c in num_cols:
        card[c] = to_num(card[c])

    card["AGE"] = card["AGE"].fillna(40).clip(20, 69)
    card["MBR_RK"] = card["MBR_RK"].fillna(3).round().clip(1, 5)
    for c in num_cols[2:]:
        card[c] = card[c].fillna(0).clip(lower=0)
    card["is_delinquent"] = to_num(card.get("is_delinquent", 0)).fillna(0).clip(0, 1)

    qnum = parse_quarter_from_BAS_YH(card["BAS_YH"])

    denom = card["TOT_USE_AM"].replace(0, np.nan)
    disc = (card["CLOTHGDS_AM"] + card["TRVLEC_AM"] + card["FSBZ_AM"]) / denom
    ess  = (card["INSUHOS_AM"] + card["OFFEDU_AM"] + card["PLSANIT_AM"] + card["DIST_AM"]) / denom
    disc = disc.fillna(0).clip(0, 1)
    ess  = ess.fillna(0).clip(0, 1)

    # Salary 계산
    base_spend = card["MBR_RK"].astype(int).map({1:0.55,2:0.50,3:0.45,4:0.40,5:0.35}).fillna(0.45)
    age_adj = ((35 - card["AGE"]) * 0.004).clip(-0.06, 0.06)
    cat_adj = 0.08*disc - 0.05*ess
    q_adj   = qnum.map({1:-0.01, 2:0.00, 3:+0.01, 4:+0.02}).fillna(0.0)
    delinquent_adj = card["is_delinquent"] * 0.02

    spend_ratio = (base_spend + age_adj + cat_adj + q_adj + delinquent_adj).clip(0.25, 0.70)
    monthly_expense = card["TOT_USE_AM"] / 3.0
    salary = monthly_expense / np.maximum(spend_ratio, 1e-6)
    lo, hi = CLIP_SALARY_RANGE
    salary = np.clip(salary, lo, hi).round().astype(int)

    # Balance 계산
    base_save = card["MBR_RK"].astype(int).map({1:0.05,2:0.08,3:0.12,4:0.16,5:0.20}).fillna(0.12)
    age_save  = ((card["AGE"] - 35) * 0.003).clip(-0.06, 0.06)
    cat_save  = 0.05*ess - 0.03*disc
    delinquent_save_adj = - card["is_delinquent"] * 0.03
    saving_rate = (base_save + age_save + cat_save + delinquent_save_adj).clip(0.00, 0.35)

    monthly_saving = np.maximum(salary - monthly_expense, 0) * saving_rate
    quarter_saving = monthly_saving * 3

    anchor = (card["MBR_RK"].astype(int) * 3_000_000) + ((card["AGE"] - 20) * 100_000)
    heavy  = (card["INTERIOR_AM"] + card["AUTO_AM"]) / card["TOT_USE_AM"].replace(0, np.nan)
    heavy  = heavy.fillna(0).clip(0, 1)
    anchor_adj = -heavy * 1_500_000

    # 💡 balance: 만원 단위 환산 (17500000 → 1750)
    balance_raw = np.maximum(anchor + anchor_adj + quarter_saving, 0)
    balance = (balance_raw // 10_000).astype(int)

    out = card[["customer_id","BAS_YH"]].copy()
    out["salary_calc"]  = salary
    out["balance_calc"] = balance
    return out

def main():
    cols = ["BAS_YH","customer_id","AGE","SEX_CD","MBR_RK","TOT_USE_AM","CRDSL_USE_AM","CNF_USE_AM",
            "INTERIOR_AM","INSUHOS_AM","OFFEDU_AM","TRVLEC_AM","FSBZ_AM","SVCARC_AM","DIST_AM",
            "PLSANIT_AM","CLOTHGDS_AM","AUTO_AM","is_delinquent"]
    card = pd.read_csv(CARD_PATH, usecols=cols, dtype={"customer_id":"string"}, low_memory=False)

    per_row = build_salary_balance(card)
    account = pd.read_csv(ACCOUNT_IN, dtype={"customer_id":"string"}, low_memory=False)

    merged = account.merge(per_row, on=["customer_id","BAS_YH"], how="left")
    merged["salary"]  = merged["salary_calc"].fillna(merged.get("salary"))
    merged["balance"] = merged["balance_calc"].fillna(merged.get("balance"))
    merged = merged.drop(columns=[c for c in ["salary_calc","balance_calc"] if c in merged.columns])
    merged.to_csv(ACCOUNT_OUT, index=False, encoding="utf-8-sig")

    print(f"✅ 완료: {ACCOUNT_OUT} (rows: {len(merged):,})")
    print("💡 balance는 만원 단위(예: 17,500,000 → 1750)로 환산되었습니다.")

if __name__ == "__main__":
    main()
