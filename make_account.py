import pandas as pd
import numpy as np
import psutil
import threading
import time

np.random.seed(42)

# ===============================================================
# 실시간 진행 모니터
# ===============================================================
stop_flag = False
def monitor(label="account 생성"):
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] {elapsed:6.1f}s | {mem:6.2f} GB | CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# card 데이터 로드
# ===============================================================
print("card.csv 로드 중...")

dtype_card = {
    'customer_id': 'string',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}

card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
print(f"card.csv 로드 완료 ({len(card):,}행, {card['customer_id'].nunique():,}명)")

# ===============================================================
# 고객별 기준급여(base salary) 생성
# ===============================================================
print("\n고객별 기준 급여 생성 중...")

customer_spend = card.groupby('customer_id')['TOT_USE_AM'].mean().rank(pct=True)

grade_weight = {'VVIP': 1.3, 'VIP': 1.2, 'Platinum': 1.1, 'Gold': 1.0, 'None': 0.9}
card['grade_weight'] = card['MBR_RK'].map(grade_weight).fillna(1.0)

age_factor = np.where(card['AGE'] < 30, 0.9, np.where(card['AGE'] < 50, 1.0, 1.1))
sex_factor = np.where(card['SEX_CD'] == '1', 1.05, 1.0)

base_salary_map = customer_spend.apply(
    lambda x: np.random.lognormal(mean=14.5 + x * 0.25, sigma=0.35)
)
card = card.merge(base_salary_map.rename('base_salary'), on='customer_id', how='left')

# ===============================================================
# 분기별 급여 & 잔액 생성
# ===============================================================
print("\n분기별 급여·잔액 생성 중...")

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("account 생성",), daemon=True)
monitor_thread.start()

card['quarter_factor'] = card['BAS_YH'].astype(str).str[-1].map({
    '1': 0.9, '2': 1.0, '3': 1.05, '4': 1.15
}).fillna(1.0)

card['salary'] = (
    card['base_salary'] *
    card['grade_weight'] *
    age_factor *
    sex_factor *
    np.random.uniform(0.9, 1.1, len(card))
)
card['salary'] = np.clip(card['salary'], 2_000_000, 15_000_000)
card['salary'] = (card['salary'] / 10_000).astype(int)

card['TOT_USE_AM'] = card['TOT_USE_AM'].fillna(0)

# 잔액 생성: 급여 10~25배, 시즌·소비 반영, 최소 1000만 원 이상
balance_base = (
    card['salary'] * 10_000 * np.random.uniform(10, 25, len(card)) *
    card['quarter_factor'] *
    np.random.uniform(0.8, 1.2, len(card)) -
    card['TOT_USE_AM'] * np.random.uniform(0.4, 0.8, len(card))
)

card['balance'] = np.clip(balance_base, 1_000_000, 500_000_000)
card['balance'] = (card['balance'] / 10_000).astype(int)

stop_flag = True
monitor_thread.join()

print(f"급여·잔액 생성 완료 | 평균 급여: {card['salary'].mean():,.0f}만원 | 평균 잔액: {card['balance'].mean():,.0f}만원")

# ===============================================================
# account.csv 저장
# ===============================================================
print("\naccount.csv 저장 중...")

account = card[['customer_id', 'BAS_YH', 'salary', 'balance']]
account.to_csv('account.csv', index=False, encoding='utf-8-sig')

print(f"account.csv 생성 완료 ({len(account):,}행, {account['customer_id'].nunique():,}명)")
print(account.head(5))
