import pandas as pd
import numpy as np
import psutil
import threading
import time
from datetime import datetime

np.random.seed(42)

# ===============================================================
# 실시간 모니터링 스레드
# ===============================================================
stop_flag = False

def monitor(label="loan 생성"):
    """5초마다 진행률, CPU, 메모리 사용량 출력"""
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] {elapsed:6.1f}s | {mem:6.2f} GB | CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# 데이터 로드 및 병합
# ===============================================================
print("Step 1: 데이터 로드 및 병합 중...")

card = pd.read_csv("card.csv", low_memory=False)
account = pd.read_csv("account.csv", low_memory=False)

if '고객 ID' in card.columns:
    card.rename(columns={'고객 ID': 'customer_id'}, inplace=True)

df = card.merge(account, on=['customer_id', 'BAS_YH'], how='inner')
print(f"병합 완료: {len(df):,}행 ({df['customer_id'].nunique():,}명)")

# ===============================================================
# NaN / inf 사전 제거
# ===============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ===============================================================
# 회원등급 수치화
# ===============================================================
grade_map = {'VVIP': 21, 'VIP': 22, 'Platinum': 23, 'Gold': 24, 'None': 25}
df['grade_num'] = df['MBR_RK'].map(grade_map).fillna(25).astype(int)

# ===============================================================
# 기본 지표 계산
# ===============================================================
consume_ratio = (df['TOT_USE_AM'] / np.maximum(df['salary'] * 10_000, 1)).clip(0, 5)
wealth_ratio = (df['balance'] / np.maximum(df['salary'] * 10_000, 1)).clip(0, 50)
age_factor = np.clip((df.get('AGE', 40) - 30) / 30, -0.5, 0.5)

# ===============================================================
# 대출 유형 (확률 기반)
# ===============================================================
salary_percentile = df['salary'].rank(pct=True)
prob_collateral = 0.2 + 0.6 * salary_percentile
df['loan_type'] = np.where(np.random.rand(len(df)) < prob_collateral, 'collateral', 'credit')

# ===============================================================
# 원금 (principal_amount)
# ===============================================================
base_principal = (df['salary'] * np.random.uniform(5, 15, len(df))) * (26 - df['grade_num']) / 6
base_principal /= np.maximum(wealth_ratio, 1)

df['principal_amount'] = np.where(
    df['loan_type'] == 'collateral',
    np.clip(base_principal, 1_000_000, 500_000_000),
    np.clip(base_principal, 500_000, 150_000_000)
)

# ===============================================================
# 금리 (interest_rate)
# ===============================================================
base_rate = (df['grade_num'] - 20) * 0.5 + consume_ratio * 0.7 - age_factor * 0.8
noise = np.random.normal(0, 0.4, len(df))
df['interest_rate'] = np.clip(2.5 + base_rate + noise, 2.5, 9.5).round(2)

# ===============================================================
# 금리 유형 / 상환 방식
# ===============================================================
df['interest_type'] = np.where(np.random.rand(len(df)) < 0.3, 'variable', 'fixed')
df['repayment_method'] = np.random.choice(
    ['amortized', 'principal_equal', 'bullet'],
    size=len(df),
    p=[0.5, 0.3, 0.2]
)

# ===============================================================
# 상환일 (각 분기 이후 1~3년)
# ===============================================================
def parse_quarter_to_date(bas_yh):
    if isinstance(bas_yh, str) and 'Q' in bas_yh:
        year = int(bas_yh[:4])
        quarter = int(bas_yh[-1])
        month = quarter * 3
        return datetime(year, month, 28)
    else:
        return datetime(2024, 3, 31)

df['quarter_date'] = df['BAS_YH'].apply(parse_quarter_to_date)

df['repayment_date'] = [
    (qd + pd.to_timedelta(np.random.randint(365, 365*3), unit='D')).date()
    for qd in df['quarter_date']
]

today = pd.Timestamp('2025-10-29')
df['repayment_date'] = pd.to_datetime(df['repayment_date'])
df = df[df['repayment_date'] > today]

# ===============================================================
# 남은 원금 (remaining_principal)
# ===============================================================
rate_scaled = (df['interest_rate'] - 2.5) / 7
df['remaining_principal'] = df['principal_amount'] * (0.2 + 0.7 * rate_scaled)
df['remaining_principal'] = np.clip(df['remaining_principal'], 0, df['principal_amount'])

# ===============================================================
# 연체 여부 (5% 내외)
# ===============================================================
# 계산 전에 최신 상태의 consume_ratio, wealth_ratio를 다시 한 번 df 기준으로 생성
df['consume_ratio'] = (df['TOT_USE_AM'] / np.maximum(df['salary'] * 10_000, 1)).clip(0, 5)
df['wealth_ratio'] = (df['balance'] / np.maximum(df['salary'] * 10_000, 1)).clip(0, 50)

# 연체 위험 점수 계산 (소비↑, 금리↑, 부↑ → 위험↑)
risk_score = (
    1.5 * np.log1p(df['consume_ratio'] * 10) +   # 과소비 영향 강화
    1.0 * (df['interest_rate'] / 9.5) -          # 금리 높을수록 위험↑
    0.7 * np.log1p(df['wealth_ratio'] + 1) +     # 자산 많을수록 위험↓
    np.random.normal(0, 0.3, len(df))            # 개별 편차 추가 (현실 반영)
)

# 스케일 조정 (정규화)
risk_score = (risk_score - risk_score.mean()) / (risk_score.std() + 1e-6)

# 로지스틱 함수를 이용한 확률화 (sigmoid)
risk_prob = 1 / (1 + np.exp(-risk_score))

# 확률 분포를 이용한 연체 여부 결정 (평균 약 5%)
threshold = np.quantile(risk_prob, 0.95)
df['is_delinquent'] = (risk_prob > threshold).astype(int)

# 실제 연체율 계산
real_rate = df['is_delinquent'].mean() * 100
print(f"현실적 연체율: {real_rate:.2f}%")

# ===============================================================
# 단위 절사 및 저장 (진행률 모니터링 포함)
# ===============================================================
print("\nStep 2: loan.csv 저장 중...")
stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("loan 저장",), daemon=True)
monitor_thread.start()

money_cols = ['principal_amount', 'remaining_principal']
for col in money_cols:
    df[col] = (df[col] // 10_000).astype(int)

loan = df[['customer_id', 'BAS_YH', 'loan_type', 'principal_amount',
           'remaining_principal', 'interest_rate', 'interest_type',
           'repayment_method', 'repayment_date', 'is_delinquent']]

chunk_size = 100_000
first = True
for i in range(0, len(loan), chunk_size):
    loan.iloc[i:i+chunk_size].to_csv(
        "loan.csv",
        mode='w' if first else 'a',
        index=False,
        header=first,
        encoding='utf-8-sig'
    )
    first = False
    print(f"  {i+chunk_size:,}행까지 저장 완료")

stop_flag = True
monitor_thread.join()

print(f"loan.csv 생성 완료 ({len(loan):,}행)")
print(loan.head(5))
