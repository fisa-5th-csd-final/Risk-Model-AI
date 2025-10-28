import pandas as pd
import numpy as np

# ===============================================================
# ① 데이터 불러오기
# ===============================================================
card = pd.read_csv('card.csv')
account = pd.read_csv('account.csv')
loan = pd.read_csv('loan.csv')

# 고객 ID 컬럼명 통일
for df in [card, account, loan]:
    if '고객 ID' in df.columns:
        df.rename(columns={'고객 ID': 'customer_id'}, inplace=True)

# ===============================================================
# ② card에서 고객별 최신 분기(BAS_YH) 데이터만 선택
# ===============================================================
# BAS_YH는 분기 정보 (예: 2024Q1, 2024Q2 등)
# 문자열 비교 시에도 Q4 > Q3 > Q2 > Q1 순서로 정렬 가능하므로 그대로 사용
card['BAS_YH'] = card['BAS_YH'].astype(str)
card_latest = card.sort_values('BAS_YH').groupby('customer_id').tail(1)

print(f"✅ 최신 분기 기준 card 데이터 개수: {len(card_latest)}")

# ===============================================================
# ③ 고객 기준 병합 (loan 기준)
# ===============================================================
df = loan.merge(card_latest[['customer_id', 'AGE', 'MBR_RK']], on='customer_id', how='left') \
         .merge(account[['customer_id']], on='customer_id', how='left')

# ===============================================================
# ④ 회원등급 숫자화 (21=VVIP ~ 25=None)
# ===============================================================
grade_order = {'VVIP': 21, 'VIP': 22, 'Platinum': 23, 'Gold': 24, 'None': 25}
df['MBR_RK'] = df['MBR_RK'].map(grade_order).fillna(25)

# ===============================================================
# ⚙️ 규칙별 생성 + 현실적 예외처리
# ===============================================================

# 규칙① 급여가 높을수록 회원등급이 높다 (2M~20M)
df['salary'] = np.random.normal(
    loc=(26 - df['MBR_RK']) * 2_000_000,
    scale=4e5
)
df['salary'] = np.clip(df['salary'], 2_000_000, 20_000_000)

# 규칙② 회원등급이 높을수록 잔액이 많다 (1M~500M)
df['balance'] = np.random.normal(
    loc=(26 - df['MBR_RK']) * 2_000_000 + df['salary'] * 0.6,
    scale=3e5
)
df['balance'] = np.clip(df['balance'], 1_000_000, 500_000_000)

# ⚠️ 예외처리 ①: 잔액이 급여보다 너무 낮으면 최소 0.2배로 보정
df['balance'] = np.maximum(df['balance'], df['salary'] * 0.2)

# 규칙③ 잔액이 많을수록 총 소비금액이 많다 + 규칙⑤ 나이가 적을수록 소비가 많다
age_effect = (df['AGE'].max() - df['AGE']) / df['AGE'].max()
df['TOT_USE_AM'] = (
    df['balance'] * np.random.uniform(0.05, 0.15, len(df)) * (1 + 0.3 * age_effect)
)
df['TOT_USE_AM'] = np.clip(df['TOT_USE_AM'], 100_000, 10_000_000)

# ⚠️ 예외처리 ②: 소비가 급여보다 너무 크면 급여의 1.5배 이하로 제한
df['TOT_USE_AM'] = np.minimum(df['TOT_USE_AM'], df['salary'] * 1.5)

# 규칙④ 급여 대비 소비가 높을수록 연체횟수가 많다
consume_ratio = df['TOT_USE_AM'] / df['salary']
df['delinquency_count'] = (
    consume_ratio * np.random.uniform(2.0, 4.0, len(consume_ratio))
)
df['delinquency_count'] = np.clip(df['delinquency_count'], 0, 10)

# ⚠️ 예외처리 ③: 소비가 급여 이하이면 연체 횟수를 0~2로 제한
df.loc[consume_ratio <= 1, 'delinquency_count'] = np.random.randint(0, 3, sum(consume_ratio <= 1))

# 규칙⑥ 회원등급이 높을수록 대출이 적고 금리가 낮다
df['principal_amount'] = np.random.normal(
    loc=(df['MBR_RK'] - 20) * 2_000_000,
    scale=5e5
)
df['principal_amount'] = np.clip(df['principal_amount'], 1_000_000, 20_000_000)

# 잔액이 높으면 대출 축소
balance_scale = np.interp(df['balance'],
                          (df['balance'].min(), df['balance'].max()),
                          (1.2, 0.6))
df['principal_amount'] *= balance_scale

# ⚠️ 예외처리 ④: 대출이 급여 × 12 + 잔액보다 크면 상한 제한
df['principal_amount'] = np.minimum(df['principal_amount'], df['salary'] * 12 + df['balance'])

# 규칙⑧ 급여가 높을수록 담보대출 비중이 높다
df['loan_type'] = np.where(df['salary'] >= np.median(df['salary']), 'collateral', 'credit')

# 규칙⑦ 연체횟수가 많을수록 금리가 높다 + 규칙⑥ 연장: 등급이 높을수록 금리 낮음
df['interest_rate'] = np.random.normal(
    loc=(df['MBR_RK'] - 20) * 0.8 + df['delinquency_count'] * 0.3,
    scale=0.4
)
df['interest_rate'] = np.clip(df['interest_rate'], 3.0, 9.0)

# 규칙⑩ 금리가 높을수록 남은 원금 비율이 높다
df['remaining_principal'] = (
    df['principal_amount'] * np.interp(df['interest_rate'],
                                       (df['interest_rate'].min(), df['interest_rate'].max()),
                                       (0.2, 0.9))
)

# ===============================================================
# 🚧 NaN 및 inf 값 완전 제거 (astype 전에 전역 적용)
# ===============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ===============================================================
# 💴 천원(만원 단위 절사)
# ===============================================================
money_cols = ['salary', 'balance', 'TOT_USE_AM', 'principal_amount', 'remaining_principal']

for col in money_cols:
    df[col] = df[col].fillna(0)
    df[col] = np.clip(df[col], 0, None)
    df[col] = (df[col] // 10000).astype('Int64')

# ===============================================================
# 추가 항목 (랜덤)
# ===============================================================
df['interest_type'] = np.where(np.random.rand(len(df)) > 0.7, 'variable', 'fixed')
df['repayment_method'] = np.random.choice(['amortized', 'principal_equal', 'bullet'],
                                          len(df), p=[0.5, 0.3, 0.2])
df['repayment_date'] = pd.to_datetime(
    np.random.choice(pd.date_range('2022-01-01', '2025-12-31'), len(df))
)

# ===============================================================
# ✅ 저장 (loan / account 분리)
# ===============================================================
account_out = df[['customer_id', 'balance']]
loan_out = df[['customer_id', 'salary', 'principal_amount', 'remaining_principal',
               'interest_rate', 'loan_type', 'interest_type', 'repayment_method',
               'repayment_date', 'delinquency_count']]

account_out.to_csv('account.csv', index=False)
loan_out.to_csv('loan.csv', index=False)

print("✅ 최신 분기 기준 + 10개 규칙 + 예외처리 + NaN/inf 처리 + 만원단위 절사 완료!")
