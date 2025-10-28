import pandas as pd
import numpy as np

# ===============================================================
# ① 데이터 불러오기 (dtype 지정으로 메모리 절약)
# ===============================================================
dtype_card = {
    'customer_id': 'category',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {'customer_id': 'category'}
dtype_loan = {'customer_id': 'category', 'loan_type': 'category',
              'interest_type': 'category', 'repayment_method': 'category'}

print("📂 파일 로드 중...")
card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)
loan = pd.read_csv('loan.csv', dtype=dtype_loan, low_memory=False)
print("✅ 파일 로드 완료:", len(card), "rows")

# ===============================================================
# ② 고객 기준 병합 (2단계로 나눔)
# ===============================================================
print("🔗 1단계: card + account 병합 중...")
df = card.merge(account, on='customer_id', how='left')
print("✅ 중간 병합 완료:", df.shape)

print("🔗 2단계: loan 병합 중...")
df = df.merge(loan, on='customer_id', how='left')
print("✅ 전체 병합 완료:", df.shape)

# ===============================================================
# ③ 결측치 처리 (벡터화)
# ===============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# 수치형은 전체 중앙값 기반으로 한번에 처리 (루프 X)
median_values = df[num_cols].median(numeric_only=True)
df[num_cols] = df[num_cols].fillna(median_values)

# 범주형은 unknown으로 일괄 처리
df[cat_cols] = df[cat_cols].fillna('unknown')

print("🧹 결측치 처리 완료")

# ===============================================================
# ④ 메모리 최적화 (downcast)
# ===============================================================
df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ===============================================================
# ⑤ 저장 (chunkwriter)
# ===============================================================
print("💾 train_dataset.csv 저장 중...")
df.to_csv('train_dataset.csv', index=False, encoding='utf-8-sig', chunksize=10000)
print("✅ train_dataset.csv 생성 완료!")

print(f"📊 최종 행 수: {len(df):,} | 컬럼 수: {len(df.columns)}")
print(df.head(3))
