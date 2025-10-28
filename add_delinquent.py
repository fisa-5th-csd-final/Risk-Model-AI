import pandas as pd
import numpy as np

# ===============================================================
# ① 데이터 불러오기
# ===============================================================
card = pd.read_csv('card.csv')

# 고객 ID 컬럼명 통일
if '고객 ID' in card.columns:
    card.rename(columns={'고객 ID': 'customer_id'}, inplace=True)

# AGE 숫자형 변환
card['AGE'] = pd.to_numeric(card['AGE'], errors='coerce')

# ===============================================================
# ② 회원등급 숫자화 (21=VVIP ~ 25=None)
# ===============================================================
grade_order = {'VVIP': 21, 'VIP': 22, 'Platinum': 23, 'Gold': 24, 'None': 25}
card['MBR_RK_NUM'] = card['MBR_RK'].map(grade_order).fillna(25)

# ===============================================================
# ③ 연체 위험 요인 계산 (현실적 비율 조정)
# ===============================================================

# 1️⃣ 등급 기반 (낮을수록 안정적)
# → 현실 반영: 0.3% ~ 5% 수준
base_risk = np.interp(card['MBR_RK_NUM'], [21, 25], [0.003, 0.05])

# 2️⃣ 소비금액 대비 중앙값 기준 (과소비 위험)
median_spend = card['TOT_USE_AM'].replace(0, np.nan).median()
spend_factor = np.clip(card['TOT_USE_AM'] / median_spend, 0, 5)
spend_risk = 0.006 * spend_factor  # 기존 0.02 → 0.006

# 3️⃣ 신용카드 비중 (신용카드 과다 사용 시 위험)
card['CRDSL_USE_AM'] = card['CRDSL_USE_AM'].fillna(0)
card['CNF_USE_AM'] = card['CNF_USE_AM'].fillna(0)
credit_ratio = np.clip(card['CRDSL_USE_AM'] / (card['CRDSL_USE_AM'] + card['CNF_USE_AM'] + 1), 0, 1)
credit_risk = credit_ratio * 0.015  # 기존 0.05 → 0.015

# 4️⃣ 연령 기반 (20대 초반 / 60대 이상 위험↑)
age = card['AGE'].clip(18, 80)
age_risk = np.where(age < 30, 0.01, np.where(age > 60, 0.008, 0.0))  # 기존 0.04 → 0.01

# 5️⃣ 소비 패턴 (여행↑, 보험↓)
travel_ratio = card.get('TRVLEC_AM', 0) / (card['TOT_USE_AM'] + 1)
health_ratio = card.get('INSUHOS_AM', 0) / (card['TOT_USE_AM'] + 1)
pattern_risk = np.clip(travel_ratio * 0.01 - health_ratio * 0.005, -0.01, 0.02)

# ===============================================================
# ④ 전체 연체 확률 계산
# ===============================================================
card['delinq_prob'] = base_risk + spend_risk + credit_risk + age_risk + pattern_risk
card['delinq_prob'] = card['delinq_prob'].clip(0.001, 0.12)  # 상한 12% 제한

# ===============================================================
# ⑤ 확률 기반으로 연체 여부 생성 (0=정상, 1=연체)
# ===============================================================
np.random.seed(42)
card['is_delinquent'] = (np.random.rand(len(card)) < card['delinq_prob']).astype(int)

# ===============================================================
# ⑥ 불필요 컬럼 제거 + 저장
# ===============================================================
card.drop(columns=['MBR_RK_NUM', 'delinq_prob'], inplace=True)

card.to_csv('card.csv', index=False)

# ===============================================================
# ⑦ 요약 출력
# ===============================================================
delinquency_rate = card['is_delinquent'].mean() * 100
print("💾 card_with_delinquent.csv 저장 완료!")
print(f"✅ 평균 연체율: {delinquency_rate:.2f}%")
print("is_delinquent = 0 → 연체 없음, 1 → 연체 발생")
