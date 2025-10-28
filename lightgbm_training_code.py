# ===============================================================
# 🧠 LightGBM 기반 고객 연체 위험도 예측 AI — 최적화 버전
# ===============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------------------
# 1) 데이터 로드
# ---------------------------------------------------------------
df = pd.read_csv('train_dataset.csv')
print(f"✅ 데이터 불러오기 완료: {df.shape[0]:,} rows, {df.shape[1]} cols")
print("📊 컬럼 예시:", list(df.columns)[:10])

# ---------------------------------------------------------------
# 2) 타깃 생성 (연체 여부: 1회 이상이면 1)
# ---------------------------------------------------------------
if 'delinquency_count' not in df.columns:
    raise ValueError("❌ 'delinquency_count' 컬럼이 없습니다.")

df['y'] = (df['delinquency_count'] > 0).astype(int)

# ---------------------------------------------------------------
# 3) X/y 분리 및 범주형 처리
# ---------------------------------------------------------------
drop_cols = ['customer_id', 'delinquency_count', 'y']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y = df['y']

# object → category 로 캐스팅 (LightGBM이 카테고리로 직접 처리)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category')

print(f"✅ Feature 개수: {X.shape[1]}개 (범주형 {len(cat_cols)}개)")

# ---------------------------------------------------------------
# 4) 학습/검증 분할
# ---------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape}, Valid: {X_val.shape}")

# ---------------------------------------------------------------
# 5) LightGBM Dataset 생성
# ---------------------------------------------------------------
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_cols, reference=train_data)

# 클래스 불균형 보정 (neg/pos)
pos = int(y_train.sum())
neg = int(len(y_train) - pos)
scale_pos_weight = float(neg) / max(1.0, float(pos))
print(f"📐 scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

# ---------------------------------------------------------------
# 6) 하이퍼파라미터 (권장 베이스라인)
# ---------------------------------------------------------------
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l2': 5.0,
    'scale_pos_weight': scale_pos_weight,
    'seed': 42,
    'verbose': -1,
    'num_threads': -1
}

# ---------------------------------------------------------------
# 7) 모델 학습 (EARLY STOPPING)
# ---------------------------------------------------------------
print("\n🚀 모델 학습 중...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    num_boost_round=5000,
    early_stopping_rounds=200,
    verbose_eval=100
)

# ---------------------------------------------------------------
# 8) 성능: AUC + 최적 임계값에서 리포트
# ---------------------------------------------------------------
preds = model.predict(X_val, num_iteration=model.best_iteration)
auc = roc_auc_score(y_val, preds)
print(f"\n🎯 Validation AUC: {auc:.4f}")

# Youden’s J를 최대화하는 임계값
fpr, tpr, thr = roc_curve(y_val, preds)
best_ix = np.argmax(tpr - fpr)
best_thr = float(thr[best_ix])
print(f"🔎 Best threshold (Youden's J): {best_thr:.4f}")

y_hat = (preds >= best_thr).astype(int)
print("\n📊 Classification Report @best_thr\n", classification_report(y_val, y_hat))
print("\n📉 Confusion Matrix @best_thr\n", confusion_matrix(y_val, y_hat))

# ---------------------------------------------------------------
# 9) 위험도(%) + 5단계 버킷(분위수 기반: 각 레벨 인원수 균형)
# ---------------------------------------------------------------
df_val = pd.DataFrame(index=X_val.index)
df_val['pred_prob']    = preds
df_val['risk_percent'] = (preds * 100).round(1)

# 분위수 경계 계산 (중복 경계 보정)
edges = np.quantile(preds, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
edges = np.unique(np.round(edges, 6))
if len(edges) < 6:  # 예측이 치우친 경우 균등 분할로 대체
    edges = np.linspace(0, 1, 6)

labels = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
df_val['risk_level'] = pd.cut(preds, bins=edges, labels=labels, include_lowest=True)

# (선택) 고객 ID 붙이기
if 'customer_id' in df.columns:
    df_val['customer_id'] = df.loc[df_val.index, 'customer_id']

print("\n🎯 예측 결과 샘플")
print(df_val[['customer_id']].join(df_val[['risk_percent','risk_level']], how='left').head()
      if 'customer_id' in df_val.columns else df_val[['risk_percent','risk_level']].head())

# ---------------------------------------------------------------
# 10) 중요 피처 시각화
# ---------------------------------------------------------------
plt.figure(figsize=(10,6))
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 11) 저장: 모델 + 메타(임계값/버킷/범주형)
# ---------------------------------------------------------------
model.save_model('lgbm_delinquency_model.txt')
meta = {
    'best_iteration': int(model.best_iteration),
    'best_threshold': best_thr,
    'quantile_edges': [float(e) for e in edges],
    'categorical_features': list(map(str, cat_cols))
}
with open('inference_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n💾 저장 완료 → lgbm_delinquency_model.txt, inference_meta.json")
