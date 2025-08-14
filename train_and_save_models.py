# train_and_save_models.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# 데이터 로드
df = pd.read_csv('basic_raw_data(원본).csv')
df['IS_DEFECT_ENC'] = df['IS_DEFECT'].map({'FALSE': 0, 'REAL': 1})

feature_ranges = {
    "SIZE_X": (0.0490, 101.9980),
    "SIZE_Y": (0.0490, 60.6270),
    "DEFECT_AREA": (0.0000, 93.0068),
    "SIZE_D": (0.0490, 101.9980),
    "RADIUS": (89.0000, 146758.0000),
    "ANGLE": (0.0000, 359.0000),
    "ALIGNRATIO": (0.0214, 1.0000),
    "SPOTLIKENESS": (0.0467, 31.7792),
    "INTENSITY": (0.0000, 3595.0000),
    "POLARITY": (-1.0000, 1.0000),
    "ENERGY_PARAM": (0.0000, 21723.0000),
    "PATCHNOISE": (1.1321, 1414.0817),
    "RELATIVEMAGNITUDE": (0.0000, 3550.0000),
    "ACTIVERATIO": (0.0000, 1.0000),
    "PATCHDEFECTSIGNAL": (-1466.0000, 7019.0000)
}

feature_m = list(feature_ranges.keys())

# 결함 데이터만 사용
df_defect = df[df['IS_DEFECT_ENC'] == 1].copy()
X = df_defect[feature_m]
y = df_defect['Class']

# 라벨 인코딩
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

# 모델 학습
models = {}
for defect in class_names:
    y_binary = (y == defect).astype(int)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
    )

    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    # 고정된 하이퍼파라미터 (CV 제거)
    xgb_clf.set_params(
        n_estimators=400,
        max_depth=12,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=1
    )

    xgb_clf.fit(X_train_bin, y_train_bin)
    models[defect] = xgb_clf

# 모델 + 라벨인코더 저장
joblib.dump({"models": models, "label_encoder": le, "features": feature_m, "feature_ranges": feature_ranges}, "defect_models.pkl")
print("✅ 모델 저장 완료: defect_models.pkl")
