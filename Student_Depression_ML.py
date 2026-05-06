# ============================================================
# STUDENT DEPRESSION PREDICTION - FULL ML PROJECT
# By: Narala Gowri Padmavathi
# Dataset: Student Depression Analysis (502 records, 10 features)
# Target: Predict whether a student has Depression (Yes/No)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv("6_Project 2 - Student Depression Analysis.csv")
# ============================================================
# STEP 2: FEATURE ENGINEERING & ENCODING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

df_ml = df.copy()

# Binary encoding
df_ml['Gender'] = df_ml['Gender'].map({'Male': 0, 'Female': 1})
df_ml['Have you ever had suicidal thoughts ?'] = df_ml['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
df_ml['Family History of Mental Illness'] = df_ml['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})
df_ml['Depression'] = df_ml['Depression'].map({'No': 0, 'Yes': 1})

# Ordinal encoding for Sleep Duration (ordered: less sleep = worse)
sleep_map = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4
}
df_ml['Sleep Duration'] = df_ml['Sleep Duration'].map(sleep_map)

# Ordinal encoding for Dietary Habits
diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
df_ml['Dietary Habits'] = df_ml['Dietary Habits'].map(diet_map)

print("Encoded features:")
print(df_ml.head(3))

# ============================================================
# STEP 3: EDA - KEY VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Student Depression - EDA Dashboard', fontsize=16, fontweight='bold', color='#6929C4')

# Plot 1: Depression distribution
axes[0,0].pie(df['Depression'].value_counts(), labels=['No Depression','Depression'],
              autopct='%1.1f%%', colors=['#4CAF50','#E53935'], startangle=90)
axes[0,0].set_title('Depression Distribution', fontweight='bold')

# Plot 2: Academic Pressure vs Depression
df_plot = df_ml.copy()
df_plot['Depression_Label'] = df_plot['Depression'].map({0:'No Depression', 1:'Depression'})

sns.boxplot(data=df_plot, x='Depression_Label', y='Academic Pressure',
            palette={'No Depression':'#4CAF50','Depression':'#E53935'}, ax=axes[0,1])
axes[0,1].set_title('Academic Pressure vs Depression', fontweight='bold')
axes[0,1].set_xlabel('')

# Plot 3: Sleep Duration vs Depression
sleep_dep = df_ml.groupby(['Sleep Duration','Depression']).size().unstack(fill_value=0)
sleep_dep.plot(kind='bar', ax=axes[0,2], color=['#4CAF50','#E53935'], rot=0)
axes[0,2].set_title('Sleep Duration vs Depression', fontweight='bold')
axes[0,2].set_xticklabels(['<5hrs','5-6hrs','7-8hrs','>8hrs'])
axes[0,2].legend(['No Depression','Depression'])

# Plot 4: Financial Stress vs Depression
sns.boxplot(data=df_plot, x='Depression_Label', y='Financial Stress',
            palette={'No Depression':'#4CAF50','Depression':'#E53935'}, ax=axes[1,0])
axes[1,0].set_title('Financial Stress vs Depression', fontweight='bold')
axes[1,0].set_xlabel('')

# Plot 5: Study Hours vs Depression
sns.histplot(data=df_plot, x='Study Hours', hue='Depression_Label',
             palette={'No Depression':'#4CAF50','Depression':'#E53935'}, ax=axes[1,1], bins=10)
axes[1,1].set_title('Study Hours vs Depression', fontweight='bold')

# Plot 6: Correlation Heatmap
corr = df_ml.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
            ax=axes[1,2], annot_kws={'size': 7})
axes[1,2].set_title('Feature Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('1_EDA_Dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA Dashboard saved!")

# ============================================================
# STEP 4: PREPARE TRAIN/TEST DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 60)

X = df_ml.drop('Depression', axis=1)
y = df_ml['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print(f"Features: {X.columns.tolist()}")

# ============================================================
# STEP 5: TRAIN MULTIPLE MODELS & COMPARE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING & COMPARISON")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest':       RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting':   GradientBoostingClassifier(random_state=42, n_estimators=100),
    'KNN':                 KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'Accuracy': round(acc*100,2), 'AUC-ROC': round(auc,3), 'CV Accuracy': round(cv*100,2)}
    print(f"{name:25s} | Accuracy: {acc*100:.2f}%  | AUC: {auc:.3f}  | CV: {cv*100:.2f}%")

# ============================================================
# STEP 6: BEST MODEL - DETAILED EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: BEST MODEL DETAILED EVALUATION")
print("=" * 60)

best_name = max(results, key=lambda x: results[x]['AUC-ROC'])
best_model = models[best_name]
y_pred_best = best_model.predict(X_test_sc)
y_prob_best = best_model.predict_proba(X_test_sc)[:,1]

print(f"Best Model: {best_name}")
print(f"Accuracy:   {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print(f"AUC-ROC:    {roc_auc_score(y_test, y_prob_best):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Depression','Depression']))

# ============================================================
# STEP 7: VISUALIZATION - MODEL RESULTS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Model Evaluation – Best: {best_name}', fontsize=14, fontweight='bold', color='#6929C4')

# Plot 1: Model Comparison Bar Chart
model_names = list(results.keys())
accuracies  = [results[m]['Accuracy'] for m in model_names]
colors = ['#E53935' if m == best_name else '#9C27B0' for m in model_names]
bars = axes[0].bar(model_names, accuracies, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([60, 100])
axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=axes[1],
            xticklabels=['No Dep.','Depression'],
            yticklabels=['No Dep.','Depression'])
axes[1].set_title(f'Confusion Matrix\n{best_name}', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

# Plot 3: ROC Curve for all models
for name, model in models.items():
    y_p = model.predict_proba(X_test_sc)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_p)
    auc = roc_auc_score(y_test, y_p)
    lw = 3 if name == best_name else 1
    axes[2].plot(fpr, tpr, lw=lw, label=f'{name} (AUC={auc:.2f})')
axes[2].plot([0,1],[0,1],'k--', lw=1)
axes[2].set_title('ROC Curves – All Models', fontweight='bold')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('2_Model_Evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nModel Evaluation Chart saved!")

# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
colors_fi = ['#E53935' if v == feat_imp.max() else '#9C27B0' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title(f'Feature Importance – {best_name}', fontsize=14, fontweight='bold', color='#6929C4')
ax.set_xlabel('Importance Score')
for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('3_Feature_Importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature Importance Chart saved!")

# ============================================================
# STEP 9: PREDICT A SAMPLE STUDENT
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: SAMPLE PREDICTION")
print("=" * 60)

# Sample: Female, 21, high academic pressure, low satisfaction,
# less sleep, unhealthy diet, suicidal thoughts, long study hours,
# high financial stress, family history
sample = pd.DataFrame([{
    'Gender': 1,               # Female
    'Age': 21,
    'Academic Pressure': 4.5,
    'Study Satisfaction': 2.0,
    'Sleep Duration': 1,       # Less than 5 hours
    'Dietary Habits': 1,       # Unhealthy
    'Have you ever had suicidal thoughts ?': 1,  # Yes
    'Study Hours': 10,
    'Financial Stress': 4,
    'Family History of Mental Illness': 1        # Yes
}])

sample_sc = scaler.transform(sample)
pred = best_model.predict(sample_sc)[0]
prob = best_model.predict_proba(sample_sc)[0][1]

print(f"Sample Student Profile:")
print(f"  Female, Age 21, High Academic Pressure, Poor Sleep, Unhealthy Diet")
print(f"  High Financial Stress, Family History: Yes, Suicidal Thoughts: Yes")
print(f"\nPrediction:     {'DEPRESSION DETECTED' if pred==1 else 'NO DEPRESSION'}")
print(f"Probability:    {prob*100:.1f}% chance of depression")

# ============================================================
# STEP 10: SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: PROJECT SUMMARY")
print("=" * 60)
print(f"Dataset:         502 students, 10 features, balanced classes")
print(f"Best Model:      {best_name}")
print(f"Best Accuracy:   {results[best_name]['Accuracy']}%")
print(f"Best AUC-ROC:    {results[best_name]['AUC-ROC']}")
print(f"\nAll Model Results:")
for m, r in results.items():
    print(f"  {m:25s}: Acc={r['Accuracy']}%  AUC={r['AUC-ROC']}  CV={r['CV Accuracy']}%")
print("\nOutputs saved:")
print("  1_EDA_Dashboard.png")
print("  2_Model_Evaluation.png")
print("  3_Feature_Importance.png")
print("\nDone!")
