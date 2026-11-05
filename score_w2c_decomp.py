"""Score When2Call yes/no-feature classifiers. Usage: python score_w2c_decomp.py FEATS(.npz or .partial.json)"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

f = sys.argv[1]
if f.endswith(".json"):
    rows = json.load(open(f))
    X = np.array([r["f"] for r in rows]); y = np.array([r["y"] for r in rows]); sp = np.array([r["split"] for r in rows])
    Xtr, ytr, Xte, yte = X[sp == "tr"], y[sp == "tr"], X[sp == "te"], y[sp == "te"]
else:
    d = np.load(f); Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
print(f"train {len(ytr)}  test {len(yte)}")
if len(yte) == 0: sys.exit()
for name, m in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))),
                ("logreg C=0.1", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))),
                ("gbm", HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05, random_state=0))]:
    m.fit(Xtr, ytr); p = m.predict(Xte)
    print(f"{name:14s} acc={np.mean(p == yte):.3f}")
print(confusion_matrix(yte, p))
