"""Stage-1 quick verdict: K2-Horizon-7B (System 1) vs Qwen3.5-4B on identical questions.

HLE: first N of the 513 text-only multiple-choice questions (accuracy vs chance).
Omniscience stand-in: AA public sample, 266 questions, same options and shuffle; the pre-set 0.7 "I don't know" rule
(chosen earlier on Qwen's knowledge proxy) is applied unchanged; plain accuracy shows raw knowledge.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # paths below are relative to the repo root
import json
import numpy as np

def hle(path, n):
    r = json.load(open(path))[:n]
    return np.mean([x["pred"] == x["ans"] for x in r]) * 100, np.mean([100 / len(x["letters"]) for x in r]), len(r)

def omni(path, t=0.7):
    r = json.load(open(path)); P = np.array([x["p"] for x in r]); a = np.array([x["ans"] for x in r])
    ans = P.max(1) >= t; top = P.argmax(1)
    corr = (ans & (top == a)).mean() * 100; wrong = (ans & (top != a)).sum(); idk = (~ans).sum()
    nh = 100 * (1 - wrong / max(1, wrong + idk))
    return np.mean(top == a) * 100, corr, nh, (corr + nh) / 2, ans.mean() * 100, len(r)

k2h = json.load(open("data/k2-horizon/hle.json")); n = len(k2h)
qa, ch, _ = hle("data/hle_4b.json", n); ka, _, _ = hle("data/k2-horizon/hle.json", n)
print(f"HLE (first {n} questions, chance {ch:.1f}%):   Qwen3.5-4B {qa:5.1f}%   |   K2-Horizon-7B {ka:5.1f}%")
for name, p in [("Qwen3.5-4B", "data/omni_public_eval.json"), ("K2-Horizon-7B", "data/k2-horizon/omni_public.json")]:
    pa, c, nh, comp, f, m = omni(p)
    print(f"Omniscience stand-in ({m} q) {name:14s}: knows {pa:5.1f}% | rule 0.7 -> accuracy {c:4.1f}, non-halluc {nh:5.1f}, "
          f"component {comp:5.1f}, answers {f:3.0f}%")
for t in (0.5, 0.6, 0.8, 0.9):
    print(f"   (info only, not tuned) threshold {t}: Qwen component {omni('data/omni_public_eval.json', t)[3]:5.1f} | "
          f"K2 component {omni('data/k2-horizon/omni_public.json', t)[3]:5.1f}")
