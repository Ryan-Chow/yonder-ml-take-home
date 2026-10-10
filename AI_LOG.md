# AI Usage Log

**TL;DR:** I used Claude Code (agent mode) to build the pipeline from one detailed prompt I wrote
with help from Claude chat, then reviewed the outputs myself and ran the real-world video test.
Full agent log, including the verbatim prompt: [docs/AGENT_RUN_LOG.md](docs/AGENT_RUN_LOG.md).

**Tools:** Claude Code (Claude Opus 5.5) for code and experiments; Claude chat for planning the prompt,
explaining results, and reviewing my write-up.

---

## 1. Planning the approach
- **Asked:** I read the take-home README, drafted a one-shot prompt, and asked Claude chat to refine it.
  I made sure the plan covered a leakage check, grouped split, CLAHE, an ablation, per-class metrics,
  error analysis, and honest logging.
- **Kept / changed:** Kept the plan. I chose the ML track, YOLOv8n, and training on my MacBook (MPS).
  I downloaded the dataset ZIP manually instead of using a Roboflow API key.
- **AI got wrong:** The prompt assumed leakage would show up as shared `.rf.` filename stems. That turned
  out to be wrong for this dataset (the agent found it by reading `sampling/`). The prompt also listed
  a repo path and a pre-unzipped dataset that didn't match my machine.

## 2. Building the pipeline (autonomous agent run)
- **Asked:** One detailed prompt to build everything end to end (full text in AGENT_RUN_LOG.md).
- **Kept / changed:** Kept the agent's code. It deviated from my prompt in places I agree with:
  - extended the ablation to 2×2 (CLAHE × custom augmentation) with cluster-bootstrap CIs
  - detected leakage through capture sequences, not just filenames (54% of val had a sibling in train)
  - evaluated at the 512 input that actually ships (`rect=False`)
  - trained a separate model on the original split to measure leakage properly
- **AI got wrong:**
  - evaluated at 544 px instead of the shipped 512 (caught by an ONNX vs PyTorch mismatch)
  - two "wrong-class" errors were actually mislabeled mallets in the dataset
  - image-level bootstrap understated uncertainty; switched to group-level
  - several false claims in its own write-up, removed in a fact-check pass
  - CLAHE gave no measurable benefit, despite its original justification
- **What else I verified**
  - checked `audit_samples.jpg`; boxes lined up with the objects
  - leakage: 54% of original val images came from same source as training image
  - shipping decision: reviewed why clahe_only shipped despite lowest F1. mAP50 comes first, which rules out baseline; among arms statistically tied on mAP50, clahe_only has the best mAP50-95. F1 differences within noise, this is a tiebreak

## 3. Real-world video test
- **Asked:** I recorded my own video (the agent's run only had a synthetic stand-in) and ran
  `scripts/video_test.py` on it.
- **Results:**
  - The model did not detect my blow dryer, the mallet stand-in, as a mallet.
  - It struggled with top-down camera angles.
  - It sometimes produced false positives on a standing fan and the blow dryer (and once on a shoe in the
    original-quality file).
- **What I think is causing it:** The training data is mostly outdoor shots of mallets and bottles on the
  ground, so top-down shots and indoor household objects are outside its distribution.
  A blow dryer is also a loose stand-in for a mallet, so a missed detection there is partly expected.
  The false positives suggest it keys on simple shapes and colors rather than the object itself.
- **Takeaway:** Validation scores (~0.87 mAP50) overstate real-world performance. More varied angles and
  hard negatives (household clutter) would be the next data to collect.

## 4. Write-up
- **Asked:** Claude chat to critique my METHODOLOGY draft and explain things I didn't fully understand (mAP50 vs mAP50-95).
- **Kept / changed:** None
- **AI error:** None, used only for explanation
