# Data card — MVTec AD subset

* **Source:** MVTec Anomaly Detection dataset — https://www.mvtec.com/research-teaching/datasets/mvtec-ad
* **Download used here:** Hugging Face mirror `foersben/mvtec-ad`, pinned to commit
  `c75b39616f84db43677bcc8228caaafaf5096d7f` (the commit that completed the upload), via `scripts/download_data.py`.
* **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). Research and teaching only;
  commercial use is not allowed. Derived example images in `examples/` are shared under the same licence.
* **Citation:** Bergmann, Batzner, Fauser, Sattlegger, Steger. *The MVTec Anomaly Detection Dataset: A Comprehensive
  Real-World Dataset for Unsupervised Anomaly Detection.* IJCV 129(4):1038–1059, 2021; and CVPR 2019.

| Product | Role | Train (good) | Test good | Test defective | Defect types |
|---|---|---|---|---|---|
| bottle | main | 209 | 20 | 63 | broken large, broken small, contamination |
| screw | main | 320 | 41 | 119 | manipulated front, scratch head, scratch neck, thread side, thread top |
| carpet | main | 280 | 28 | 89 | color, cut, hole, metal contamination, thread |
| metal_nut | few-shot new product | 220 | 22 | 93 | bent, color, flip, scratch |

**Splits (seed 42):** train/good → 80% `model_train_good` + 20% `calibration_good`; test → per defect type 35%
`oracle_calibration` (real-label threshold comparison only) + 65% `operating_test`. See `reports/splits.csv`.

**Known properties (EDA):** controlled lighting (narrow brightness spread per product); bottle 900×900 RGB, screw and
carpet 1024×1024; screw images are near-grayscale with random part orientation; most defects are small (median mask
area well below 2% of the image); test sets are defect-heavy (~70% defective) unlike a real line (~1%), so rates are
reported per class and costs are re-weighted to a 1% defect rate.
