# Demo images

Upload these during a live demonstration (🎬 Live demo → Inspect a part, after selecting the product).
They come from the MVTec AD **operating test split** — never used to train a model or choose a threshold —
so the expected outcomes below are honest. Images: CC BY-NC-SA 4.0 (MVTec AD), research and teaching only.

| file | product | ground truth | expected decision | expected score | what to show |
|---|---|---|---|---|---|
| `bottle/good_017.png` | bottle | good | **PASS** | 22.3 | passes cleanly |
| `bottle/good_008.png` | bottle | good | **PASS** | 26.6 | passes cleanly |
| `bottle/defect_broken_large_004.png` | bottle | defective: broken large | **REJECT** | 71.6 | rejected; heat map on the defect |
| `bottle/defect_broken_small_010.png` | bottle | defective: broken small | **REJECT** | 69.1 | rejected; heat map on the defect |
| `bottle/defect_contamination_009.png` | bottle | defective: contamination | **REJECT** | 64.8 | rejected; heat map on the defect |
| `bottle/defect_contamination_borderline_003.png` | bottle | defective: contamination | **CHECK** | 38.1 | borderline → human review |
| `drift_demo/bottle_good_part_lamp_failure.png` | bottle | good (lamp failure simulated) | **CHECK** | high | camera-health gate: no automatic REJECT on an unqualified image |
| `carpet/good_005.png` | carpet | good | **PASS** | 16.1 | passes cleanly |
| `carpet/good_010.png` | carpet | good | **PASS** | 25.3 | passes cleanly |
| `carpet/defect_color_011.png` | carpet | defective: color | **REJECT** | 60.6 | rejected; heat map on the defect |
| `carpet/defect_cut_006.png` | carpet | defective: cut | **REJECT** | 106.8 | rejected; heat map on the defect |
| `carpet/defect_hole_004.png` | carpet | defective: hole | **REJECT** | 83.7 | rejected; heat map on the defect |
| `carpet/defect_metal_contamination_006.png` | carpet | defective: metal contamination | **REJECT** | 68.3 | rejected; heat map on the defect |
| `carpet/defect_thread_010.png` | carpet | defective: thread | **REJECT** | 112.5 | rejected; heat map on the defect |
| `carpet/defect_color_borderline_006.png` | carpet | defective: color | **CHECK** | 38.5 | borderline → human review |
| `drift_demo/carpet_good_part_lamp_failure.png` | carpet | good (lamp failure simulated) | **CHECK** | high | camera-health gate: no automatic REJECT on an unqualified image |
| `screw/good_025.png` | screw | good | **PASS** | 21.2 | passes cleanly |
| `screw/good_020.png` | screw | good | **PASS** | 23.9 | passes cleanly |
| `screw/defect_manipulated_front_010.png` | screw | defective: manipulated front | **REJECT** | 49.3 | rejected; heat map on the defect |
| `screw/defect_scratch_head_009.png` | screw | defective: scratch head | **REJECT** | 48.4 | rejected; heat map on the defect |
| `screw/defect_scratch_neck_003.png` | screw | defective: scratch neck | **REJECT** | 60.2 | rejected; heat map on the defect |
| `screw/defect_thread_side_014.png` | screw | defective: thread side | **REJECT** | 43.7 | rejected; heat map on the defect |
| `screw/defect_thread_top_001.png` | screw | defective: thread top | **REJECT** | 52.7 | rejected; heat map on the defect |
| `screw/defect_manipulated_front_borderline_021.png` | screw | defective: manipulated front | **CHECK** | 36.3 | borderline → human review |
| `drift_demo/screw_good_part_lamp_failure.png` | screw | good (lamp failure simulated) | **CHECK** | high | camera-health gate: no automatic REJECT on an unqualified image |
| `carpet_new_lot_good_parts/ (all files)` | carpet | good parts of the current lot | **—** | — | Drift & new lots → section C: upload them all → Re-qualify |

Scores are on the 0–100 scale (25 ≈ typical good part, 75 ≈ typical synthetic defect). Decisions use the deployed thresholds; carpet uses the model re-qualified for the current lot. Small differences can occur with resized uploads.
