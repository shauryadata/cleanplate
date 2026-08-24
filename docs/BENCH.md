# RotoBench — Task 4 baseline and hair experiments

Scored against reference alpha. Tier A clips have alpha that is exact by
construction; tier B is a keyed reference and carries the keyer's own
errors. Metric definitions and their validation are in
`cleanplate/accuracy.py` and `tests/test_accuracy.py`.

## Whole frame, mean over all clips

| Metric | baseline_960 | binary_960 | fullres_1920 | guided_960 | hairzoom2_960 | hairzoom_960 | matanyone2_960 | vitmatte_960 |
|---|---|---|---|---|---|---|---|---|
| MAD (x1e3, lower better) | 4.563 | 8.661 | 80.56 | 6.62 | **4.055** | 4.361 | 4.234 | 4.81 |
| MSE (x1e3, lower better) | 3.301 | 7.665 | 79.55 | 3.536 | **3.119** | 3.274 | 3.172 | 3.309 |
| Grad (x1e3, lower better) | 0.0541 | 0.1392 | 0.2461 | 0.0731 | **0.0449** | 0.0509 | 0.0486 | 0.0523 |
| dtSSD (x1e2, lower better) | 2.109 | 3.481 | 3.112 | 2.328 | **1.883** | 2.02 | 1.978 | 2.317 |
| Boundary F (0-1, higher better) | 0.9592 | 0.9275 | 0.4855 | 0.9586 | 0.9624 | 0.9591 | **0.9625** | 0.9595 |

## Hair region only, mean over all clips

The head bounding box, derived from the reference so it is identical for
every method. This is where the argument is: a whole-frame average is
dominated by the easy interior.

| Metric | baseline_960 | binary_960 | fullres_1920 | guided_960 | hairzoom2_960 | hairzoom_960 | matanyone2_960 | vitmatte_960 |
|---|---|---|---|---|---|---|---|---|
| MAD (x1e3, lower better) | 10.14 | 15.46 | 210.7 | 21.36 | **6.33** | 7.516 | 8.665 | 11.32 |
| MSE (x1e3, lower better) | 4.248 | 10.66 | 205.8 | 6.385 | **3.307** | 4.024 | 3.979 | 3.845 |
| Grad (x1e3, lower better) | 0.2184 | 0.5711 | 1.128 | 0.3573 | **0.1456** | 0.1821 | 0.1935 | 0.1714 |
| dtSSD (x1e2, lower better) | 3.672 | 5.948 | 5.891 | 4.223 | **2.947** | 3.19 | 3.495 | 4.318 |
| Boundary F (0-1, higher better) | 0.9559 | 0.9205 | 0.4182 | 0.957 | 0.9655 | 0.9468 | 0.9581 | **0.9702** |

## Cost

| Method | s/frame | peak RSS | clips |
|---|---|---|---|
| baseline_960 | 0.190 | 5728 MB | 6 |
| binary_960 | 0.131 | 5728 MB | 6 |
| fullres_1920 | 10.189 | 5728 MB | 6 |
| guided_960 | 0.222 | 5728 MB | 6 |
| hairzoom2_960 | 0.557 | 4237 MB | 6 |
| hairzoom_960 | 0.685 | 5728 MB | 6 |
| matanyone2_960 | 0.201 | 5728 MB | 6 |
| vitmatte_960 | 0.286 | 5728 MB | 6 |

## Per clip — hair-region MAD (lower better)

| Clip | tier | baseline_960 | binary_960 | fullres_1920 | guided_960 | hairzoom2_960 | hairzoom_960 | matanyone2_960 | vitmatte_960 |
|---|---|---|---|---|---|---|---|---|---|
| A1_man_wave_mars | A | 5.837 | 6.699 | **2.539** | 12.66 | 2.57 | 2.879 | 5.162 | 16.98 |
| A2_pair_city | A | 20.64 | 37.58 | 15.93 | 37.7 | **15.81** | 17.08 | 17.86 | 17.54 |
| A3_longhair_canal | A | 5.726 | 11.71 | 276.8 | 14.57 | 3.217 | **2.991** | 3.878 | 4.344 |
| A4_longhair_ruins | A | 7.702 | 15.24 | 392.4 | 28.6 | **2.534** | 6.278 | 6.085 | 4.114 |
| A5_coat_harbour | A | 8.5 | 9.188 | 146 | 17 | **3.882** | 4.055 | 6.64 | 11.28 |
| B1_tos_greenscreen_hair | B | 12.41 | 12.36 | 430.6 | 17.61 | **9.971** | 11.81 | 12.36 | 13.68 |

## Per clip — whole-frame MAD (lower better)

| Clip | tier | baseline_960 | binary_960 | fullres_1920 | guided_960 | hairzoom2_960 | hairzoom_960 | matanyone2_960 | vitmatte_960 |
|---|---|---|---|---|---|---|---|---|---|
| A1_man_wave_mars | A | 2.111 | 2.136 | **0.7948** | 4.441 | 1.455 | 1.707 | 1.81 | 3.635 |
| A2_pair_city | A | 3.294 | 24.61 | 2.883 | 5.818 | **2.438** | 3.109 | 2.547 | 3.137 |
| A3_longhair_canal | A | 1.782 | 3.023 | 149.4 | 3.386 | **1.492** | 1.589 | 1.539 | 1.688 |
| A4_longhair_ruins | A | 16.02 | 17.06 | 148.6 | 18.2 | **15.48** | 15.96 | 15.63 | 15.89 |
| A5_coat_harbour | A | 1.64 | 2.244 | 24.79 | 3.11 | **1.308** | 1.32 | 1.506 | 1.854 |
| B1_tos_greenscreen_hair | B | 2.535 | 2.894 | 157 | 4.76 | **2.162** | 2.479 | 2.375 | 2.652 |
