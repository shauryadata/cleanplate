# ToS shots — self-consistency (no reference alpha exists for these)

The three graded Tears of Steel shots have no ground truth, so these are the
Task 2 stability metrics only. Accuracy numbers live in BENCH.md.

## walk

| Metric | binary (SAM 2) | baseline (MatAnyone v1) | new default (MatAnyone 2) | winner (zoom + MatAnyone 2) |
|---|---|---|---|---|
| Flicker - mean frame-to-frame area change | 1.647 % | 1.489 % | 1.520 % | 1.524 % |
| Flicker - max frame-to-frame area change | 7.207 % | 5.563 % | 5.831 % | 5.892 % |
| Consecutive-frame IoU - mean | 0.928 | 0.928 | 0.928 | 0.928 |
| Consecutive-frame IoU - min | 0.859 | 0.855 | 0.863 | 0.863 |
| Boil - perimeter CV | 9.170 % | 8.785 % | 8.870 % | 8.881 % |
| Boil - shape-normalised perimeter CV | 6.826 % | 7.111 % | 7.191 % | 7.192 % |
| Softness - pixels with 0 < alpha < 255 | 0 % of frame | 0.791 % of frame | 1.088 % of frame | 1.118 % of frame |
| Softness - distinct alpha values | 2 | 256 | 256 | 256 |
| Fragmentation - mean significant components (>=1% of largest) | 1 | 1 | 1 | 1 |
| Fragmentation - max significant components | 1 | 1 | 1 | 1 |
| Threshold noise - mean speck pixels | 0.100 px | 0.490 px | 0.480 px | 0.470 px |

Cost: track 0.8045 s/frame, refine 0.1243 (default) / 0.1876 (winner) s/frame, peak RSS 1775 MB.

## dialogue

| Metric | binary (SAM 2) | baseline (MatAnyone v1) | new default (MatAnyone 2) | winner (zoom + MatAnyone 2) |
|---|---|---|---|---|
| Flicker - mean frame-to-frame area change | 0.565 % | 0.597 % | 0.554 % | 0.813 % |
| Flicker - max frame-to-frame area change | 2.575 % | 3.076 % | 2.553 % | 4.944 % |
| Consecutive-frame IoU - mean | 0.948 | 0.947 | 0.947 | 0.957 |
| Consecutive-frame IoU - min | 0.872 | 0.871 | 0.866 | 0.865 |
| Boil - perimeter CV | 4.063 % | 4.763 % | 5.963 % | 5.168 % |
| Boil - shape-normalised perimeter CV | 6.201 % | 6.703 % | 7.996 % | 6.216 % |
| Softness - pixels with 0 < alpha < 255 | 0 % of frame | 1.062 % of frame | 1.102 % of frame | 1.317 % of frame |
| Softness - distinct alpha values | 2 | 256 | 256 | 256 |
| Fragmentation - mean significant components (>=1% of largest) | 2.740 | 2.583 | 2.823 | 2.833 |
| Fragmentation - max significant components | 4 | 3 | 4 | 4 |
| Threshold noise - mean speck pixels | 0.470 px | 0.720 px | 0.280 px | 4.930 px |

Cost: track 0.7415 s/frame, refine 0.1238 (default) / 0.2128 (winner) s/frame, peak RSS 2270 MB.

## hair

| Metric | binary (SAM 2) | baseline (MatAnyone v1) | new default (MatAnyone 2) | winner (zoom + MatAnyone 2) |
|---|---|---|---|---|
| Flicker - mean frame-to-frame area change | 0.295 % | 0.297 % | 0.304 % | 0.301 % |
| Flicker - max frame-to-frame area change | 2.530 % | 1.721 % | 2.057 % | 2 % |
| Consecutive-frame IoU - mean | 0.956 | 0.955 | 0.956 | 0.956 |
| Consecutive-frame IoU - min | 0.841 | 0.846 | 0.844 | 0.843 |
| Boil - perimeter CV | 2.893 % | 2.896 % | 2.797 % | 2.829 % |
| Boil - shape-normalised perimeter CV | 3.688 % | 3.703 % | 3.856 % | 3.874 % |
| Softness - pixels with 0 < alpha < 255 | 0 % of frame | 0.881 % of frame | 0.945 % of frame | 0.990 % of frame |
| Softness - distinct alpha values | 2 | 256 | 256 | 256 |
| Fragmentation - mean significant components (>=1% of largest) | 1 | 1 | 1 | 1 |
| Fragmentation - max significant components | 1 | 1 | 1 | 1 |
| Threshold noise - mean speck pixels | 0 px | 0.500 px | 0 px | 0 px |

Cost: track 0.7605 s/frame, refine 0.1282 (default) / 0.2531 (winner) s/frame, peak RSS 2274 MB.
