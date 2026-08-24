# Checkpoint 0a — the hair symptom, measured on B1

B1 is the Tears of Steel green-screen plate of the **same actor** as the
`hair` demo shot, so it is the one clip where we hold reference alpha for
this hair. Tier B: keyed reference, not gospel.

## Hair region

| Metric | their_run (1 click, MatAnyone v1) | 1 click + HQ (MatAnyone 2 + zoom) | well-formed clicks + HQ |
|---|---|---|---|
| MAD (x1e3, lower better) | 12.41 | 11.1 | **10.96** |
| MSE (x1e3, lower better) | 5.614 | 5.242 | **5.123** |
| Grad (x1e3, lower better) | 0.2913 | 0.2749 | **0.2694** |
| dtSSD (x1e2, lower better) | 5.673 | 5.506 | **5.46** |
| Boundary F (0-1, higher better) | 0.8825 | 0.9008 | **0.9048** |

## Whole frame

| Metric | their_run (1 click, MatAnyone v1) | 1 click + HQ (MatAnyone 2 + zoom) | well-formed clicks + HQ |
|---|---|---|---|
| MAD (x1e3, lower better) | 2.535 | **2.389** | 2.691 |
| MSE (x1e3, lower better) | 1.133 | **1.061** | 1.321 |
| Grad (x1e3, lower better) | 0.0625 | **0.0582** | 0.0622 |
| dtSSD (x1e2, lower better) | 3.052 | **2.956** | 2.98 |
| Boundary F (0-1, higher better) | **0.9883** | 0.9878 | 0.972 |

## Cost and memory

| Config | s/frame | peak RSS | lowest available | swap |
|---|---|---|---|---|
| their_run (1 click, MatAnyone v1) | 0.202 | 5833 MB | 6029 MB | 9720 → 9720 MB |
| 1 click + HQ (MatAnyone 2 + zoom) | 0.274 | 5833 MB | 7107 MB | 9632 → 9632 MB |
| well-formed clicks + HQ | 1.082 | 5833 MB | 4713 MB | 9322 → 9322 MB |

Peak RSS understates pressure — a process being paged out reports a small resident size — so the guard's available-memory and swap columns are the ones to read.
