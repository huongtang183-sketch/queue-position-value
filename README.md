# Queue Position Value: Passive Realized Spread at 1-Tick vs Wider Spreads

Does queue position at a one-tick spread earn a premium over wider spreads? Tested on Binance USD-M futures (bookTicker and aggTrades), five pairs (WOOUSDT, RNDRUSDT, MTLUSDT, ASTRUSDT, KEYUSDT), 2023-09-01 to 2023-10-30, 12.6M fills at the best quote.

## Finding

Passive realized spread at one tick lies 0.096 ticks below the line through its two- and three-tick values without controls (p = 0.002) and 0.004 ticks below once volatility, book depth and fill size relative to depth are controlled (p = 0.87). One-tick quotes earn no queue-position premium; the gap without controls disappears with the controls.

| Test | Fills | G (ticks) | p-value | Decision at 5% |
|---|---|---|---|---|
| Q1: no controls | 12.58M | -0.096 | 0.002 | H01 rejected |
| Q2: with controls | 12.56M | -0.004 | 0.87 | H02 not rejected |

Full write-up: [report/Queue_Position_Value.pdf](report/Queue_Position_Value.pdf)

## Repository

| Path | Content |
|---|---|
| `src/screen.py`, `src/find_pairs.py` | Pair selection: share of time at a one-tick spread |
| `src/build_fills.py` | Downloads quote and trade files, builds one row per trade |
| `src/manifest.py` | Manifest of the fill files on disk |
| `src/describe_sample.py` | Fills by pair and spread state |
| `src/fills.py` | Loading, control bins and estimation shared by the scripts |
| `src/run_tests.py` | Tests Q1 and Q2 with the robustness checks |
| `results/` | Output files behind the report |
| `report/` | Final report |

## Reproduce

Python 3.12. Run from the repository root.

```
pip install -r requirements.txt
python src/build_fills.py --start 2023-09-01 --end 2023-10-30
python src/manifest.py --data data
python src/describe_sample.py --data data
python src/run_tests.py --data data
```

`build_fills.py` downloads the daily files from data.binance.vision, writes `data/<PAIR>/<PAIR>-<DAY>.csv.gz` and deletes the raw zips; days already built are skipped. The `data/` folder is not part of the repository. All scripts write their output to `results/`.

Pair selection (optional, already applied in the defaults of `build_fills.py`):

```
python src/screen.py
python src/find_pairs.py --day 2023-09-12 --n 30
```

## Data

Binance data archive, https://data.binance.vision/ (daily USD-M futures bookTicker and aggTrades files).

## References

- Moallemi, C. C., & Yuan, K. (2017). A model for queue position valuation in a limit order book. Columbia Business School working paper.
- Yao, C., & Ye, M. (2018). Why trading speed matters: A tale of queue rationing under price controls. The Review of Financial Studies, 31(6), 2157-2183.
