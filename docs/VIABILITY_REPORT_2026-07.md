# Is the ORB edge worth real money — and whose money?

**Decision report · ms-orb · 5 July 2026**
Own capital vs. prop-firm funding vs. buying richer data. Built on the 19,260-signal
research dataset (Jun 2025–Jun 2026), 30k+ Monte Carlo runs, and primary-source research
on 2026 prop-firm terms, algo policies, and market-data costs.
Rendered version: `docs/VIABILITY_REPORT_2026-07.html` ·
[published artifact](https://claude.ai/code/artifact/3ad9c976-6f8f-4de1-af38-0a124b201164)

---

## Verdict

**The strategy is worth funding — through a static-drawdown CFD prop firm, not the OANDA
account, and only after a 2–3 month forward-test gate.**

The Tokyo-session portfolio (XAU + NAS100, 1.5×ATR stop, 3:1, first-breakout, HTF-aligned)
has a statistically real but modest edge: +0.31R/trade on XAU (95% CI [+0.04, +0.59]) and
+0.22R on NAS100, with near-zero monthly correlation between the two. Conditional on that
edge holding live, an FTMO-style $100k evaluation passes ~83–94% of the time at 0.5–0.75%
risk with **EV ≈ +$41–58k per attempt** against a ~$540 fee (≈ $33–46k after a counterparty
haircut), funded in ~2–3.5 months. Nothing else comes close: your own account earns the same
edge on your own (smaller) capital at the most expensive venue tested, and the
order-book-data path is a documented dead end for 5-minute alpha. The one honest blocker:
this is a 12-month backtest that had **never traded forward** — the deployed demo bots ran a
different config (5:1 bracket). Close that gap first; it costs nothing but time.
*(Status: forward test running as `ms-orb-configa` since 2026-07-05.)*

## 01 · The edge, stress-tested

Both researched configs were reproduced independently from `research_signals.csv` before
trusting anything (Config A: 35.0% WR / PF 1.62 gross vs. the documented 34.8% / 1.60;
Config B: exact match at n=679, PF 1.22 net, +92R).

| | Config A — XAU Tokyo 3:1 bracket | Config B — runner (Tokyo+NY) |
|---|---|---|
| Rules | seq 0, HTF-aligned, 1.5×ATR stop | XAU+NAS, HTF, rvol≥1, 1×ATR stop, 2×ATR trail |
| Net expectancy / trade | **+0.313R** | +0.136R |
| 95% CI on expectancy | [+0.04, +0.59] | [−0.01, +0.29] |
| P(no edge at all) | 1.3% | 3.1% |
| PF net (95% CI) | 1.45 [1.04–1.94] | 1.22 [1.00–1.49] |
| Trades / year | 181 (1/day max) | 679 (~3/day, max 10) |
| Worst single day | −1.3R | **−7.0R** |
| Worst 6 months | −12R (losing) | −17R (losing) |

**Config A's rules are the horse — but run them on both instruments.** Applying the
identical rules per (instrument × session) shows the edge lives in Tokyo, and only in
Tokyo, for *both* instruments; London and NY are net-negative everywhere and stay excluded:

| Instrument · session | n/yr | WR | PF net | expR net | P(no edge) |
|---|---|---|---|---|---|
| XAU · Tokyo | 181 | 35.0% | 1.45 | **+0.313** | 1.2% |
| NAS100 · Tokyo | 171 | 32.5% | 1.31 | **+0.216** | 6.0% |
| XAU · London / NY | 378 | ~22% | 0.73–0.83 | −0.21 / −0.12 | >85% |
| NAS100 · London / NY | 422 | ~23% | 0.78–0.88 | −0.18 / −0.08 | >78% |

XAU and NAS100 Tokyo monthly returns are essentially uncorrelated (−0.10), so the
two-instrument portfolio nearly doubles yearly expectancy (+93R vs +57R) while the worst
combined day stays at −2.4R — comfortably inside a 5% daily-loss rule at recommended
sizing. Config B's −7R days breach that limit at any sizing above 0.7%, and its CI touches
zero; it stays off propped accounts. Everything here is regime-dependent — 9/13 months
positive with genuinely negative half-years. Whatever funds this must survive a six-month
flat stretch without abandoning it.

## 02 · Path 1 — your own account

Day-block bootstrap, compounding at fixed fractional risk, Config A:

| Risk/trade | Median year | 5th pct | 95th pct | P(losing yr) | p95 drawdown |
|---|---|---|---|---|---|
| 0.5% | +32% | +7% | +64% | 1.6% | 12% |
| 1.0% | +70% | +13% | +163% | 1.8% | 23% |
| 2.0% | +172% | +20% | +528% | 2.4% | 41% |

Excellent *percentages* — the problem is the base. On a £5–10k account, a +70% median year
is £3.5–7k, earned at the **most expensive venue tested**: OANDA's gold spread (~$0.38)
plus overnight financing (≈1% admin fee on metals, and Tokyo holds are always overnight)
is 0.86bp/round-trip vs 0.18–0.35bp on CME futures. Your own account is the right place to
*prove* the edge, not to monetize it.

## 03 · Path 2 — prop funding ✅ RECOMMENDED

Simulated with exact 2026 terms, the empirical trade distribution (time-exits, clustering,
costs included), 5,000 runs per cell. Two-year funded horizon, 80% split, fee refund on
first payout.

### Tokyo portfolio (XAU + NAS100) on a $100k CFD evaluation

| Policy | P(pass) | Months to funded | E[payout \| funded, 2yr] | EV / attempt | EV stressed¹ |
|---|---|---|---|---|---|
| 0.5% throughout (EV-max, recommended) | 94% | 3.4 | $62,100 | **+$57,900** | +$41,300 |
| 0.75% eval → 0.5% funded (faster) | 83% | 2.0 | $62,100 | +$50,800 | +$34,500 |
| 1% eval → 0.5% funded (fastest) | 73% | 1.3 | $61,800 | +$44,800 | +$30,400 |

¹ Stressed = +0.05R/trade extra cost (≈60% higher than modeled). FTMO's actual gold pricing
(~$0.26/oz all-in) is *cheaper* than the $0.35 in the backtest costs, so the base case is,
if anything, conservative on costs. Haircut all EVs a further ~10–20% for counterparty risk.

XAU-only (the single-instrument fallback if NAS100 fails its forward test) still clears
+$38–42k EV at 0.5–0.75% risk but takes 3.4–5.7 months to fund — the NAS100 leg buys speed
and ~40% more EV for the same drawdown behavior.

The asymmetry is the whole argument: downside is a ~$540 fee, upside is trading someone
else's $100k with a strategy that essentially cannot breach a 10% *static* drawdown at
0.5% sizing. The industry's dismal base rate (~7% of participants ever get paid — FPFX
data, 300k accounts) coexists with this math because most buyers have no measured edge.

### Firm shortlist — policy is the real filter

| Firm | Fit | Why |
|---|---|---|
| **FTMO 2-Step Swing** $100k · ~$540 | ✅ best | 10% static DD, 5% daily; own-built bots explicitly allowed; Swing = no news/weekend rules (Tokyo overnights fine); best payout record in CFD prop ($450M+ paid). Port: MT5 Python bridge. Also runs **OANDA Prop Trader** — same rulebook. |
| **FundingPips 2-Step** $100k · $399 | ✅ strong | Same static structure, cheapest fee, proof-of-ownership algo policy (source + git history). Port: **cTrader Open API** — cleanest Linux path, no Windows VM. |
| **FundedNext Stellar 2-Step** · $550 | ✅ good | 8%/5% targets (easiest), static DD, news = 40% profit haircut not breach. Verify the XAU 1:10 leverage change and EA add-on fee at checkout. |
| **MyFundedFutures Pro / Topstep** (futures) | ⚠ backup | Cheaper fills, real APIs (Tradovate / ProjectX), automation legal (MFFU end-to-end; Topstep until Live). But 3% EOD-trailing DD forces ≤0.4% risk → EV ~$23–31k. Topstep bans VPS; MFFU requires "active supervision" — get it defined in writing for a Tokyo bot. |
| **Apex · Alpha Capital · Earn2Trade · E8** | ❌ avoid | Apex: automation banned + documented hostility to winners (May 2025 mass bans). Alpha: autonomous EAs "prohibited under any circumstances." Earn2Trade: bans EAs. E8: intraday-trailing DD is structurally wrong for 35% WR / 3:1. |

**Sizing rule that falls out of the math:** with the two-instrument portfolio, 0.5%
throughout is both EV-max and fast enough (~3.4 months); go to 0.75% during eval only if
speed matters more than ~$7k of EV. Never take Config B onto a propped account at more
than 0.25%. EV is flat across 0.5–1.0%, so smaller size buys survival, not returns.
*(Post-report note: the realized worst DD of the portfolio is 19.7R — 9.8% at 0.5% risk,
kissing FTMO's 10% static limit; consider 0.4% funded sizing.)*

## 04 · Path 3 — buy order-book data ❌ NOT FOR ALPHA

The hypothesis ("beating PF 1.6 needs richer data") meets a hostile evidence base:
order-flow predictability is concentrated at millisecond-to-seconds horizons, is dead by
~3 minutes (Cont/Cucuringu/Zhang 2023: *negative* OOS R² at 1-minute; Aït-Sahalia et al.:
predictability gone by 3 min, 10ms of latency destroys 80% of it), and **no disinterested
study shows flow-confirmation improving breakout strategies** — the one transparent NQ-ORB
study found high-volume breakouts had *worse* follow-through. The in-sample CVD+ramp filter
result (PF 1.07→1.37) is worth a cheap falsification, not a subscription.

- **Rational spend:** ~$150–400 one-off — Databento historical MBP-10/MBO for NQ+GC
  (starts against their $125 free credit). Pre-register the test: does delta-confirmation
  lift Config A out-of-sample? Accept the likely null.
- **Skip entirely:** live depth feeds ($150–1,750/mo). No 5-minute strategy justifies them.
- **The real payoff of depth data is measurement, not alpha** — queue/fill simulation to
  tighten slippage estimates. Useful after funding, not before.
- The literature's better-supported levers at this horizon are cross-asset/regime filters —
  which is what the HTF-trend and RVOL filters already are.

## 05 · The plan

1. **Close the backtest-authority gap** (done 2026-07-05, commit `f190016`): Config A now in
   the authoritative `backtest.py` (`--config-a` preset).
2. **Forward-test the Tokyo portfolio on the demo droplet** (running since 2026-07-05 as
   `ms-orb-configa`, own practice sub-account). Pull `trades.db` monthly and measure realized
   slippage + Tokyo spread vs the cost model. **Gate: forward PF net > ~1.2 and fills within
   ~0.05R of model — judged per instrument; if NAS100 fails alone, fall back to XAU-only.**
3. **Port the bot once, broker-agnostically** (parallel): no prop firm speaks OANDA v20 —
   write a cTrader Open API client (FundingPips; Linux-native) or an MT5 bridge (FTMO).
   Test on the firm's free trial before buying anything.
4. **On gate pass (~$400–540): buy one eval** — FTMO Swing or FundingPips, $100k, 0.5% risk,
   both Tokyo instruments. Expected: funded in ~3.4 months, ~94% pass. Keep git history +
   a strategy note ready for proof-of-ownership review. Withdraw on every payout window —
   banked payouts are the hedge against firm risk.
5. **Scale by adding accounts, not risk.** A second eval at a different firm diversifies
   counterparty risk. Optionally run the $150–400 Databento falsification then.

## 06 · What would change this answer

- **Forward test fails the gate** (PF net < 1.2 or slippage > 0.05R worse than modeled) →
  stop; the whole EV chain is conditional on the edge being real. Don't rationalize a miss.
- **The edge was 1-year-regime luck** — 2026 was a strong gold-trend year. Cheap check:
  re-run `backtest_research.py` on the older Feb 2025–Feb 2026 cache files in `cache/`.
- **Firm rule drift** — CFD prop terms change quarterly (~100 firms died 2024–25). Re-verify
  drawdown type, algo policy, and FTMO Swing news rules the week of purchase, not from this
  report.
- **Regulation** — ESMA says retail prop isn't a priority (July 2026), but a UK/EU crackdown
  would compress the whole option. Another reason to withdraw aggressively.

## 07 · Methodology & honesty notes

- All trade outcomes are the empirical per-trade R values (costs included per instrument),
  resampled by trading day; monthly-block bootstrap changed pass rates by <2pts for Config A
  (95.2% vs 96.9%), so serial correlation isn't hiding much — *within this one year*.
- The deep risk is the sample, not the sampling: ~350 trades, one year, one market regime,
  zero live fills at analysis time. Plan steps 1–2 convert that assumption into a measurement.
- The NAS100 Tokyo leg is statistically softer than XAU (P(no edge) 6.0% vs 1.2%) and was
  measured under rules originally selected on pooled data — not fully independent evidence.
  It earns its place through near-zero correlation and the same rule set, but it's the leg
  more likely to fail the forward gate; the plan degrades gracefully to XAU-only. FTMO's
  US100 spread (~1.5–2.5pt) also runs wider than the 1.5pt modeled — the stressed column is
  the honest row for the NAS leg.
- Funded-lifetime payouts assume the firm honors a 2-year horizon; haircut EVs 10–20% for
  denial/collapse risk. Forfeiture-at-denial is the real mechanism, so bot rule-compliance
  must be airtight: visible SLs, <200 trades/day, no VPS where banned.
- Prop pricing/rules verified against primary firm pages where fetchable (2026-07-05);
  flagged items (FTMO exact fee, FundedNext XAU leverage, E8 everything, Apex 4.0 pricing)
  came from secondary sources — re-verify at checkout.
- Independent cross-check: an idealized-parameters simulation (35% WR, clean 3:1, no
  time-exits) gave 87.9% pass at 1% risk vs the empirical-distribution 78.7% — same
  conclusion, honest spread.

---

*Sources: `oanda_bot/cache/research_signals.csv` (19,260 signals, 2025-06-16 → 2026-06-10) ·
ms-orb research memories (wr3r, ml_classifier, strategy_research) · primary-source firm
research 2026-07-05 (FTMO, FundedNext, The5ers, FundingPips, E8, Alpha, Topstep, Apex, MFFU,
Tradeify, Earn2Trade) · Databento/CME/AMP/Sierra pricing · Cont & Kukanov & Stoikov 2014,
Cont & Cucuringu & Zhang 2023, Aït-Sahalia et al. 2025, DeepLOB, Briola et al. 2024.*
