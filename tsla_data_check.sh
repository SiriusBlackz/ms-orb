#!/usr/bin/env bash
# Read-only: asks the IB Gateway on the droplet whether TSLA quotes arrive as
# real-time or 15-minute delayed on the paper account. Places no orders.
# Works with the market closed. Run on the Mac:  bash tsla_data_check.sh
ssh root@146.190.76.14 'bash -s' <<'EOF'
cd /opt/ms-orb-ibkr || exit 1
./venv/bin/python3 - <<'PY'
import logging, math
from ib_insync import IB, Stock
logging.getLogger("ib_insync").setLevel(logging.CRITICAL)

NAMES = {1: "real-time", 2: "real-time (frozen, market closed)",
         3: "DELAYED 15 min", 4: "DELAYED (frozen, market closed)"}
ib = IB()
codes = []
ib.errorEvent += lambda reqId, code, msg, contract: (
    codes.append(code), print(f"  IB message {code}: {msg[:150]}"))
ib.connect("127.0.0.1", 4002, clientId=98, timeout=20, readonly=True)
print("account:", ", ".join(ib.managedAccounts()))
c = Stock("TSLA", "SMART", "USD")
ib.qualifyContracts(c)
ib.reqMarketDataType(4)      # best available: live if subscribed, else delayed
t = ib.reqMktData(c, "", False, False)
t.marketDataType = 0         # 0 = IB has not told us the type yet
ib.sleep(8)
have = lambda x: x is not None and not (isinstance(x, float) and math.isnan(x))
print(f"data type reported: {t.marketDataType} = {NAMES.get(t.marketDataType, 'not reported')}")
print(f"last={t.last} close={t.close} bid={t.bid} ask={t.ask}")
bad = [k for k in codes if k in (354, 10089, 10090, 10091, 10167, 10168)]
if bad or t.marketDataType in (3, 4):
    print("VERDICT: NOT real-time - the paper account is getting delayed data.")
elif t.marketDataType in (1, 2) and any(have(x) for x in (t.last, t.close, t.bid, t.ask)):
    print("VERDICT: REAL-TIME data is available to the paper account.")
else:
    print("VERDICT: inconclusive - no data type or prices came back.")
ib.disconnect()
PY
EOF
