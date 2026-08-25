# Market data

This module downloads checksum-verified Binance archives and normalises registered
Dukascopy/JForex Bid/Ask exports for USA500, USATECH and VIX. It builds causal M1, M5,
M15 and H1 inputs required by the registered experiments. Features and labels are created
downstream; raw loaders do not use future rows.
