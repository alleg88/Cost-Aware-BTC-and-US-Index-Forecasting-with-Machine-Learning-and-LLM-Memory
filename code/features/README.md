# Features

Feature builders transform market inputs into causal price, order-flow, positioning,
sentiment, index and channel features. The target is a three-class down/flat/up label with
a registered dead zone and forecast horizon. Event/news joins are as-of: an observation is
eligible only when its timestamp precedes the corresponding decision time.
