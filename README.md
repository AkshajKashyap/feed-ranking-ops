
FeedRank Ops

A two-stage personalized news-feed recommendation system covering temporal data preparation, candidate retrieval, learning-to-rank, offline evaluation, low-latency serving, feedback simulation, and monitoring.

Planned Architecture

behavior logs + articles
-> temporal data pipeline
-> offline and online features
-> candidate retrieval
-> second-stage ranking
-> diversification
-> recommendation API
-> impression and click logging

Initial Scope
MIND-small dataset
Temporal train, validation, and test splits
Popularity and content baselines
Candidate retrieval
Learning-to-rank
Ranking metrics and failure analysis
FastAPI serving
Latency and feature-freshness monitoring
