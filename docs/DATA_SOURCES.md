# Data Sources

Sphinx Corpus begins with public Polymarket and Polygon data. This document records
the intended source surface; exact endpoints, contract addresses and backfill
cursors must be frozen in snapshot manifests.

## Polymarket

| Source | Intended use |
| --- | --- |
| [Gamma API](https://gamma-api.polymarket.com) | Markets, events, tokens, categories and resolution metadata |
| [Data API](https://data-api.polymarket.com) | Trades, activity, positions, holders and public profiles |
| [CLOB API](https://clob.polymarket.com) | Prices, spreads, orderbooks and order execution |
| [Market WebSocket](https://docs.polymarket.com/market-data/websocket/market-channel) | Live book, price, trade and market lifecycle events |
| [Public subgraphs](https://docs.polymarket.com/market-data/subgraph) | Indexed fills, matches, positions, PnL and on-chain activity |

## Polygon

Polygon RPC is used to verify transaction ordering, contract events, transfers and
funding relationships. Wallet clustering must retain provenance and uncertainty;
shared funders or relayers do not prove common ownership.

## Collection Rules

- Save raw payloads before normalization.
- Record request parameters, response time and pagination cursor.
- Use bounded time windows instead of relying on offset-only pagination.
- Store event time and publication/observation time separately.
- Version protocol and contract migrations explicitly.
- Never log private keys, API secrets or signed order payloads.
- Respect source terms, rate limits and geographic restrictions.

## Redistribution

The repository does not claim redistribution rights over third-party data. Dataset
releases require a source-by-source license and terms review before publication.
