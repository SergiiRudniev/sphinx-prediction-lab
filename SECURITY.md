# Security

## Reporting

Report vulnerabilities through GitHub private vulnerability reporting rather than a
public issue:

<https://github.com/SergiiRudniev/sphinx-prediction-lab/security/advisories/new>

## Secrets

Never commit:

- wallet private keys or seed phrases;
- API keys, HMAC secrets or passphrases;
- signed orders or reusable authentication payloads;
- private RPC credentials;
- raw user-identifying datasets.

## Execution Boundary

Repository defaults keep live trading disabled. Model outputs are recommendations.
Jurisdiction checks, key custody, exposure limits, stale-data rejection, slippage
limits and emergency exits must be deterministic and independently reviewed.

## Wallet Analysis

Graph clustering is probabilistic. Shared infrastructure, relayers and exchange
funding can create false links. Do not publish identity claims or accusations from
model scores.
