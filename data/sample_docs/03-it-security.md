# IT Security and Acceptable Use

_Fictional security policy. Contains no real API keys, tokens, or customer data._

Vision AI Ops treats production credentials as secrets. **Never paste live API keys, Slack tokens, or customer prompts into chat, tickets, or this knowledge assistant.**

## Accounts and MFA

- SSO is required for TraceLight Cloud, GitHub, and email.
- Multi-factor authentication (MFA) is mandatory. SMS is not an approved second factor; use a hardware key or an authenticator app.
- Laptops must use full-disk encryption (FileVault or BitLocker) and auto-lock after 5 minutes.

## Acceptable use

- Company GitHub orgs are for Vision AI Ops work only. Do not push private customer datasets.
- Sample documents in this repository are fictional. Do not replace them with real customer transcripts.
- Phishing: if a message asks you to "confirm your Okta password", report it to security@visionaiops.example — that mailbox is fictional as well.

## Incident reporting

Suspected laptop loss, leaked `.env` files, or unexpected TraceLight ingest spikes must be reported within **1 hour** to the on-call security rotation posted in the internal #sec-oncall channel.

## Vendors

Third-party tools need a short security review before they receive customer traces. Shadow AI tools that send prompts to an unknown host are not allowed on corporate machines.
