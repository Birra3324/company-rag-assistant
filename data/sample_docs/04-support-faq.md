# Customer Support FAQ

_Fictional support FAQ for TraceLight and AlertMesh. No real customer names._

## How do I open a ticket?

Email support@visionaiops.example or use the TraceLight workspace **Help → Contact support**. Include the workspace slug (for example `acme-demo`) and the trace ID. Do not attach production `.env` files.

## What are the P1 support SLAs?

P1 first response time is **30 minutes** (example: TraceLight ingest pipeline down, no new traces). P2 is **4 business hours**. P3 how-to questions are **1 business day**.

| Severity | Example | First response |
| --- | --- | --- |
| P1 | Ingest pipeline down, no new traces | 30 minutes |
| P2 | Evaluators failing for a subset of projects | 4 business hours |
| P3 | How-to questions, UI nits | 1 business day |

## Can we keep data in the EU?

Yes. Choose **EU-West** when the workspace is created. Residency cannot be switched later; open a P3 ticket if you need a second workspace in another region.

## Does TraceLight train on our prompts?

No. TraceLight Cloud does not use customer traces to train foundation models. Optional LLM-as-judge evaluators call the customer's own provider key. Keys stay in the customer's secrets manager — Vision AI Ops never stores raw provider keys in this demo dataset.

## AlertMesh webhook format

AlertMesh POSTs JSON `{ "evaluator": "...", "score": 0.0, "trace_id": "...", "project": "..." }` to the URL you configure. Rotate the webhook secret from the AlertMesh settings page. Sample secrets in docs are placeholders like `whsec_demo_not_real`.
