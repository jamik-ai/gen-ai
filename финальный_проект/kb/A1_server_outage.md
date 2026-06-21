# Category A: server and test machine outages

A critical outage of a production server (host unreachable, connect status = failed) is restored by the SRE
team under an SLA of **4 hours** from ticket registration. Root-cause diagnostics (logs, metrics) take up to
**1 business day**.

Rebooting any production host requires approval from the on-call engineer — rebooting it yourself without
approval is against policy.

Escalation queue: **INFRA-L2**. Contact: Infrastructure Helpdesk.
