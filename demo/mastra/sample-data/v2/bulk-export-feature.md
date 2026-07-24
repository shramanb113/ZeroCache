# Bulk Export

Bulk Export lets you download an entire bucket as a single compressed archive instead of downloading files individually. Start an export from the dashboard or via the API's `POST /buckets/{id}/export` endpoint; large buckets are processed asynchronously and you'll receive a download link by email when the archive is ready. Bulk Export is available on the Pro and Business tiers only, and counts against the tier's monthly egress allowance.
