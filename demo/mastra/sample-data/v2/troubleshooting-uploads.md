# Troubleshooting Upload Errors

A `413 Payload Too Large` error means a single file exceeds the 5GB per-object limit; split large files before uploading. A `403 Forbidden` error usually means the API key used is read-only — check the key's scope in Account Settings. A `429 Too Many Requests` error means the account's concurrent-upload limit (20 simultaneous uploads) was exceeded; retry with backoff, or upload in smaller batches.
