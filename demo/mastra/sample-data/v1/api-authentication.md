# API Authentication

All Aurora Cloud Storage API requests require an `Authorization: Bearer <api-key>` header. API keys are generated from Account Settings > API Keys and can be scoped to read-only or read-write access. Keys do not expire automatically but can be revoked at any time from the same page. There is no support for API keys scoped to a single bucket in the current version.
