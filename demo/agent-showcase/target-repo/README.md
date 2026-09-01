# linkstash

A tiny zero-dependency URL bookmarking API (`node:http` + `node:test`).

```sh
npm start      # serves on :3000 (PORT overrides)
npm test       # node --test
```

Routes: `GET /links`, `POST /links {"url": "..."}`, `DELETE /links/:id`.
