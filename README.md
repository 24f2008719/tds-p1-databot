# Data-Analyst Telegram Bot

Replies to a Telegram data-analysis question with exactly one JSON object:
`{"answer": <shaped as asked>, "log_url": "https://your-host/run.jsonl"}`

## 1. Create the bot
In Telegram: `@BotFather` → `/newbot` → get a token like `1234567890:AAE...`.
Username must end in `bot`.

## 2. Local test
```bash
export BOT_TOKEN=...
export OPENAI_API_KEY=...
export BASE_URL=http://localhost:8000
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8000
```
Message your bot on Telegram from your own account and confirm you get back a
single clean JSON line. Then check:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/run.jsonl
```

## 3. Deploy on Render
1. Push this folder to a **public** GitHub repo.
2. Render → New → Web Service → connect the repo (or use `render.yaml` / Blueprint).
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
   - Env vars: `BOT_TOKEN`, `OPENAI_API_KEY`, `BASE_URL=https://<service>.onrender.com`
3. After setting env vars, **trigger a manual deploy** (Render doesn't auto-restart on env var changes alone).
4. Verify:
   ```bash
   curl https://<service>.onrender.com/health
   wget https://<service>.onrender.com/run.jsonl
   ```

## 4. Test like the grader
- Send the worked example question from your own Telegram account.
- Send a multi-turn sequence ("I will send data next." then the data+question) — confirm the bot replies to *both*.
- `wget` the `log_url` from a different network to confirm it's truly public.
- Optionally clone `github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot`, point it at your bot, and run your own questions from `evals/questions.json`.

## 5. Register
Submit, comma-separated: `https://github.com/<you>/<repo>, your_bot_username`

## Notes / gotchas
- No secrets in the repo — tokens live only in env vars.
- Free Render instances sleep after ~15 min idle; the built-in `/health`
  self-ping thread (every 10 min) keeps it warm, but an external pinger
  (UptimeRobot) is a good backup.
- Use a **direct** OpenAI API key, not a weekly-expiring proxy token — grading
  happens after the deadline and an expired key = a dead bot.
- `gpt-4o-mini` / `gpt-4.1-mini` get real-world stats questions wrong more
  often; `gpt-4o` (default here) is the tested-working floor.
