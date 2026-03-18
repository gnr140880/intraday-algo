# 🔐 Credentials Rotation Guide – URGENT

**Status**: Your secrets were exposed in `.env` and `.env.example` files and pushed to this repository. **IMMEDIATE ACTION REQUIRED**.

---

## 1. Zerodha Kite API Credentials

Your old credentials are **COMPROMISED**. Rotate immediately:

### Steps:
1. **Revoke old API key**:
   - Go to https://kite.zerodha.com/settings/api
   - Click your old API key → "Delete"
   - Wait 5 seconds for revocation to complete

2. **Generate new API key**:
   - https://kite.zerodha.com/settings/api → "Create API Key"
   - Copy the new `API Key` and `API Secret`
   - Save in a password manager (not in code/git)

3. **Get new access token**:
   - Use the login flow in AlgoTest: `/api/auth/login-url`
   - Complete Kite login → callback captures `access_token`
   - It will be auto-saved to `.env` (but still rotate periodically)

4. **Update `.env`**:
   ```dotenv
   KITE_API_KEY=<new-api-key>
   KITE_API_SECRET=<new-api-secret>
   KITE_ACCESS_TOKEN=<new-access-token>
   ```

5. **Test**:
   - Run backend: `python -m uvicorn main:app --reload`
   - Hit `/api/health` → should show `"kite_connected": true`
   - If `false`, check credentials

---

## 2. Telegram Bot Token

Your bot token is **COMPROMISED**. Rotate:

### Steps:
1. **Revoke old bot**:
   - Message @BotFather on Telegram
   - `/revoke` → select your bot → confirm
   - Bot will stop responding

2. **Create new bot**:
   - Message @BotFather: `/newbot`
   - Follow prompts (name, username)
   - Copy new token (e.g., `8541238647:AAEFqqwsjqHbuEfUyXMRQcIZHXuBJ7ywWpE`)

3. **Get new chat ID** (if you don't know it):
   - Send any message to your new bot
   - Curl: `curl https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Find your message → note the `chat.id`

4. **Update `.env`**:
   ```dotenv
   TELEGRAM_BOT_TOKEN=<new-token>
   TELEGRAM_CHAT_ID=<your-chat-id>
   ```

5. **Test**:
   - `/api/telegram/test` → bot should send you a message

---

## 3. Application API Key

Your app's `API_KEY` (for protecting `/api/engine/*` and `/api/orders/*` endpoints) is already set in `.env`:

```dotenv
API_KEY=eJmH7kL9pQ2xR5vT8wN3bF6dG4jY1cZ0sM9aW2eK5nF3hL7p
```

### Generate your own strong key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output → update `.env`:
```dotenv
API_KEY=<your-new-long-random-string>
```

When making protected API calls, include header:
```bash
curl -H "X-API-Key: <your-api-key>" https://localhost:8000/api/engine/run-cycle
```

---

## 4. Git History Cleanup

The exposed secrets are still in git history. **Optional but recommended**:

```bash
# Install bfg-repo-cleaner or use git-filter-branch
# Remove .env from all commits (keep current clean version)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch backend/.env' \
  --prune-empty --tag-name-filter cat -- --all

# Force push (warning: rewrites history)
git push origin --force --all
git push origin --force --tags
```

Or simply:
- Delete the repo and re-push without `.env` history.

---

## 5. Security Best Practices Going Forward

1. ✅ **Never commit secrets** – `.env` is in `.gitignore`
2. ✅ **Use `.env.example`** – Share template with real values filled by user
3. ✅ **Rotate credentials periodically** – At least monthly for live trading
4. ✅ **Use a secrets manager** – For production, use AWS Secrets Manager, Azure Key Vault, etc.
5. ✅ **Audit API access** – Zerodha logs all API calls; review monthly
6. ✅ **Limit API key scope** – If Zerodha allows, restrict to specific IPs/endpoints

---

## 6. Verification Checklist

After rotation, verify all credentials work:

```bash
# Test Zerodha connection
curl -X GET http://localhost:8000/api/auth/status

# Test Telegram
curl -X POST http://localhost:8000/api/telegram/test \
  -H "X-API-Key: <your-api-key>"

# Test engine (should work if Kite is connected)
curl -X POST http://localhost:8000/api/engine/run-cycle \
  -H "X-API-Key: <your-api-key>"
```

✅ All 3 should respond with `"success": true` or similar.

---

**Questions?** Check backend logs:
```bash
tail -f backend.log  # or watch console output
```

