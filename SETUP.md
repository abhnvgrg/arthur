# Setting Diana up

Everything in this document is a step **you** have to take — accounts, keys,
apps to install, permissions to grant. The code is already written and tested.

Work top to bottom. Each section says what breaks if you skip it, so you can
stop wherever you have had enough for one evening.

---

## 0. Dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

The optional set is what gives her ears, a voice, desktop notifications and
screenshots. Without it everything else still runs.

`sounddevice` is not currently installed in this environment — voice will not
start until it is.

---

## 1. The model key — *nothing works without this*

You already have this working, but it belongs in `.env` now.

1. Go to <https://console.groq.com/keys>, create a key.
2. Copy `.env.example` to `.env` and paste it into `ARTHUR_LLM_API_KEY`.

One key covers the chat model, the ears (Whisper) and the voice (Orpheus).

---

## 2. Always-on voice

```powershell
python -m arthur talk
```

She listens continuously. Say **"Diana"** and then what you want. For 25
seconds afterwards you can keep talking without naming her again.

**What you may want to change:**

- `ARTHUR_WAKE_WORD` — her name. Pick something two syllables or more; short
  words trigger constantly on ordinary speech.
- `ARTHUR_REQUIRE_WAKE=0` — she answers everything she hears. Only sane in a
  room by yourself.
- `ARTHUR_BARGE_IN=1` — lets you cut her off mid-sentence. **Use headphones.**
  On speakers she hears her own voice and stops herself. This is a real
  limitation, not a bug: cancelling that properly needs acoustic echo
  cancellation, which is not built.

**Approvals by voice — all of them.** Writing actions ("add a task",
"archive that mail") she asks aloud and you answer "yes" or "no".

Irreversible actions (deleting, sending mail) are also spoken, but they need a
specific word. She reads the action back, says it cannot be undone, and waits
for you to say **"confirm"**. A plain "yes" will not do it — she will tell you
so and ask again. "No" stops it immediately, and silence always means no.

Change the word with `ARTHUR_CONFIRM_WORD`. Pick something you would not say
by accident in ordinary conversation. If you ever want typing back for
irreversible actions only, set `ARTHUR_TYPED_IRREVERSIBLE=1`.

---

## 3. The hub — one Diana, every device

Right now the terminal, the phone and the mail watcher would each be a
separate brain. Point them all at one process instead.

```powershell
python -m arthur serve
```

The first run prints an **owner token, once**. Save it in your password
manager immediately — it is not recoverable.

### 3a. Reach it from your phone — Tailscale

Do **not** open port 8765 to the internet.

1. Install Tailscale on the laptop and the phone: <https://tailscale.com/download>
2. Sign in with the same account on both. It is free for personal use.
3. On the laptop run `tailscale ip -4` — or just use the machine name.
4. On the phone open `http://<laptop-name>:8765`.

Cloudflare Tunnel works too if you later want a real domain.

### 3b. Give the phone its own token

In the web UI, or with the owner token:

```powershell
curl -X POST http://127.0.0.1:8765/api/tokens `
  -H "Authorization: Bearer YOUR_OWNER_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"name\": \"phone\", \"scopes\": [\"read\", \"chat\"]}'
```

Deliberately **no `approve` scope**. A lost phone should not be able to
authorise sending mail. Never give a phone `admin`.

### 3c. Install it as an app

On the phone, open the page → browser menu → **Add to Home Screen**. It
installs as "Diana", runs full screen, and works offline for the shell.

The **Talk** button next to Send records, transcribes and speaks the reply.
Tick **speak replies** to have her talk back. Your phone will ask for
microphone permission the first time — say yes, and note that browsers only
allow the microphone over HTTPS or on localhost. Over plain HTTP on a
Tailscale name, Chrome may refuse; if it does, use Tailscale's HTTPS
certificates (`tailscale cert`) or Cloudflare Tunnel.

### 3d. Start it automatically

Task Scheduler → Create Task:

- Trigger: **At log on**
- Action: `pythonw.exe` with arguments `-m arthur serve`
- Start in: `D:\Code\Projects\arthur`

`pythonw` rather than `python` keeps it from opening a console window.

---

## 4. Push to your phone

1. Install **ntfy** from the Play Store / App Store.
2. Invent a long random topic — `diana-a7f3k9x2qm-inbox`, not `diana`.
   Anyone who guesses the topic reads your notifications.
3. Put it in `ARTHUR_NTFY_TOPIC`, and subscribe to the same topic in the app.

### Approving from your phone

For the Approve/Deny buttons to appear on notifications you need two more
values:

- `ARTHUR_PUBLIC_URL` — where your phone reaches the hub, e.g.
  `http://your-laptop:8765` (the Tailscale name).
- `ARTHUR_APPROVE_TOKEN` — a token minted with the `approve` scope, as in 3b
  but with `"scopes": ["approve"]`.

Without both, notifications still arrive; they just have no buttons.

---

## 5. Mail

This is the biggest automation win and the fiddliest setup.

### Gmail

1. Turn on 2-Step Verification: <https://myaccount.google.com/security>
2. Create an App Password: <https://myaccount.google.com/apppasswords>
   — pick "Mail", name it "Diana". You get 16 characters.
3. Put that in `ARTHUR_IMAP_PASSWORD`. **Your normal Google password will not
   work**, and neither will this if 2-Step is off.
4. Check IMAP is enabled: Gmail → Settings → Forwarding and POP/IMAP.

Other providers: use their IMAP host and port, and an app-specific password
if they offer one.

### What she can do with it

| Tool | Risk | Behaviour |
|---|---|---|
| `search_mail`, `read_mail` | read-only | runs without asking |
| `mark_mail_read`, `archive_mail`, `draft_reply` | writes | asks first |
| `send_mail` | irreversible | you must say the confirm word aloud |

`draft_reply` writes into your Drafts folder and sends nothing. Prefer it.

---

## 6. Automation — jobs and triggers

```powershell
python -m arthur watch
```

That one process runs task reminders, scheduled jobs, and mail triage
together. Leave it running (or add it to Task Scheduler like 3d).

You do not write jobs by hand. Ask her:

> "Diana, every morning at eight, tell me what is due today and anything
> unread from a real person."

She calls `schedule_job` herself. Check what she has set up with:

```powershell
python -m arthur jobs
```

Schedules she understands: `daily at 08:00`, `every 30 minutes`, an ISO
datetime for a one-off, or `on mail` / `on webhook` / `on file` for triggers.

**Jobs never approve their own risky calls.** A scheduled job that wants to
send mail will fail rather than send it. This is on purpose: nobody is
watching at 8am. If you want a job to act unattended you have to widen that
deliberately, and you should think hard first.

The `on webhook` trigger is fired by `POST /api/triggers/webhook`, which is
how you wire in Tasker, IFTTT, or anything else.

### Android automation

Install **Tasker** or **MacroDroid**. Point an HTTP Request action at
`POST http://<laptop>:8765/api/chat` with header
`Authorization: Bearer <phone token>` and body
`{"message": "I just got home"}`. Trigger it on location, SMS, battery,
whatever you like.

---

## 7. What she remembers

After every exchange she pulls out lasting facts about you — where you live,
what you work on, how you take your coffee — and writes them to
`~/.arthur/memory.json`. They are fed back into every future conversation.

Turn it off with `ARTHUR_AMBIENT_MEMORY=0`. Read or prune the file whenever
you like; it is plain JSON. Ask her to "forget X" and she will, with a
confirmation.

---

## 8. Windows control

She can open apps, focus windows, control media and volume, read and write
the clipboard, take screenshots, and report battery and memory. All of it is
behind approval except reading stats and the clipboard.

`take_screenshot` needs `pillow`; `system_stats` gives fuller numbers with
`psutil`. Both are in the optional requirements.

---

## Security, in plain terms

- The audit log hash-chains every tool call. `verify` in the terminal checks
  it. Do not disable it.
- Phone tokens: `read` + `chat`. Never `admin`.
- Anything irreversible needs the confirm word spoken aloud, and is never
  automatic. Scheduled jobs cannot approve themselves at all.
- `ARTHUR_NTFY_TOPIC` is a secret. Treat it like a password.
- If you lose your phone, revoke its token: `DELETE /api/tokens/<id>`.

---

## Quick reference

| Command | What it does |
|---|---|
| `python -m arthur` | terminal chat |
| `python -m arthur talk` | always-on voice |
| `python -m arthur serve` | hub for phone and laptop |
| `python -m arthur watch` | reminders, jobs, mail triage |
| `python -m arthur jobs` | list what is scheduled |
