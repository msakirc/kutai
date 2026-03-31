# Telegram Menu Redesign — Final Design

## Principles

1. **Reply keyboard for navigation** — all menu navigation happens via reply keyboard swaps, no inline keyboards for navigation
2. **Inline keyboards for contextual actions only** — todo toggles, mission pause/cancel, confirmations
3. **Auto-list on entry** — Listem and Gorevler show content immediately when tapped
4. **Dashboard on entry** — Sistem prints status dashboard automatically
5. **Sub-steps save LLM calls** — workflow selection on mission creation, not LLM classification
6. **Back button everywhere** — every sub-keyboard has Geri
7. **Wrapper handles lifecycle** — Sistem button polled by wrapper for health check fallback

---

## Reply Keyboard States

### Default (app running)

```
[⚡ Hizmet] [🛒 Alışveriş] [📋 Listem]
[🎯 Görevler] [⚙️ Sistem]
```

### App Down (wrapper polling)

```
[▶️ Başlat]
```

Wrapper sends this keyboard + warning message when it detects the orchestrator is dead/hung.

---

## Category: ⚡ Hizmet (Quick Services)

Keyboard becomes:
```
[🏥 Eczane] [💰 Döviz] [🌤 Hava]
[⛽ Yakıt]  [🕌 Namaz] [📰 Haber]
[🪙 Altın]  [🌍 Deprem]
[🔙 Geri]
```

Each button fires an API call, returns result as message. No LLM needed.

Location-dependent buttons (Eczane, Hava, Namaz): if no location saved in `user_preferences`, triggers location setup flow (see below).

---

## Category: 🛒 Alışveriş (Shopping)

Keyboard becomes:
```
[⚡ Hızlı Ara] [🔬 Detaylı Araştır]
[🔙 Geri]
```

| Button | What it does |
|--------|-------------|
| Hızlı Ara | Prompts for query → single shopping_advisor agent |
| Detaylı Araştır | Prompts for query → multi-task shopping mission (research → compare → recommend) |

---

## Category: 📋 Listem (Todos)

Tapping sends todo list as message automatically:
```
📋 Listem

☐ Marketten süt al
☐ Elektrik faturası öde
☑ Kargo takip et
```
With inline toggle buttons per item (existing todo_toggle: flow).

Keyboard becomes:
```
[📝 Yeni Ekle] [⏰ Hatırlat]
[🔙 Geri]
```

### Hatırlat (Quick Reminder)
One-shot timed reminders. Flow:
1. Tap ⏰ Hatırlat → "Ne zaman?" (10dk, 1 saat, 14:30, yarın 09:00)
2. Type time → "Ne hatırlatılsın?"
3. Type text → creates scheduled_task with one-shot flag, fires at the specified time

Yeni Ekle prompts "Ne ekleyelim?" then adds and refreshes.

---

## Category: 🎯 Görevler (Missions)

Tapping lists active missions as message automatically with selectable items:
```
🎯 Görevler

1. ☕ Kahve makinesi araştır [çalışıyor]
2. 📝 Blog yazısı [bekliyor]
```

Keyboard becomes:
```
[🎯 Yeni Görev] [📬 İş Kuyruğu]
[⏰ Zamanla]
[🔙 Geri]
```

### Zamanla (Recurring Task Scheduler)
Schedule recurring tasks with cron expressions. Flow:
1. Tap ⏰ Zamanla → shows existing scheduled tasks + "Yeni eklemek için açıklama yaz"
2. Type description → "Ne sıklıkla?" (her gün 09:00, her 2 saatte, her pazartesi)
3. Type schedule → creates scheduled_task with cron expression

Tapping a mission (via inline selector) shows contextual inline buttons:
```
☕ Kahve makinesi araştır
Durum: çalışıyor | Öncelik: 5

[⏸ Duraklat] [🚫 İptal]
[🔢 Öncelik] [📄 Sonuç]
[🔙 Geri]
```

### Yeni Görev Flow

1. Prompt: "Ne yapılsın?"
2. User types description
3. Workflow selection keyboard:

```
[⚡ Hızlı Cevap] [📊 Araştır & Raporla]
[🏗 Yeni Proje]  [🤖 Otomatik]
[💻 Kod / Diğer]
[🔙 Geri]
```

| Button | Workflow | What it does |
|--------|----------|-------------|
| Hızlı Cevap | (none) | Single agent, quick answer |
| Araştır & Raporla | research | 5-phase: Question → Search → Sources → Synthesize → Report |
| Yeni Proje | i2p_v2 | Full dev pipeline: Plan → Code → Test → Deploy |
| Otomatik | (LLM decides) | LLM picks the right workflow |
| Kod / Diğer | — | Shows reminder message that these are typed, not menu-driven |

Tap "Kod / Diğer":
```
🚧 Bu iş akışları henüz menüden desteklenmiyor.
Görevini yazarak gönder, LLM doğru akışa yönlendirecek.

Örnekler:
• "router.py'deki hatayı düzelt"
• "db modülünü refaktör et"
• "agent base sınıfını dokümante et"
```

---

## Category: ⚙️ Sistem

Tapping auto-prints dashboard:
```
📊 KutAI Durum

🤖 Qwen2.5-32B-Q4 | 14.2 t/s
   GPU: 18/24 katman | 11.2/16 GB VRAM
   Thinking: ON | Yük Modu: full

📋 Kuyruk: 3 bekleyen | 1 çalışan
   Bugün: 12 tamamlandı | 1 başarısız

🔄 Model değişim: 2/3 (son 5dk)
💰 Bugün: $0.42 / $2.00
⏱ Çalışma: 4s 23dk
🛡️ Otonomi: medium
```

Keyboard becomes:
```
[🖥 Yük Modu]  [🐛 Debug]
[📭 DLQ]
[🔄 Yeniden Başlat] [⏹ Durdur]
[🔙 Geri]
```

### Yük Modu
Shows current mode, keyboard:
```
[⚡ Full] [🔋 Heavy] [⚖️ Shared]
[🔻 Minimal] [🤖 Otomatik]
[🔙 Geri]
```

### Debug
Message with recent tasks:
```
🐛 Son Görevler

1. ☕ Kahve makinesi araştır — shopping_advisor — 2dk önce
2. 📊 Hava durumu — weather_api — 5dk önce
3. 📝 Blog yazısı — researcher — 12dk önce
```
Keyboard:
```
[1️⃣] [2️⃣] [3️⃣]
[🔙 Geri]
```
Tap one → task detail (agent, model, duration, tokens, sources, status).

### DLQ
Message with failed/stuck tasks, same pattern as Debug. Tap one → error details + optional retry.

### Restart / Stop
Confirmation via inline keyboard:
```
⚠️ KutAI yeniden başlatılsın mı?
[✅ Evet] [❌ Hayır]
```

---

## Wrapper Behavior

### Normal Operation
- Wrapper polls Telegram alongside the bot
- Wrapper watches for `⚙️ Sistem` button tap
- On seeing Sistem: checks orchestrator health (heartbeat file)
- If healthy: ignores, lets bot handle
- If dead/hung: kills process, sends `[▶️ Başlat]` keyboard

### App Down
- Wrapper shows: `⚠️ KutAI durdu. Başlatmak için butona bas.` + `[▶️ Başlat]`
- On Başlat tap: starts orchestrator
- On startup: bot sends full 5-button keyboard with first message

### Health Check
- Orchestrator writes timestamp to `logs/heartbeat` every 30 seconds
- Wrapper checks if timestamp is older than 90 seconds = hung
- On hang detection: kill orchestrator, send Başlat keyboard

---

## Location Setup Flow

Triggered when user taps a location-dependent Quick Service and no location is saved.

1. Bot sends message with temporary reply keyboard:
   ```
   📍 Konum bilgin henüz kayıtlı değil.
   [📍 Konumumu Paylaş]  ← Telegram native location picker
   [✏️ İlçe Adı Yaz]
   [❌ İptal]
   ```
2a. User shares GPS → reverse geocode → save → proceed with original request
2b. User types district name → geocode → confirm → save → proceed
3. Saved to `user_preferences`: lat, lon, district, city
4. Subsequent requests skip setup

---

## Pending Action Timeout

- `_pending_action` expires after 5 minutes
- If user taps a different category button while pending: clear pending, handle new input
- Every prompt shows `[❌ Vazgeç]` inline button to explicitly cancel

---

## Callback Data Convention

Navigation callbacks (inline, contextual only):
- `m:task:detail:<id>` — show mission detail
- `m:task:pause:<id>` — pause specific mission
- `m:task:cancel:<id>` — cancel specific mission
- `m:debug:detail:<id>` — show task debug detail
- `m:dlq:detail:<id>` — show DLQ item detail
- `m:confirm:restart` / `m:confirm:stop` — lifecycle confirmation

Existing callbacks unchanged:
- `todo_toggle:`, `todo_help:`, `todo_help_accept:`, `todo_help_cancel`, `todo_close`

---

## Dropped Items

See `memory/project_menu_disconnected_items.md` for full list. Decision deferred — test new menu first, then decide what to reintroduce or permanently remove.

Key drops: price watch, mystuff, deals, compare, mission_wf merge, standalone pause/resume/cancel, autonomy button, credential button, workspace, metrics (broken).
