# Telegram Menu Redesign Plan

> **For agentic workers:** Implement this plan task-by-task. The design is fully specified.

**Goal:** Replace the 9-button reply keyboard with a single Menu button that opens a categorized inline menu system with Turkish labels. Add Quick Services (API-driven, no LLM), location setup, and monitoring dashboard.

**Architecture:** Single `[📂 Menu]` reply button → inline category menu → sub-menus per category. All 48 existing commands preserved, just reorganized.

---

## Reply Keyboard

Replace current 9-button layout with:

```python
REPLY_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton("📂 Menu")]],
    resize_keyboard=True,
    one_time_keyboard=False,
)
```

Update `_REPLY_BUTTON_MAP` to: `{"📂 Menu": "start"}`

---

## Menu Categories (7 groups, Turkish labels)

### Top-level inline menu (shown on Menu tap):
```
[⚡ Hizmetler] [🛒 Alışveriş]
[🎯 Görevler]  [📋 Yapılacaklar]
[📊 İzleme]    [⚙️ Sistem]
[        ⚙️ Ayarlar        ]
```

### ⚡ Hizmetler (Quick Services — no LLM, direct API calls)
```
[🏥 Nöbetçi Eczane] [💰 Döviz Kuru]
[🌤 Hava Durumu]    [⛽ Yakıt Fiyat]
[🕌 Namaz Vakti]    [📰 Haberler]
[🪙 Altın Fiyat]    [🌍 Deprem]
[🔙 Geri]
```

Each button triggers API call directly, returns formatted message. Location-dependent features (pharmacy, weather, prayer) trigger location setup if not configured.

### 🛒 Alışveriş (Shopping)
```
[🔍 Ürün Ara]       [💰 Fiyat Kontrol]
[⚖️ Karşılaştır]    [👁 Fiyat İzle]
[🔬 Detaylı Araştır] [🏷 Fırsatlarım]
[📦 Ürünlerim]
[🔙 Geri]
```
Maps to: shop, price, compare, watch, research_product, deals, mystuff

### 🎯 Görevler (Missions & Tasks)
```
[🎯 Yeni Görev]    [🔄 İş Akışı]
[📋 Hızlı Görev]   [📋 Görev Listesi]
[📬 Kuyruk]        [📄 Sonuç Gör]
[🛠 Görev Yönetimi]
[🔙 Geri]
```
Görev Yönetimi sub-submenu: cancel, pause, resume, priority, graph, wfstatus

### 📋 Yapılacaklar (Todo & Memory)
```
[📝 Ekle]     [📋 Listele]
[🗑 Tamamlananları Sil]
[💾 Hatırla]   [🔎 Ara]
[📥 İçerik Aktar]
[🔙 Geri]
```
Maps to: todo, todos, cleartodos, remember, recall, ingest

### 📊 İzleme (Monitoring)
```
[📊 Durum]     [📰 Günlük Özet]
[📈 İlerleme]  [🤖 Model Durum]
[📉 Metrikler] [💰 Maliyet]
[💵 Bütçe]     [🐛 Debug]
[📭 DLQ]
[🔙 Geri]
```
Maps to: status, digest, progress, modelstats, metrics, cost, budget, debug, dlq

### ⚙️ Sistem (System)
```
[🎚 Otonomi]    [🖥 GPU Yükü]
[🎛 Ayar]       [🔑 Kimlik]
[🗂 Çalışma Alanı] [🧪 İyileştir]
--- Tehlikeli ---
[♻️ Sıfırla]    [☢️ Tümünü Sıfırla]
[🔄 Yeniden Başlat] [⏹ Durdur]
[🔙 Geri]
```
Maps to: autonomy, load, tune, credential, workspace, improve, reset, resetall, restart, stop

### ⚙️ Ayarlar (Settings)
```
[📍 Konum Ayarla]    [🌐 Dil]
[🔔 Bildirim Ayarları]
[🔙 Geri]
```

---

## Callback Data Convention

Pattern: `m:<category>:<action>:<command>`

- `m:top` — show top-level menu
- `m:quick` — open Quick Services submenu
- `m:quick:pharmacy` — trigger pharmacy lookup
- `m:quick:exchange` — trigger exchange rates
- `m:shop` — open Shopping submenu
- `m:shop:ask:shop` — prompt for product search
- `m:shop:run:deals` — execute deals (no arg needed)
- `m:task:manage` — open management sub-submenu
- `m:set:location` — start location setup

Old prefixes `menu_cat:`, `menu_cmd:`, `menu_ask:`, `menu_back` replaced by `m:` prefix.
Entity callbacks unchanged: `todo_toggle:`, `todo_help:`, `wfstatus:`, `wfcancel:`, `approve_`, `reject_`

---

## User Preferences Table

```sql
CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

DB functions: `get_user_pref(key, default)`, `set_user_pref(key, value)`, `get_all_user_prefs()`

Location keys: `location_lat`, `location_lon`, `location_district`, `location_city`, `location_source` (gps/geocoded), `location_updated`

---

## Location Setup Flow

Triggered when location-dependent Quick Service is used and no location saved.

1. Show: "📍 Konum bilgin henüz kayıtlı değil."
2. Buttons: `[📍 Konumumu Paylaş]` (Telegram native location), `[✏️ İlçe Adı Yaz]`, `[❌ İptal]`
3a. GPS: Telegram sends lat/lon → reverse geocode via Nominatim → save all 6 keys → proceed
3b. Text: User types district → geocode via Nominatim → confirm → save → proceed
4. Never asked again. Editable via Settings.

Implementation: `KeyboardButton(text="📍 Konumumu Paylaş", request_location=True)` + `MessageHandler(filters.LOCATION, handle_location)`

---

## Quick Service Handlers

Each Quick Service is a direct function call, no task creation:

| Service | Function | API |
|---------|----------|-----|
| Pharmacy | `find_nearest_pharmacy()` | Nosyapi + OSRM |
| Exchange | `call_api("Frankfurter")` | Frankfurter/TCMB |
| Weather | `call_api("wttr.in")` | wttr.in |
| Fuel | `call_api("Turkey Fuel Prices")` | CollectAPI |
| Prayer | `call_api("Diyanet Prayer Times")` | Diyanet |
| News | `web_search("son dakika")` | ddgs |
| Gold | `call_api("Gold Price Turkey")` | CollectAPI |
| Earthquake | `call_api("Kandilli Observatory")` | Kandilli |

---

## Migration Steps

1. Add `user_preferences` table to `db.py`
2. Add `get_user_pref`, `set_user_pref`, `get_all_user_prefs` to `db.py`
3. Update `REPLY_KEYBOARD` to single button
4. Define new `MENU_CATEGORIES_V2` with Turkish labels
5. Build new callback handler for `m:` prefix
6. Add Quick Service handler functions (inline, no task creation)
7. Add location setup flow (`handle_location`, `_pending_action` for text input)
8. Add Settings submenu handlers
9. Update `cmd_start` to show new top-level menu
10. Register `MessageHandler(filters.LOCATION)` in `_setup_handlers`
11. Test each category end-to-end
12. Remove old `_REPLY_BUTTON_MAP` entries for deleted buttons
