# dabidabi

**The DB engine.** Owns the SQLite layer for KutAI: the single long-lived
aiosqlite connection (WAL, autocommit), per-mission transaction-lock sharding,
schema creation + migrations (`init_db`), and the shared query helpers that
every package and the app use to read/write the database.

Named in the house style (Yaşar Usta, Fatih Hoca, …). `dabidabi` is the layer
nobody thinks about until the writes stop landing.

## Why a package

`src/infra/db.py` used to live inside the app tree, so every package that
touched the DB had to `import src.infra.db` — which makes those packages
impossible to publish standalone (you can't `pip install` a unit that reaches
back into an app). Lifting the engine into `dabidabi` removes that reach: a
package depends on `dabidabi`, not on `src/`.

This is the **engine** only. Relationally-coupled, domain-specific tables
(missions/tasks spine, model registry, cost ledger, …) migrate into their
owning packages over time — `dabidabi` keeps the connection/lock/migration
machinery and the generic query layer, not every domain's schema forever.

## Public API (selected)

```python
from dabidabi import get_db, close_db, configure, DB_PATH, init_db
```

- `get_db()` — the singleton aiosqlite connection (WAL, `isolation_level=None`).
- `configure(db_path)` — override the active DB path at runtime (app startup /
  standalone embed). Otherwise `DB_PATH` resolves from the `DB_PATH` env var
  (with a `.env` load) and must be absolute.
- `init_db()` — create schema + run migrations.
- plus the mission/task/cost/growth/… query helpers (back-compat surface).

## Install

Editable, like the other local packages:

```
-e ./packages/db
```

`src/infra/db.py` and `src/infra/times.py` remain as thin `sys.modules` aliases
to `dabidabi` / `dabidabi.times` so existing `src.infra.*` imports keep working
during the migration.

---

# dabidabi (TR)

**Veritabanı motoru.** KutAI'nin SQLite katmanını yönetir: tek uzun-ömürlü
aiosqlite bağlantısı (WAL, autocommit), görev-bazlı (per-mission) transaction
kilit paylaştırması, şema oluşturma + migration (`init_db`) ve tüm paketlerin
kullandığı ortak sorgu yardımcıları.

Ev geleneğindeki gibi isimlendirildi (Yaşar Usta, Fatih Hoca, …). Yazımlar
durana kadar kimsenin aklına gelmeyen katman.

## Neden paket

`src/infra/db.py` uygulama ağacının içindeydi; DB'ye dokunan her paket
`import src.infra.db` yapmak zorundaydı — bu da o paketleri tek başına
yayınlanamaz kılıyor (bir uygulamaya geri uzanan birimi `pip install`
edemezsiniz). Motoru `dabidabi`'ye taşımak bu bağımlılığı kaldırır.

Bu sadece **motor**. İlişkisel olarak bağlı, alana özgü tablolar (mission/task
omurgası, model registry, maliyet defteri, …) zamanla kendi paketlerine taşınır.
