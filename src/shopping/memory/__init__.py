"""Shopping memory subsystem — persistent user profiles, sessions, and purchase data."""

from src.shopping.memory._db import get_memory_db, close_memory_db, MEMORY_DB_PATH

from src.shopping.memory.user_profile import (
    get_user_profile,
    update_user_profile,
    add_owned_item,
    remove_owned_item,
    set_preference,
    record_behavior,
    clear_user_data,
    init_user_profile_db,
)

from src.shopping.memory.price_watch import (
    add_price_watch,
    get_active_watches,
    get_all_active_watches,
    update_watch_price,
    trigger_watch,
    expire_old_watches,
    remove_watch,
    init_price_watch_db,
)

from src.shopping.memory.session import (
    create_session,
    get_session,
    update_session,
    add_session_product,
    add_session_question,
    get_recent_session,
    clear_session,
    init_session_db,
)

from src.shopping.memory.purchase_history import (
    log_purchase,
    get_purchase_history,
    get_recent_purchases,
    has_purchased,
    get_complementary_suggestions,
    init_purchase_history_db,
)


async def init_memory_db() -> None:
    """Initialise all memory tables (call once at startup)."""
    await init_user_profile_db()
    await init_price_watch_db()
    await init_session_db()
    await init_purchase_history_db()
