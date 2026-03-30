"""Seed the skills database with curated routing skills."""

from src.infra.logging_config import get_logger

logger = get_logger("memory.seed_skills")

SEED_SKILLS = [
    # --- Currency / Exchange Rates ---
    {
        "name": "currency_api_routing",
        "description": "For currency and exchange rate queries, use api_call with TCMB or Frankfurter instead of web_search. Much faster and more accurate.",
        "trigger_pattern": "dolar|euro|kur|currency|exchange.rate|döviz|sterling|pound|yen|altın.fiyat",
        "tool_sequence": "tool=api_call, api_name=TCMB EVDS (Turkish rates) or Frankfurter (international). Do NOT use web_search for simple rate lookups.",
        "examples": "dolar kuru ne; EUR/TRY rate; current gold price in TL; 100 USD to TRY",
    },
    # --- Weather ---
    {
        "name": "weather_api_routing",
        "description": "For weather queries, use api_call with wttr.in or Open-Meteo instead of web_search.",
        "trigger_pattern": "weather|hava.durumu|sıcaklık|temperature|yağmur|rain|forecast|tahmin",
        "tool_sequence": "tool=api_call, api_name=wttr.in (simple) or Open-Meteo (detailed). Format: wttr.in/{city}?format=j1",
        "examples": "istanbul hava durumu; weather in ankara tomorrow; will it rain today",
    },
    # --- Time ---
    {
        "name": "time_api_routing",
        "description": "For time/timezone queries, use api_call with WorldTimeAPI.",
        "trigger_pattern": "what.time|saat.kaç|timezone|time.in|saat.farkı",
        "tool_sequence": "tool=api_call, api_name=WorldTimeAPI. Endpoint: worldtimeapi.org/api/timezone/{Area}/{City}",
        "examples": "what time is it in Tokyo; saat kaç; time difference between Istanbul and New York",
    },
    # --- Wikipedia / Quick Facts ---
    {
        "name": "wikipedia_routing",
        "description": "For encyclopedia-style factual queries about people, places, concepts, use api_call with Wikipedia API before web_search.",
        "trigger_pattern": "who.is|kim|nedir|what.is|wikipedia|vikipedi|tarih|history.of|biography",
        "tool_sequence": "tool=api_call, api_name=Wikipedia API or Wikipedia TR. Try TR first for Turkish topics.",
        "examples": "Atatürk kim; what is quantum computing; history of Istanbul",
    },
    # --- Play Store / App Research ---
    {
        "name": "play_store_routing",
        "description": "For mobile app research, competitor analysis, app reviews, use the play_store tool instead of web_search.",
        "trigger_pattern": "play.store|app.store|mobile.app|uygulama|android|ios|app.review|rakip.uygulama|competitor.app",
        "tool_sequence": "tool=play_store. Actions: search (find apps), app (details), reviews (user reviews), similar (competitors).",
        "examples": "find review apps on Play Store; competitor apps for todo list; Trendyol app reviews",
    },
    # --- GitHub / Open Source ---
    {
        "name": "github_routing",
        "description": "For open source project research, code examples, repo analysis, use the github tool instead of web_search.",
        "trigger_pattern": "github|repo|repository|açık.kaynak|open.source|source.code|kaynak.kod|stars|fork",
        "tool_sequence": "tool=github. Actions: repos (search), code (search code), readme (fetch README). Use GITHUB_TOKEN for higher rate limits.",
        "examples": "find Python web scraping libraries on GitHub; llama.cpp repo README; trending AI repos",
    },
    # --- Shopping: Turkish E-commerce ---
    {
        "name": "shopping_turkish_sources",
        "description": "For Turkish product shopping, use shopping_search which queries Trendyol, Hepsiburada, Akakçe, Amazon TR scrapers. For price comparison specifically, Akakçe is best.",
        "trigger_pattern": "fiyat|ücret|price|trendyol|hepsiburada|akakçe|amazon.tr|n11|satın.al|buy",
        "tool_sequence": "tool=shopping_search for product discovery. For price comparison: Akakçe aggregates all retailers. For reviews: Trendyol has the best review API.",
        "examples": "iPhone 15 fiyat; en ucuz RTX 4070; kahve makinesi karşılaştır",
    },
    # --- Shopping: Reviews & Complaints ---
    {
        "name": "shopping_review_sources",
        "description": "For product reviews, user complaints, brand reputation in Turkey, use shopping_fetch_reviews with specific scrapers.",
        "trigger_pattern": "review|yorum|şikayet|complaint|sikayetvar|technopat|donanimhaber|kullanıcı.yorumu",
        "tool_sequence": "tool=shopping_fetch_reviews. For complaints: use source=sikayetvar. For tech reviews: use source=technopat or source=donanimhaber. For e-commerce reviews: use source=trendyol or source=hepsiburada.",
        "examples": "Bosch çamaşır makinesi yorumları; Samsung şikayetleri; laptop forum reviews",
    },
    # --- Football / Sports ---
    {
        "name": "sports_web_search",
        "description": "For live sports data (scores, lineups, match predictions), always use web_search with search_depth=standard. API alternatives are limited for Turkish football.",
        "trigger_pattern": "maç|kadro|skor|score|lineup|predicted.xi|süper.lig|champions.league|football|futbol",
        "tool_sequence": "tool=web_search. These queries are time-sensitive — the classifier should set search_depth=standard automatically. Always search, never answer from memory.",
        "examples": "turkey predicted xi; Galatasaray maç skoru; Champions League results tonight",
    },
    # --- PDF / Document Processing ---
    {
        "name": "pdf_processing",
        "description": "For reading PDF files, extracting text, analyzing documents, use read_pdf or read_pdf_advanced tool.",
        "trigger_pattern": "pdf|document|döküman|belge|rapor|report|extract.text|oku",
        "tool_sequence": "tool=read_pdf_advanced (multi-backend: PyMuPDF > pdfplumber > PyPDF2). Pass file_path and optional max_pages.",
        "examples": "read this PDF; extract text from report.pdf; analyze the document at /path/to/file.pdf",
    },
    # --- Coding: Error Lookup ---
    {
        "name": "coding_error_search",
        "description": "For programming error messages, stack traces, use web_search targeting Stack Overflow and GitHub issues.",
        "trigger_pattern": "error|hata|traceback|exception|TypeError|ImportError|ModuleNotFoundError|stack.overflow",
        "tool_sequence": "tool=web_search with query=error message. Add 'site:stackoverflow.com' or 'site:github.com/issues' to query for better results.",
        "examples": "TypeError: 'bool' object is not iterable; ModuleNotFoundError: No module named 'trafilatura'",
    },
    # --- Translation ---
    {
        "name": "translation_routing",
        "description": "For translation requests, try api_call with LibreTranslate. Falls back to web_search if API unavailable.",
        "trigger_pattern": "translate|çevir|tercüme|translation|İngilizce|Türkçe|English|Turkish",
        "tool_sequence": "tool=api_call, api_name=LibreTranslate. POST to /translate with q, source, target params. If unavailable, use web_search.",
        "examples": "translate 'hello world' to Turkish; bu cümleyi İngilizceye çevir",
    },
    # --- News ---
    {
        "name": "news_routing",
        "description": "For current news queries, use web_search with search_depth=quick. For structured news, use api_call with GNews (if GNEWS_API_KEY is set).",
        "trigger_pattern": "news|haber|son.dakika|breaking|gündem|headline|güncel",
        "tool_sequence": "Check api_lookup for GNews first. If GNEWS_API_KEY is set, use api_call. Otherwise, web_search with search_depth=quick.",
        "examples": "son dakika haberleri; Turkey news today; tech news this week",
    },
    # --- IP / Network ---
    {
        "name": "network_tools_routing",
        "description": "For IP lookup, geolocation, network diagnostics, use api_call with ipapi or shell commands.",
        "trigger_pattern": "ip.address|my.ip|geolocation|ping|dns|network|bağlantı",
        "tool_sequence": "tool=api_call, api_name=ipapi for IP geolocation. tool=shell for ping, traceroute, nslookup.",
        "examples": "what is my IP; geolocate this IP; ping google.com",
    },
    # --- i2p Workflow: Competitor Research ---
    {
        "name": "i2p_competitor_research",
        "description": "For idea-to-product competitor research phases, combine play_store (app competitors) + github (open source alternatives) + web_search (market analysis).",
        "trigger_pattern": "competitor|rakip|alternative|alternatif|market.analysis|pazar.analizi|similar.apps|benzer",
        "tool_sequence": "1. play_store action=search to find competing apps. 2. play_store action=similar for direct competitors. 3. github action=repos for open source alternatives. 4. web_search for market size and trends.",
        "examples": "find competitors for a review platform app; analyze the todo app market; open source alternatives to Notion",
    },
    # --- Pharmacy on Duty ---
    {
        "name": "pharmacy_on_duty",
        "description": "For finding pharmacies on duty (nöbetçi eczane) in Turkey, use the pharmacy tool which calculates distance from your location.",
        "trigger_pattern": "eczane|pharmacy|nöbetçi|nobetci|on.duty|ilaç|medicine|eczaneler",
        "tool_sequence": "tool=pharmacy. Pass city='ankara' for all districts, or city='ankara' district='cankaya' for specific district. Falls back to eczaneler.gen.tr web scraping if no API key.",
        "examples": "nöbetçi eczane kadıköy; pharmacy on duty istanbul; en yakın açık eczane; ankara nöbetçi eczaneler; nöbetçi eczane istanbul",
    },
    # --- Earthquake ---
    {
        "name": "earthquake_data",
        "description": "For earthquake data in Turkey, use api_call with Kandilli Observatory. Real-time seismic data.",
        "trigger_pattern": "deprem|earthquake|sarsıntı|kandilli|seismic|zelzele",
        "tool_sequence": "tool=api_call, api_name=Kandilli Observatory. Returns live earthquake list with magnitude, location, depth.",
        "examples": "son depremler; earthquake in Turkey today; was there an earthquake",
    },
    # --- Gas Prices ---
    {
        "name": "fuel_price_routing",
        "description": "For fuel/gas prices in Turkey, use api_call with Turkey Fuel Prices.",
        "trigger_pattern": "benzin|mazot|diesel|lpg|yakıt|fuel|gas.price|akaryakıt|petrol",
        "tool_sequence": "tool=api_call, api_name=Turkey Fuel Prices. Requires COLLECTAPI_KEY.",
        "examples": "benzin fiyatı; mazot ne kadar; LPG fiyat istanbul",
    },
    # --- Gold Prices ---
    {
        "name": "gold_price_routing",
        "description": "For gold prices in Turkey (gram altın, çeyrek, yarım, tam), use api_call with Gold Price Turkey.",
        "trigger_pattern": "altın|gold|çeyrek|gram.altın|yarım.altın|tam.altın|cumhuriyet.altını",
        "tool_sequence": "tool=api_call, api_name=Gold Price Turkey. Requires COLLECTAPI_KEY.",
        "examples": "altın fiyatı; gram altın ne kadar; çeyrek altın fiyat",
    },
    # --- Map/Directions ---
    {
        "name": "map_directions_routing",
        "description": "For directions, distance, and routing between locations, use api_call with OSRM or Nominatim for geocoding. Privacy-safe (no Google).",
        "trigger_pattern": "yol.tarifi|directions|route|distance|mesafe|nasıl.gidilir|how.to.get|harita|map",
        "tool_sequence": "1. Geocode addresses with api_call api_name=Nominatim. 2. Get route with api_call api_name=OSRM. Both are free, no API key, privacy-safe.",
        "examples": "Kadıköy'den Taksim'e nasıl gidilir; distance from Ankara to Istanbul; walking route to nearest pharmacy",
    },
    # --- Prayer Times ---
    {
        "name": "prayer_times_routing",
        "description": "For prayer/namaz times in Turkey, use api_call with Diyanet Prayer Times.",
        "trigger_pattern": "namaz|ezan|prayer|imsak|iftar|sahur|akşam.ezanı|öğle.namazı",
        "tool_sequence": "tool=api_call, api_name=Diyanet Prayer Times. Query by district code.",
        "examples": "namaz vakitleri istanbul; ezan ne zaman; iftar saati",
    },
    # --- Travel/Tickets ---
    {
        "name": "travel_ticket_routing",
        "description": "For flight and bus ticket prices in Turkey, check EnUygun MCP or use web_search targeting enuygun.com, obilet.com.",
        "trigger_pattern": "uçak|flight|bilet|ticket|otobüs|bus|enuygun|obilet|seyahat|travel",
        "tool_sequence": "First try: web_search targeting enuygun.com or obilet.com. EnUygun also has an MCP endpoint at mcp.enuygun.com.",
        "examples": "istanbul ankara uçak bileti; en ucuz otobüs bileti; flight prices to Antalya",
    },
    # --- Turkish Holidays ---
    {
        "name": "turkish_holidays_routing",
        "description": "For Turkish public holidays and calendar, use api_call with Turkey Holidays.",
        "trigger_pattern": "tatil|holiday|resmi.tatil|bayram|arife|ramazan|kurban",
        "tool_sequence": "tool=api_call, api_name=Turkey Holidays. Returns official public holidays for any year.",
        "examples": "2026 resmi tatiller; next public holiday in Turkey; bayram ne zaman",
    },
]


async def seed_skills():
    """Seed the skills database with curated routing skills.

    Only adds skills that don't already exist (by name).
    Returns the number of new skills added.
    """
    from .skills import add_skill, list_skills, _ensure_table
    from ..infra.db import get_db

    db = await get_db()
    await _ensure_table(db)

    existing = await list_skills()
    existing_names = {s["name"] for s in existing}

    added = 0
    for skill in SEED_SKILLS:
        if skill["name"] in existing_names:
            continue
        await add_skill(**skill)
        added += 1

    if added:
        logger.info(f"Seeded {added} routing skills")
    return added
