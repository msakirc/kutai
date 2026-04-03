"""Seed the skills database with curated execution recipe skills."""

from src.infra.logging_config import get_logger

logger = get_logger("memory.seed_skills")

SEED_SKILLS = [
    {
        "name": "currency_lookup",
        "description": "Looking up currency exchange rates, conversion between currencies, checking current dollar/euro/gold prices in Turkish Lira",
        "strategy_summary": "Use api_call with TCMB for Turkish rates or Frankfurter for international. Faster and more accurate than web search for simple rate lookups.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "weather_check",
        "description": "Checking weather forecasts, current temperature, rain predictions for a specific city or location",
        "strategy_summary": "Use api_call with wttr.in (simple format) or Open-Meteo (detailed forecast). Format: wttr.in/{city}?format=j1. Avoid web_search for basic weather.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "timezone_lookup",
        "description": "Finding current time in a city, timezone differences, time conversion between locations",
        "strategy_summary": "Use api_call with WorldTimeAPI. Endpoint: worldtimeapi.org/api/timezone/{Area}/{City}.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "encyclopedia_lookup",
        "description": "Looking up factual information about people, places, historical events, scientific concepts from encyclopedia sources",
        "strategy_summary": "Use api_call with Wikipedia API. Try Turkish Wikipedia first for Turkish topics, then English. Structured data beats web_search for factual lookups.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "app_store_research",
        "description": "Researching mobile apps, finding app alternatives, reading app reviews, comparing app features and ratings across stores",
        "strategy_summary": "Use play_store tool: search (find apps), app (details), reviews (user reviews), similar (competitors). For competitor analysis, combine play_store with github for open-source alternatives.",
        "tools_used": ["play_store", "smart_search"],
    },
    {
        "name": "github_code_research",
        "description": "Searching for open source projects, code examples, library comparison, repository analysis on GitHub",
        "strategy_summary": "Use github tool: repos (search projects), code (search code), readme (fetch README). Combine with web_search for ecosystem context. Use GITHUB_TOKEN for higher rate limits.",
        "tools_used": ["github", "smart_search"],
    },
    {
        "name": "turkish_product_shopping",
        "description": "Shopping for products in Turkey, finding prices on Turkish e-commerce sites, comparing prices across Trendyol, Hepsiburada, Akakce, Amazon TR",
        "strategy_summary": "Use shopping_search for product discovery across Turkish retailers. For price comparison, Akakce aggregates all retailers. For reviews, Trendyol has the best review data. Search multiple sources then compare.",
        "tools_used": ["shopping_search", "smart_search"],
    },
    {
        "name": "product_review_research",
        "description": "Finding product reviews, user complaints, brand reputation analysis in Turkish sources including Sikayetvar, Technopat, DonanımHaber",
        "strategy_summary": "Use shopping_fetch_reviews with specific sources: sikayetvar for complaints, technopat/donanimhaber for tech reviews, trendyol/hepsiburada for e-commerce reviews. Aggregate multiple sources for balanced view.",
        "tools_used": ["shopping_fetch_reviews", "smart_search"],
    },
    {
        "name": "sports_live_data",
        "description": "Getting live sports scores, match lineups, football predictions, league standings, especially Turkish Super Lig and Champions League",
        "strategy_summary": "Use web_search with search_depth=standard. Sports data is time-sensitive — always search, never answer from memory. No reliable free API for Turkish football.",
        "tools_used": ["web_search", "smart_search"],
    },
    {
        "name": "pdf_document_processing",
        "description": "Reading PDF files, extracting text from documents, analyzing report contents",
        "strategy_summary": "Use read_pdf_advanced (multi-backend: PyMuPDF > pdfplumber > PyPDF2). Pass file_path and optional max_pages parameter.",
        "tools_used": ["read_pdf_advanced"],
    },
    {
        "name": "programming_error_diagnosis",
        "description": "Diagnosing programming errors, looking up stack traces, finding solutions for exception messages and error codes",
        "strategy_summary": "Use web_search targeting Stack Overflow and GitHub issues. Add 'site:stackoverflow.com' or 'site:github.com/issues' to query for focused results.",
        "tools_used": ["web_search", "smart_search"],
    },
    {
        "name": "text_translation",
        "description": "Translating text between languages, especially Turkish-English translation",
        "strategy_summary": "Try api_call with LibreTranslate first (POST /translate with q, source, target). Falls back to web_search if API unavailable.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "current_news_lookup",
        "description": "Finding current news, breaking news headlines, today's news in Turkey or worldwide",
        "strategy_summary": "Check api_lookup for GNews first (needs GNEWS_API_KEY). Otherwise web_search with search_depth=quick. News queries are time-sensitive.",
        "tools_used": ["smart_search", "web_search"],
    },
    {
        "name": "network_diagnostics",
        "description": "Looking up IP addresses, geolocation, running network diagnostics like ping and DNS queries",
        "strategy_summary": "Use api_call with ipapi for IP geolocation. Use shell for ping, traceroute, nslookup commands.",
        "tools_used": ["smart_search", "api_call", "shell"],
    },
    {
        "name": "competitor_analysis_research",
        "description": "Researching competitors for a product idea, market analysis, finding similar apps and open-source alternatives",
        "strategy_summary": "1. play_store search for competing apps. 2. play_store similar for direct competitors. 3. github repos for open-source alternatives. 4. web_search for market size and trends. Combine all sources.",
        "tools_used": ["play_store", "github", "smart_search", "web_search"],
    },
    {
        "name": "pharmacy_finder",
        "description": "Finding pharmacies on duty (nobetci eczane) in Turkey, nearest open pharmacy with distance calculation",
        "strategy_summary": "Use pharmacy tool. Pass city for all districts, or city+district for specific area. Falls back to eczaneler.gen.tr web scraping if no API key.",
        "tools_used": ["pharmacy"],
    },
    {
        "name": "earthquake_data_lookup",
        "description": "Checking recent earthquakes in Turkey, seismic activity data from Kandilli Observatory",
        "strategy_summary": "Use api_call with Kandilli Observatory. Returns live earthquake list with magnitude, location, depth. Real-time data.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "fuel_price_lookup",
        "description": "Checking current fuel prices in Turkey: gasoline, diesel, LPG prices by city",
        "strategy_summary": "Use api_call with Turkey Fuel Prices. Requires COLLECTAPI_KEY in .env.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "gold_price_lookup",
        "description": "Checking gold prices in Turkey: gram altin, ceyrek, yarim, tam, cumhuriyet altini",
        "strategy_summary": "Use api_call with Gold Price Turkey. Requires COLLECTAPI_KEY in .env.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "directions_and_routing",
        "description": "Getting directions between locations, calculating distance, finding routes for driving or walking",
        "strategy_summary": "1. Geocode addresses with api_call HERE Geocoding (or Photon for privacy). 2. Get route with api_call OSRM. OSRM and Photon are free, no API key needed.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "prayer_times_lookup",
        "description": "Looking up prayer times (namaz vakitleri), ezan times, iftar/sahur times in Turkey",
        "strategy_summary": "Use api_call with Diyanet Prayer Times. Query by district code.",
        "tools_used": ["smart_search", "api_call"],
    },
    {
        "name": "travel_ticket_search",
        "description": "Searching for flight, bus, train tickets and prices in Turkey, especially YHT and domestic flights",
        "strategy_summary": "1. api_call Kiwi Tequila (needs KIWI_API_KEY, free, 750+ carriers). 2. api_call Rome2rio for route planning. 3. web_search targeting enuygun.com or obilet.com as fallback.",
        "tools_used": ["smart_search", "api_call", "web_search"],
    },
    {
        "name": "product_spec_comparison",
        "description": "Detailed product specification comparison using Epey.com, finding products by specific technical requirements like RAM, GPU, screen size",
        "strategy_summary": "Use shopping_search with epey.com source. Epey has 85+ spec fields per product. For detailed specs, use get_product_details(url) on individual products. Best for 'find laptop with X and Y' queries.",
        "tools_used": ["shopping_search"],
    },
    {
        "name": "turkish_holiday_lookup",
        "description": "Looking up Turkish public holidays, bayram dates, official holiday calendar for any year",
        "strategy_summary": "Use api_call with Turkey Holidays. Returns official public holidays for any year.",
        "tools_used": ["smart_search", "api_call"],
    },
]


async def seed_skills():
    """Seed the skills database with curated execution recipe skills.

    Only adds skills that don't already exist (by name).
    Returns the number of new skills added.
    """
    from .skills import list_skills, add_skill

    existing = await list_skills()
    existing_names = {s["name"] for s in existing}

    added = 0
    for skill in SEED_SKILLS:
        if skill["name"] in existing_names:
            continue
        await add_skill(
            name=skill["name"],
            description=skill["description"],
            strategy_summary=skill["strategy_summary"],
            tools_used=skill.get("tools_used", []),
            source_grade="seed",
        )
        added += 1

    if added:
        logger.info("Seeded %d execution recipe skills", added)
    return added
