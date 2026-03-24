Phase 0: Data Models & Schema
Step 0.1 — Product Data Model
Create src/shopping/models.py. Define dataclasses:

Product: name, original_price, discounted_price, discount_percentage, currency (default "TRY"), url, source, image_url, specs dict, rating, review_count, availability enum (in_stock, low_stock, out_of_stock, preorder), seller_name, seller_rating, seller_review_count, shipping_cost, shipping_time_days, free_shipping bool, installment_info dict, warranty_months, category_path (e.g. "Elektronik > Bilgisayar > RAM Bellek").

Review: text, rating, date, source, author, verified_purchase bool, helpful_count, language.

ProductMatch: list of Product across sources, canonical_name, canonical_specs, confidence_score.

PriceHistoryEntry: price, source, date, was_campaign bool.

ShoppingQuery: raw_query, interpreted_intent, constraints list, budget, category, generated_searches list.

UserConstraint: type enum (dimensional/compatibility/budget/dietary/electrical/temporal), value, hard_or_soft, source (user-stated or system-inferred).

Combo: list of ProductMatch, total_price, compatibility_notes, value_score.

ShoppingSession: session_id, user_query, analyzed_intent, products_found, recommendations_made, user_actions, timestamps.

Step 0.2 — Product Cache Schema
Extend src/infra/db.py or create src/shopping/cache.py. SQLite tables: products (hash of url as key, full product JSON, source, fetched_at, ttl_category), reviews (product_url hash, review JSON, source, fetched_at), price_history (product_url hash, price, source, observed_at), search_cache (query hash + source as key, result JSON, searched_at). Implement TTL-based invalidation: specs 30 days, prices 24 hours, reviews 7 days, search results 12 hours. Every scraping tool checks cache before hitting network.

Step 0.3 — Request Tracking Schema
Create request_log table in cache database. Fields: domain, url, timestamp, status_code, response_time_ms, cache_hit bool, scraper_used, session_id. This feeds the anti-detection monitoring (Phase 10) and rate budget manager. Also create domain_health table: domain, success_count_24h, failure_count_24h, last_success, last_failure, current_status enum (healthy, degraded, blocked). Separate from product cache because this is operational data with different lifecycle — no TTL, rolling window aggregation.

Step 0.4 — Turkish Text Utilities
Create src/shopping/text_utils.py. Functions:

parse_turkish_price(text) -> float handling "1.299,99 TL", "₺1299.99", "1299,99" formats.

normalize_turkish(text) handling İ/ı, Ş/ş, Ç/ç, Ö/ö, Ü/ü, Ğ/ğ for consistent matching.

generate_search_variants(query) producing both Turkish and English search terms (e.g., "bellek" and "RAM", "bulaşık makinesi" and "dishwasher").

extract_dimensions(text) -> dict parsing "60x55x45 cm", "genişlik: 60cm", "W600 x D550 x H450mm" and similar patterns from product descriptions.

extract_weight(text) -> float in kg.

extract_capacity(text) -> dict for appliance capacities ("9 kg yıkama kapasitesi", "500 litre", "12 fincan").

extract_energy_rating(text) -> str parsing Turkish energy labels (A+++, A++, etc.).

extract_material(text) -> str for furniture/cookware ("paslanmaz çelik", "meşe", "mdf").

extract_volume_weight_for_grocery(text) -> dict parsing "1 kg", "500 ml", "6'lı paket", "2.5 L" for unit price calculation.

detect_language(text) -> str since some product titles are English on Turkish sites.

normalize_product_name(text) that strips common filler words from Turkish product titles ("süper fırsat", "kampanyalı", "orjinal ürün", "hızlı kargo", "aynı gün kargo") to get cleaner product names for matching.

Step 0.5 — Configuration Schema
Create src/shopping/config.py and src/shopping/shopping_config.yaml. Central configuration for all shopping settings: per-domain rate limits and daily budgets, cache TTL values per data type, scraper priority order per data need, GitHub Actions toggle (enabled/disabled), API keys reference (which credential store keys to use for Amazon PA-API, Google CSE), default user preferences (language, currency, preferred stores), feature flags for intelligence modules (enable/disable seasonal analysis, substitution suggestions, etc.). Load from YAML for easy editing without code changes.

Phase 1: Scraping Tools
Step 1.1 — Base Scraper Tool
Create src/shopping/scrapers/base.py. Abstract class BaseScraper with: async search(query, max_results), async get_product(url), async get_reviews(url, max_pages). Built-in: cache check before every request, rate limiting per domain (configurable delay, default 10s), random User-Agent rotation from a list, request logging to request_log table (Step 0.3), exponential backoff retry (max 3 attempts, delays of 5s/15s/45s) before marking a request as failed. preflight_check() method that each scraper can override — verifies the domain is currently healthy before attempting requests (checks domain_health table). validate_response(response) -> bool abstract method — each scraper implements its own check for "did I get real content or a block page / CAPTCHA". extract_structured_data(html) -> dict that looks for JSON-LD, Open Graph meta tags, and Schema.org markup before falling back to HTML parsing — many Turkish e-commerce sites embed structured product data in these standard formats, which is more reliable than CSS selectors. request_metrics that logs to the request tracking table from Step 0.3. Include a ScraperRegistry that maps domain names to scraper classes.

Step 1.2 — Akakçe Scraper
Create src/shopping/scrapers/akakce.py. Implements search(): hit akakce.com search, extract product cards (name, price range, store count, product URL). Implements get_product(): hit product detail page, extract price comparison table (store name, price, shipping info, link to store), extract spec table if present. Extract price history data — Akakçe embeds historical price data in a JavaScript variable on product pages (typically something like var defined_chart_data = [...]), extract with regex from script tag, parse the JSON. Also extract: number of stores carrying the product, lowest/highest prices across stores, "fiyat düştü" / "fiyat arttı" badges, "benzer ürünler" (similar products) links as candidates for the alternative generator. Handle Akakçe's category pages — sometimes browsing category listings reveals products that don't show up in search. This is the primary price aggregation source. Low anti-bot protection, simple httpx + BeautifulSoup should work.

Step 1.3 — Trendyol Scraper
Create src/shopping/scrapers/trendyol.py. For search: use the public discovery API endpoint (public.trendyol.com/discovery-web-searchgw-service/...) which returns JSON — no HTML parsing needed. For reviews: use the review API endpoint (public-mdc.trendyol.com/discovery-web-socialgw-service/...) which also returns JSON with review text, rating, date, seller reply. Extract: product name, price (original + discounted), rating, review count, specs from attributes field, seller name and rating, promotions field (active campaigns, coupon codes, "süper fırsat" badge), installmentTable or similar field showing bank-specific installment plans, merchantId and merchantName for seller identity, favoriteCount as a popularity proxy, freeCargo boolean and rushDelivery boolean. Handle pagination for reviews. Add Trendyol category browsing capability — Trendyol has a hierarchical category API that can be used to discover products by category rather than just keyword search, helpful for queries like "best coffee machine" where browsing a category is better than keyword search.

Step 1.4 — Hepsiburada Scraper
Create src/shopping/scrapers/hepsiburada.py. This is the hardest site — heavy JS rendering, aggressive anti-bot. Primary method: look for __NEXT_DATA__ or similar Next.js data hydration blocks — Hepsiburada uses Next.js and often embeds full product data in the initial page load JSON. If the embedded JSON approach works, you get structured data without needing Scrapling or Playwright at all, significantly reducing complexity and detection risk. Also attempt internal API endpoints (inspect network tab for api.hepsiburada.com calls). Fall back to Scrapling if both approaches fail. Product page: extract specs from the structured product info section. Reviews: Hepsiburada loads reviews via separate API calls — find and use those endpoints. Flag this scraper as "high risk" — if it fails, skip gracefully and rely on other sources.

Step 1.5 — Amazon TR Scraper
Create src/shopping/scrapers/amazon_tr.py. Primary method: Amazon PA-API 5.0 (requires affiliate account registration — document this as a prerequisite). PA-API gives: search, product details, pricing, reviews, specs — all structured JSON, no scraping needed, zero ban risk. Note: Amazon PA-API requires making at least 3 qualifying sales through your affiliate link within 30 days of registration or API access is revoked — document this clearly. For price history: integrate Keepa.com which tracks Amazon prices historically and works for amazon.com.tr (limited free tier API). Fallback if PA-API isn't set up: the scraper should use embedded JSON approach (Amazon also embeds product data in script tags), or fall back to Scrapling-based scraper. Store credentials via existing src/security/credential_store.py.

Step 1.6 — Forum Scraper (Technopat + DonanimHaber)
Create src/shopping/scrapers/forums.py. Both are standard forum sites with light protection. For Technopat: search via technopat.net/sosyal/ara/?q=, extract thread titles and URLs from results, for relevant threads extract posts (text, author, date, quoted text). For DonanimHaber: similar approach via forum.donanimhaber.com/ara. These are static HTML, simple httpx + BeautifulSoup. After fetching thread results from search, score each thread for relevance before deep-scraping: thread title match quality (exact product name > category mention > tangential mention), reply count (more replies = more data), last reply date (recent threads more relevant), sub-forum location (Technopat "donanım" section more relevant for hardware than "sohbet" section). Only deep-scrape the top 3 most relevant threads, max 50 posts per thread. Extract "çözüm" (solution) or "tecrübe" (experience) tagged posts specifically — forums often tag helpful answers, and these are higher signal than random discussion posts.

Step 1.7 — Ekşi Sözlük Scraper
Create src/shopping/scrapers/eksisozluk.py. Search: eksisozluk.com/?q=. Extract entry list from topic page: entry text, author (suser nick), date, favorite count. Ekşi rate-limits aggressively — enforce minimum 5s delay. Handle "başlık bulunamadı" (topic not found) gracefully. Search for both product name and product category as separate queries. Also search for the brand as a topic — e.g., "arçelik" başlığı might have relevant entries about the brand's quality even if not about the specific model. Filter by favorite_count (entries with more favorites are community-validated as useful), filter by length (entries under 50 characters are rarely informative for product research), detect and filter sarcasm/joke entries (entries that start with "*" which is Ekşi convention for jokes/asides, or use LLM for ambiguous cases).

Step 1.8 — Şikayetvar Scraper
Create src/shopping/scrapers/sikayetvar.py. Search: sikayetvar.com/arama?q=. Extract complaint cards: title, brand, short text, date, resolution status (solved/unsolved). For detail pages: full complaint text + company response if exists. Calculate resolution rate for the specific brand on that product category (Şikayetvar shows "çözüm oranı" — extract this). Extract response time from the brand. Aggregate: "Brand X has 89% resolution rate with average 2-day response for washing machines, but only 45% resolution for small appliances." Also extract "memnuniyet puanı" (satisfaction score) after resolution — distinguish between complaints resolved happily vs grudgingly. This is specifically for negative sentiment and known issues — treat it as a "warnings" data source, not general reviews.

Step 1.9 — Grocery / FMCG Scrapers
Create src/shopping/scrapers/grocery.py. Multiple sub-scrapers:

AktuelKatalogScraper — sites like aktuelkatalogu.com, haftaninindirim.com, and indirimdekiurunler.com already compile weekly A101/BİM/ŞOK campaigns from their PDF catalogs into searchable web pages. Scrape these instead of the chains directly. Much easier, no anti-bot issues, already structured.

GetirScraper — Getir has a clean internal API (inspect the web app at getir.com, network tab shows REST API calls for product search with prices, availability, and unit pricing). High-value source for instant grocery price comparison.

MigrosVirtualScraper — migros.com.tr virtual market has searchable products with campaign badges and loyalty card prices.

TrendyolMarketScraper — subset of Trendyol for groceries.

All grocery scrapers must: extract product name, unit price, price per kg/L, campaign badge, stock status. Add unit price calculation to every grocery result: if product is "Banvit Tavuk Göğüs 900g — 79.90 TL", calculate "88.78 TL/kg". Essential for comparing across different package sizes and brands. Add campaign detection: identify which prices are temporary campaign prices vs regular prices (campaign items usually have end dates or special badges).

Step 1.10 — Scrapling Integration Layer
Create src/shopping/scrapers/scrapling_client.py. Wrapper around Scrapling library for sites that need it (Hepsiburada, any site that blocks simple requests). Configure: stealth mode, adaptive selectors, Playwright backend. This is used as fallback by individual scrapers when httpx requests fail. Include a method fetch_with_fallback(url) that tries httpx first, falls back to Scrapling, logs which method succeeded for future optimization.

Step 1.11 — Google Custom Search Tool
Create src/shopping/scrapers/google_cse.py. Uses Google Custom Search JSON API (100 free queries/day). Executes site-scoped searches: site:trendyol.com {query}, site:hepsiburada.com {query}. Returns: title, URL, snippet. Use this as a discovery mechanism when direct site search is blocked or unreliable. Also use for general product research queries without site restriction. Store API key via credential store. Cache CSE results for 48 hours (longer than other search caches) because Google results change slowly. Implement a quota tracker that shows remaining daily queries. Add smart query batching: instead of separate searches for site:trendyol.com DDR5 RAM and site:hepsiburada.com DDR5 RAM, consider one unrestricted search and let Google rank across all sites — uses 1 query instead of N. Reserve 20% of daily budget (20 queries) for ad-hoc user requests that can't be anticipated.

Step 1.12 — Sahibinden.com Scraper
Create src/shopping/scrapers/sahibinden.py. Sahibinden is Turkey's dominant classifieds/used market site. Essential for used market intelligence (Step 11.3). Also carries new product listings from small businesses. Search: sahibinden.com/arama?query=. Extract: listing title, price, location, date, condition (new/used/refurbished), seller type (individual/business), photo count (more photos = more trustworthy listing). Sahibinden has moderate anti-bot protection. Rate limit strictly (15+ second delays). Focus on: electronics, furniture, appliances, vehicles and parts. Don't scrape real estate or automotive categories unless specifically asked.

Step 1.13 — Koçtaş / IKEA Scrapers
Create src/shopping/scrapers/home_improvement.py. For furniture and home improvement queries (the cupboard-for-oven use case). KoctasScraper: koctas.com.tr has searchable catalog with detailed dimensions in product specs. IKEAScraper: ikea.com.tr has clean product pages with very structured dimension data. Both are important for dimensional constraint queries. Both have relatively light anti-bot measures. IKEA in particular embeds detailed product specs in structured data markup (Schema.org) — use the base scraper's extract_structured_data() method from Step 1.1.

Phase 2: Remote Execution via GitHub Actions
Step 2.1 — GitHub Actions Workflow Files
Create two workflow variants:

.github/workflows/shopping-search-light.yml — single job, simple search, for quick queries.

.github/workflows/shopping-search-deep.yml — parallel jobs split by source group (one job for e-commerce sites, one for forums/reviews — merge results at the end). The deep variant uses more minutes but runs faster wall-clock because sources are scraped in parallel across separate VMs with separate IPs.

Both triggered via workflow_dispatch with inputs: query (string), search_plan (JSON string containing list of specific scrape tasks), session_id (to correlate results). Job steps: checkout repo, setup Python with actions/cache@v4 for pip cache directory (reduces install time from ~90 seconds to ~10 seconds), install dependencies (scrapling, httpx, beautifulsoup4, playwright), run src/shopping/remote_runner.py with the search plan, upload results.json as artifact. Timeout: 15 minutes per run. Add concurrency group to prevent multiple shopping searches from running simultaneously and burning through minutes.

Step 2.2 — Remote Runner Script
Create src/shopping/remote_runner.py. Standalone script that runs on GitHub Actions. Reads search plan from argument or environment variable. Search plan is a JSON list of tasks: [{"scraper": "trendyol", "action": "search", "query": "DDR5 RAM"}, {"scraper": "trendyol", "action": "get_reviews", "url": "..."}, ...]. Executes tasks sequentially with delays. Writes partial results after each scraper completes, not just at the end — if the job is cancelled or times out, partial results are still available. Maintains a progress.json artifact updated after each scraper task completes, containing: completed tasks, pending tasks, any errors so far. This enables the local system to show progress like "Trendyol done ✅, Hepsiburada scraping... ⏳, Akakçe done ✅". Final output: results.json. This script imports scrapers from Phase 1 but runs independently of the orchestrator.

Step 2.3 — GitHub Actions Dispatcher
Create src/shopping/github_dispatcher.py. Integrates with existing src/integrations/ pattern. Methods: trigger_search(search_plan) -> run_id (POST to GitHub API to trigger workflow), poll_status(run_id) -> status (check if workflow completed), download_results(run_id) -> dict (download artifact and parse JSON). Primary result delivery: runner writes results to a private GitHub Gist (create or update), dispatcher reads the Gist via its URL — faster and simpler than artifacts, no expiry concerns. Fallback: artifact download if Gist creation fails. Uses existing credential store for GitHub token. Include retry logic, timeout handling, and estimated completion time calculation based on search plan size and historical run durations.

Step 2.4 — Local vs Remote Decision Logic
Create src/shopping/execution.py. Decides whether to scrape locally or via GitHub Actions based on: target site risk level (Akakçe/forums = local is fine, Hepsiburada/Amazon = remote), number of requests in plan (>10 = remote), user preference (configurable flag to always use remote), time of day (2-6 AM Turkey time = lower risk for local scraping, less monitoring), session history (if user has already done 3 local searches today across the same domain, switch to remote for the 4th). If user explicitly wants instant results and the search plan is small (<5 requests), prefer local with VPN (ProtonVPN CLI) for speed — GitHub Actions has ~30 second startup overhead. Add a configuration option: always_remote: true for users who never want their home IP exposed. If remote: dispatch via Step 2.3. If local: execute scrapers directly. If mixed: split plan into local and remote batches, merge results.

Phase 3: Knowledge Base
Step 3.1 — Turkish Market Knowledge File
Create src/shopping/knowledge/turkish_market.md. Plain text file loaded into LLM context. Contents:

Turkish shopping calendar: 11.11 Trendyol, Black Friday / Efsane Cuma, Ramazan Bayramı discounts, back-to-school September, seasonal patterns per category (air conditioners — buy March-April, prices peak June-August; heaters — buy May-June, prices peak November; school supplies — buy July, prices peak September).

Grocery campaign cycles: A101 Tuesday/Thursday, BİM Friday, ŞOK Wednesday.

USD/TRY impact rules: imported electronics track dollar, domestic appliances less affected, when TL drops consider domestic brands.

Customs/import duties: products ordered from international sites (AliExpress, Amazon.com global) are subject to customs above €150 threshold.

Installment culture: most Turkish consumers buy electronics and appliances in installments (taksit). Banks offer 2/3/6/9/12 month options, some interest-free, via specific credit cards on specific stores. This massively affects effective pricing — a product that costs 12,000 TL with 12-month interest-free is effectively cheaper than 11,000 TL with no installment option for many Turkish consumers.

General market notes: Philips reducing TR operations, Arçelik/Beko strong local service, Vestel budget domestic option. Update this file manually every few months.

Step 3.2 — Store Profiles Knowledge File
Create src/shopping/knowledge/store_profiles.md. Per-store information including: general description, marketplace vs direct sales, shipping options and speed, return policy specifics (Trendyol: 15 days free return for most items, Hepsiburada: 15 days with HepsiJet free return, Amazon: 30 days generous return), which bank cards offer installment options, typical maximum installment periods, whether they accept Param/Papara/digital wallets, store-specific gotchas (Trendyol "original price" inflation — some sellers inflate the "original" price to show large discount percentages, Hepsiburada "HepsiBurada Premium" items are more reliable than marketplace sellers), physical store availability per chain for users who want to see/touch before buying or need immediate pickup. Per-store: Trendyol, Hepsiburada, Amazon TR, N11, MediaMarkt, Vatanbilgisayar, Koçtaş, IKEA TR, Migros, Getir.

Step 3.3 — Compatibility Knowledge Base
Create src/shopping/knowledge/compatibility/. Multiple small files:

cpu_sockets.json: socket types to compatible CPU families (AM4 → Ryzen 1000-5000, AM5 → Ryzen 7000-9000, LGA1700 → 12th-14th gen Intel, LGA1851 → Core Ultra 200).

ram_compatibility.json: chipsets to DDR generation and max speeds.

appliance_standards.json: Turkish standard kitchen countertop height 85cm, standard base cabinet width 60cm, standard built-in oven cavity 56-60cm, standard dishwasher width 60cm (full) or 45cm (slim), standard washing machine width 60cm, standard fridge width 60cm (single) or 90cm (side-by-side).

electrical_standards.json: Turkey uses 220V 50Hz, Type F (Schuko) plugs, common circuit breaker ratings for kitchen/bathroom, maximum wattage considerations for older buildings.

plumbing_standards.json: standard Turkish tap connection sizes for washing machine and dishwasher hoses.

common_dimensions.json: standard countertop height 85cm, standard interior door width 80cm, entrance door 90cm.

These are small, stable datasets. LLM reads relevant file based on product category.

Step 3.4 — Category Substitution Map
Create src/shopping/knowledge/substitutions.json. Intent-based structure rather than flat category mapping: {"intent": "protein_for_cooking", "primary": "chicken_breast", "substitutes": [{"item": "turkey_breast", "suitability": "most_recipes", "exceptions": "strong chicken flavor needed"}, {"item": "thigh_meat", "suitability": "stews_and_casseroles", "exceptions": "lean_diet"}, {"item": "fish_fillet", "suitability": "generic_protein_need", "exceptions": "specific_meat_recipe"}]}. Include similar mappings for other categories (cooking fats, hot beverage machines, seating, storage furniture, etc.). Include anti_substitutions — things that seem related but should NOT be suggested as substitutes: {"regular_flour": "NOT_SUBSTITUTE_FOR": "gluten_free_flour"}, {"cow_milk": "NOT_SUBSTITUTE_FOR": "lactose_free_milk_if_intolerant"}. Include substitution_boundaries: {"never_substitute": ["prescription_medicine", "baby_formula", "specific_car_parts_by_VIN"]}. The LLM uses this to trigger lateral thinking, not as a rigid rule system.

Step 3.5 — Turkish Search Terms Dictionary
Create src/shopping/knowledge/search_terms.json. Maps common product concepts to Turkish search terms and their variants: {"RAM": ["ram bellek", "bellek", "memory"], "motherboard": ["anakart", "ana kart"], "washing machine": ["çamaşır makinesi", "çamaşır makinası"], ...}. Include common misspellings and colloquial terms: "bulaşık makinası" (common misspelling of "makinesi"), "ekran kartı" and "grafik kartı" (both mean GPU), "kasa" (can mean PC case or cash register — need disambiguation), "hoparlör" vs "speaker" vs "bluetooth hoparlör". Include brand name variants and measurement term variants ("cm" vs "santim" vs "santimetre"). Used by the query decomposition step to generate better search terms. LLM can also generate terms on-the-fly, but this provides a reliable baseline.

Step 3.6 — Installment Database
Create src/shopping/knowledge/installments.json. Map of: which stores partner with which banks for installment plans, typical installment tiers (2/3/6/9/12 months), which tiers are typically interest-free, which product categories are eligible (electronics almost always, groceries rarely). The LLM uses it as baseline knowledge, and actual installment availability is scraped from product pages when available. Having the baseline means the LLM can say "Trendyol typically offers 9-month interest-free on electronics with X bank card" even when the specific product page installment data isn't available.

Step 3.7 — Brand Service Network Map
Create src/shopping/knowledge/brand_service.md. For major appliance and electronics brands: number of authorized service centers in Turkey, geographic coverage (Istanbul/Ankara/Izmir only vs nationwide), reputation for warranty handling (from Şikayetvar aggregate data). Organized by brand: "Arçelik/Beko: 700+ service points, nationwide, excellent warranty reputation. Dyson: limited to major cities, expensive out-of-warranty repairs, parts availability can be slow. Xiaomi: growing service network but still limited outside major cities." Updated manually from Şikayetvar data analysis. Directly feeds into recommendations.

Phase 4: Intelligence Modules
Step 4.1 — Query Analyzer
Create src/shopping/intelligence/query_analyzer.py. Takes raw user query, sends to local LLM with a structured prompt. LLM returns JSON:

text
{
  "intent": "buy_product | compare_products | find_deal | research | upgrade_existing | gift | explore",
  "category": "electronics | appliances | grocery | furniture | ...",
  "products_mentioned": [...],
  "constraints": {"budget": ..., "dimensions": ..., "compatibility_with": ..., "use_case": ...},
  "missing_info": ["budget not specified", "don't know DDR generation needed"],
  "follow_up_questions": ["What's your motherboard model?", "What's your budget range?"],
  "urgency": "urgent | researching | no_rush",
  "experience_level": "expert | intermediate | novice",
  "explicitness": "specific_product | general_category | vague_intent",
  "multi_product": true/false,
  "gift_intent": true/false,
  "search_complexity": "simple | medium | complex"
}
This is the first step in every shopping query — determines what the LLM needs to know and do. Search complexity routes to the appropriate workflow.

Step 4.2 — Search Plan Generator
Create src/shopping/intelligence/search_planner.py. Takes analyzed query from Step 4.1, generates a list of concrete search tasks. Prompt the LLM with the analyzed intent to produce searches like:

text
[
  {"query": "DDR5 32GB RAM", "sources": ["akakce", "trendyol"], "purpose": "primary_search", "phase": 1},
  {"query": "2x16GB DDR5 kit", "sources": ["akakce"], "purpose": "alternative_config", "phase": 1},
  {"query": "DDR5 vs DDR4 2024", "sources": ["perplexica"], "purpose": "market_context", "phase": 1},
  {"query": "DDR5 RAM sorun", "sources": ["sikayetvar"], "purpose": "warnings", "phase": 1},
  {"query": "[top 3 product reviews]", "sources": ["trendyol", "amazon"], "purpose": "deep_reviews", "phase": 2}
]
Phase 1 searches can run immediately; Phase 2 searches depend on Phase 1 results. Include plan_justification field for each search — why this search is included and what gap it fills. The plan must be aware of remaining daily quotas for each source (from rate budget manager). If Google CSE has 15 remaining queries today, don't plan 20 CSE queries. Prioritize higher-value sources first. Limit total searches to max 15-20 per session.

Step 4.3 — Alternative Generator
Create src/shopping/intelligence/alternatives.py. Given a product query, generates alternative search queries that might yield better value. Two modes: rule-based (for known patterns like "if X GB RAM, also search 2×(X/2) GB") and LLM-based (for open-ended alternatives). Feed the LLM: product name, category, any known constraints, substitution map from Step 3.4. LLM returns list of alternatives with reasoning: [{"query": "2x16GB DDR5", "reason": "Dual channel kits often cheaper than single stick"}, {"query": "DDR4 32GB", "reason": "Check if DDR4/DDR5 price gap justifies upgrade"}]. Each alternative becomes an additional search in the search plan.

Step 4.4 — Substitution Engine
Create src/shopping/intelligence/substitution.py. Given the product category and intent, suggests substitute products from adjacent categories. Uses substitution map (Step 3.4) as seed, LLM for reasoning. Key logic: determine the user's underlying NEED (not just the product). "Buying chicken for dinner" → need is "protein for dinner" → turkey, fish are substitutes. "Buying chicken for tavuk sote recipe" → need is "specific ingredient" → fewer substitutes. For food substitutions: the LLM considers nutritional similarity (protein content, calories, allergens) and briefly notes nutritional differences. For non-food: ranks substitutes by functional closeness, not just category membership. Include price-triggered mode: after initial search, if a substitute is found significantly cheaper (>20% less), flag it as a suggestion. Track user_has_rejected — if user rejected a substitution previously (from session memory), don't suggest it again in the same session.

Step 4.5 — Constraint Checker
Create src/shopping/intelligence/constraints.py. Given products and user constraints, filters and validates. Types of constraints:

Dimensional: product must fit in X space — compare extracted dimensions including tolerances. Include not just "does it fit" but installation constraints: delivery path (can a 200cm sofa fit through a standard Turkish apartment door — typical interior door is 80cm wide), ventilation clearance for heat-generating appliances (ovens need 50mm rear, 5mm sides; fridges need 50mm+ on top and back), door swing radius (fridge in a narrow kitchen).

Compatibility: does it work with what they already have — check against compatibility knowledge base.

Electrical: voltage, wattage, plug type — Turkey: 220V, Type F plug.

Budget: hard limit or soft preference.

Availability: in stock, ships to user's city.

Temporal: is now a good time to buy.

The checker should ASK the user for missing critical constraints rather than silently ignoring them. If user says "cupboard for my oven" and hasn't provided kitchen dimensions, generate a follow-up question. Return filtered product list with compatibility notes per product.

Step 4.6 — Value Scorer
Create src/shopping/intelligence/value_scorer.py. Given a set of products with prices and specs, calculates a value score. For quantifiable categories: price per unit of performance (price per GB, price per MHz, etc.). For non-quantifiable categories: use review rating × review count as quality proxy, calculate quality-per-lira. Factor in: seller reputation from store profiles, shipping cost if available, warranty length. Total cost of ownership: for appliances include energy cost over expected lifetime (from Step 11.5), for products with consumables include consumable costs (Nespresso machine is cheap but capsules are expensive). For installment purchases: calculate real cost including interest if applicable — most Turkish installment plans technically have interest built into the price even when labeled "interest-free", as the cash price is sometimes lower. Output multiple value perspectives: best immediate price, best TCO (1 year / 3 year / 5 year), best on installment plan. Normalize scores to 0-100 range. LLM provides the "what metric matters" reasoning per category.

Step 4.7 — Market Timing Advisor
Create src/shopping/intelligence/timing.py. Given a product category and current date, advises whether now is a good time to buy. Sources: Turkish market calendar from Step 3.1 (is a sale event coming soon?), price history from Akakçe (is the product trending up or down?), product lifecycle (LLM knowledge — is a new generation about to launch?, supplemented by upcoming_launches.json that you update when major product announcements happen), currency trends (use a free exchange rate API like exchangerate.host to get current USD/TRY rate — if TRY has weakened >5% in the last 30 days, warn that import electronics prices are likely to increase, buy now if price matches last month's rates; if TRY has strengthened, prices may drop slightly). Output: {"recommendation": "wait | buy_now | neutral", "reason": "Black Friday is in 3 weeks, this category typically drops 15-25%", "confidence": 0.7}. Advisory — always included in final output but never blocks a purchase decision.

Step 4.8 — Combo Builder
Create src/shopping/intelligence/combo_builder.py. For multi-component needs (PC upgrade, kitchen setup, etc.). Takes: list of component types needed, budget, compatibility constraints. Process: search for each component type independently, build compatibility graph (which components work together), generate N feasible combinations within budget, score each combination (total value, performance tier, future upgrade potential). LLM does heavy lifting: "Given these 5 CPUs, 4 motherboards, and 6 RAM kits, generate 3 recommended combinations at budget/mid/premium tiers with reasoning."

Factor in cross-store logistics: shipping costs per store (some have free shipping thresholds — buying more from one store might cross the threshold), delivery time alignment (if CPU comes from Trendyol in 1 day but motherboard from Amazon takes 5 days, suggest same-store alternatives if price difference is small), return risk aggregation (buying from 3 stores means 3 different return processes). Include a "convenience score" alongside value score — single-store combo at 5% more might be worth it. Detect when a prebuilt product exists that replaces the combo — e.g., a prebuilt mini PC or barebone kit that includes all three components at competitive total price.

Output per combo: component list with prices and sources, total price, compatibility confirmation, pros/cons, upgrade path notes, convenience score.

Step 4.9 — Review Synthesizer
Create src/shopping/intelligence/review_synthesizer.py. Takes raw reviews from multiple sources for a single product. Sends to local LLM with structured prompt. Apply temporal weighting: reviews from last 6 months at 2x weight vs older reviews. Volume awareness: a product with 3 five-star reviews is much less reliable than 3,000 at 4.2 stars — calculate confidence-adjusted rating. Cross-source sentiment comparison: if Trendyol reviews are 4.8★ but Şikayetvar has 50 complaints, that's a red flag — detect and highlight discrepancies. Extract specific product defect patterns from negative reviews: "screen bleed" mentioned in 15% of negative reviews is more actionable than generic "bad product" in 30% — LLM categorizes negative reviews into specific issue types and reports frequency. Estimate fake review percentage (suspiciously similar wording, all 5-star with no detail). Note Turkey-specific concerns (warranty experience, shipping damage reports, voltage issues).

Output:

text
{
  "overall_sentiment": 0.78,
  "confidence_adjusted_rating": 4.3,
  "positive_themes": ["fast shipping", "good build quality"],
  "negative_themes": ["noisy fan", "poor packaging"],
  "defect_patterns": {"screen_bleed": "15%", "DOA": "3%"},
  "warnings": ["3 reports of DOA units from same Trendyol seller"],
  "turkey_specific": ["Arçelik service center handles warranty locally — positive"],
  "review_quality": "mixed — 30% genuine detailed, rest shallow",
  "cross_source_discrepancy": false
}
Step 4.10 — Product Matcher
Create src/shopping/intelligence/product_matcher.py. Given product listings from multiple sources, determines which listings refer to the same physical product. Matching hierarchy: exact match on EAN/UPC/barcode if available (most reliable), match on MPN/manufacturer part number (very reliable), match on model name with fuzzy matching using normalized text (good — use thefuzz library), match on spec fingerprint — same brand + same key specs = likely same product (fallback).

Handle variants: same product often comes in color variants (cosmetic difference), size/capacity variants (different product for comparison), bundle variants (product + accessory). Group variants under same parent product, mark which attribute differs. Use model number as primary key for variant grouping — "Kingston Fury Beast DDR5-6000 KF560C36BBK2-32" where the last segment often encodes variant info.

When matching fails entirely (confidence below 0.6): still include the product but flag as "unmatched — verify manually" rather than silently dropping it. Output: grouped ProductMatch objects where each group contains all listings for one physical product across sources, with confidence score per match.

Step 4.11 — Installment Calculator
Create src/shopping/intelligence/installment_calculator.py. Given a product price and store, calculate installment options. Use: scraped installment data from product pages if available, knowledge base installment data (Step 3.6) as fallback. Calculate: monthly payment for each installment tier, total amount paid (including interest if any), effective annual cost for comparison. When comparing products across stores: include installment-adjusted comparison — "Product A is 500 TL more on Store X, but Store X offers 12-month interest-free while Store Y only offers 3-month. Monthly payment: Store X = 1,000 TL/mo, Store Y = 3,500 TL/mo." Flag credit card requirements per installment plan (some plans only available with specific bank cards).

Step 4.12 — Delivery Comparison
Create src/shopping/intelligence/delivery_compare.py. Compare delivery options across sources for the same product. Extract from scraped data: estimated delivery date, shipping cost, express/priority option availability, cargo company (Yurtiçi Kargo, Aras Kargo, MNG, Sürat, Trendyol Express, HepsiJet — Turkish consumers have preferences and experiences with specific carriers). Factor in user's location if known (from user profile): some stores have better delivery times to certain regions. Flag "Ships from abroad" products that may take 15-30 days and are subject to customs. Calculate effective price including shipping for fair comparison.

Step 4.13 — Return Policy Analyzer
Create src/shopping/intelligence/return_analyzer.py. For each recommended product+store combination: extract or look up return policy (from store profiles knowledge base). Highlight: return window (15 days standard in Turkey, some stores offer 30), free return shipping availability, conditions (sealed products, hygiene products, custom-made items — non-returnable), refund method (original payment method vs store credit). Present as "Easy return ✅" or "Difficult return ⚠️" badge per recommendation.

Phase 5: Shopping Agents
Step 5.1 — Shopping Advisor Agent
Create src/agents/shopping_advisor.py. Inherits from base.py. This is the main conversational agent for shopping queries. System prompt includes: Turkish market knowledge (Step 3.1), store profiles (Step 3.2), the universal reasoning framework (clarify → decompose → expand → constrain → search → analyze → present). Capabilities: interprets user shopping intent, asks clarifying questions when needed, maintains conversational state (user's current setup, budget, preferences), delegates to search/analysis sub-steps, presents results with reasoning.

Conversational strategy: when to ask questions vs when to just search (if user gives enough detail, don't ask unnecessary clarifying questions — respect their time), present 3-4 options (not 20), only proactively suggest things the user didn't ask about when the suggestion is significantly valuable. Tone: helpful and knowledgeable but not overwhelming, use Turkish product terminology naturally when conversation is in Turkish, don't hedge excessively — just say "Buy this one because...". Include a conversation_strategy method that adapts based on user's detected experience level: expert users get data-dense tables, novice users get guided recommendations with explanations.

Step 5.2 — Product Researcher Agent
Create src/agents/product_researcher.py. Inherits from base.py. Executes search plans generated by the intelligence layer. Takes a search plan (list of search tasks), dispatches to appropriate scrapers (local or via GitHub Actions), collects results, runs product matching (Step 4.10), deduplicates and merges data. Returns structured data to the shopping advisor.

Adaptive during execution: if early results reveal that the product is only available on 2 sources instead of expected 5, expand search to alternative terms. If results suggest the user's intended product doesn't exist as described (discontinued), detect this early and report back to the advisor agent rather than continuing to search fruitlessly. If one source returns unexpectedly many results, sample rather than exhaustively scraping. Maintain a running "confidence score" — do we have enough data to make a recommendation, or do we need more searches? Handles failures gracefully: if one source fails, continues with others and notes the gap.

Step 5.3 — Deal Analyst Agent
Create src/agents/deal_analyst.py. Inherits from base.py. Specialized in evaluating value and deals. Takes product data from researcher, applies: value scoring (Step 4.6), market timing analysis (Step 4.7), substitution suggestions (Step 4.4), price history analysis if available. Identifies: best value option, best quality option, best deal (biggest discount from normal price), hidden gems (good product with few reviews = lower price), red flags (too cheap, suspicious seller, known issues).

Deal authenticity detection: Turkish e-commerce has a significant fake discount problem. The analyst should: compare the "discounted" price against Akakçe price history to verify the discount is real, flag products where the "original price" has never actually been the selling price, flag discount percentages above 50% for non-fashion items as likely inflated, compare sale price against 3-month average price — if the "sale" price equals the average, it's not really a sale.

Step 5.4 — Clarification Agent
Create src/agents/shopping_clarifier.py. Specialized agent for the clarification dialog phase. Determines the minimum set of questions to ask (don't ask 10 questions — find the 2-3 that matter most), offers smart defaults ("What's your budget?" → provide range buttons like "<2000 TL | 2000-5000 TL | 5000-10000 TL | 10000+ TL" instead of open-ended), detects implied constraints from context (if user profile says they have a small apartment, infer space constraints for furniture without asking). Handles the "I don't know what I need" case — guides the user from vague intent ("I want to make coffee at home") to searchable specifics through a short decision tree, driven by LLM not hard-coded.

Phase 6: Shopping Workflows
Step 6.1 — Main Shopping Workflow
Create src/workflows/shopping/shopping.json. Follows existing workflow JSON pattern. Steps: analyze_query (Step 4.1) → check_clarification_needed (conditional: if missing critical info, use clarifier agent and loop) → generate_search_plan (Step 4.2) → execute_searches (Step 5.2 agent, local or remote) → match_products (Step 4.10) → check_constraints (Step 4.5, conditional: only if constraints exist) → deep_dive_reviews (only for top candidates, max 5 products) → analyze_value (Step 5.3 agent) → generate_alternatives (Step 4.3, feeds back into search if promising alternatives found — limit to one expansion loop) → synthesize_results (Step 5.1 agent compiles final output) → present_to_user.

Support re-entry and iterative refinement: after presenting results, offer "refine search" (re-filter without re-scraping), "search more sources" (expand), "compare specific items" (user picks 2-3 for deep comparison), "buy later — watch price" (transition to price watch workflow). Support loops: analyze → search → present → user feedback → refine → present again (maximum 3 refinement loops). If user comes back to a shopping topic within 24 hours, resume from cached results — "I found these results yesterday, want me to update prices or continue from where we left off?"

Include quality gates: minimum 2 sources for price comparison, minimum 1 review source, abort gracefully if all scrapers fail.

Step 6.2 — Quick Search Sub-Workflow
Create src/workflows/shopping/quick_search.json. Lightweight version for simple queries. Pipeline: query → Perplexica search → local LLM summarizes with prices → done. No scraping at all. Perplexica already does web search and returns sourced data — for a simple "how much is X" question, this is often sufficient. Target completion: <30 seconds locally. Only escalate to full workflow if user wants deeper analysis or Perplexica results are insufficient.

Step 6.3 — Combo Research Sub-Workflow
Create src/workflows/shopping/combo_research.json. For multi-component queries. Steps: analyze full need → decompose into components → load compatibility constraints → search for each component independently → build combos (Step 4.8) → price each combo across sources → evaluate combos → present tiered recommendations. Longer running, more searches, but the most valuable for complex purchase decisions.

Step 6.4 — Price Watch Workflow
Create src/workflows/shopping/price_watch.json. Triggered when user says "tell me when X drops below Y TL" or accepts the price watch offer at end of shopping session. Steps: validate product and target price, store in price watch store (Step 7.2), set up scheduled check (lightweight GitHub Action that runs daily, checks watched products, minimal requests per product — just hit Akakçe for current price), compare against target, notify via Telegram if target met. Include automatic expiry after 90 days if target never reached (with notification: "price watch for X expired, lowest seen was Y TL — want to extend or adjust target?").

Step 6.5 — Gift Recommendation Workflow
Create src/workflows/shopping/gift_recommendation.json. Specialized flow for "I need a gift for X" queries. Steps: clarify recipient (relationship, age, interests), clarify budget, clarify occasion (birthday, holiday, no occasion), generate category suggestions (LLM excels here — "for a 30-year-old who likes cooking: premium olive oil set, spice collection, quality knife, kitchen gadget"), search for specific products in suggested categories, apply gift-specific criteria (brand prestige, packaging quality, uniqueness, deliverability — can it be shipped directly to recipient with gift wrapping?), present options organized by category.

Step 6.6 — Exploration Workflow
Create src/workflows/shopping/exploration.json. For "I don't know what I need" queries — "I want to make coffee at home", "I want to improve my home office", "I want to start running." Steps: understand user's goal through guided conversation (clarifier agent Step 5.4), generate a "shopping map" of everything they might need (LLM: "For home coffee, you'll need: a machine, grinder, beans, cups/mugs, optionally a milk frother, and a storage solution"), prioritize (what to buy first vs what can wait), search for the priority items, present as a phased plan rather than a flat product list.

Phase 7: Memory Extensions
Step 7.1 — User Profile Store
Create src/shopping/memory/user_profile.py. Persistent storage (extending existing memory system) for: owned items (user's PC specs, kitchen appliances, car model), dimensional constraints (kitchen dimensions, room sizes if provided), preferences (preferred stores, avoided brands, budget tendencies), dietary restrictions/allergies (for grocery suggestions — user mentioned they're lactose intolerant once, remember forever, never suggest dairy substitutes), household composition hints (single vs family affects quantity recommendations, appliance sizes), location if voluntarily provided (affects delivery time estimates, available stores, local service centers).

Track purchase-adjacent behavior as inferred preferences: user always picks cheapest option → budget-conscious, user always asks about reviews → quality-conscious, user ignored substitution suggestions twice → stop suggesting substitutions. Store inferred preferences separately from stated ones, with lower confidence.

All preference data should be deletable by user command ("forget everything about me" or "forget my PC specs"). Stored as structured JSON per user. Updated conversationally — when user mentions "I have a B550 motherboard", persist it. Loaded into LLM context for every shopping query.

Step 7.2 — Price Watch Store
Create src/shopping/memory/price_watch.py. Stores products the user is interested in but hasn't bought yet. Fields: product name, target price, current best price, source, last checked, alert threshold, current price at time of watch creation, historical low (from Akakçe), watch creation date.

Smart target price suggestions: when user wants to watch a price but doesn't specify a target, analyze Akakçe price history, calculate reasonable target ("this product was 8,500 TL last Black Friday, currently 10,200 TL — suggest watching for 9,000 TL"). Support relative target ("notify me if it drops 15% from current price") and "any discount" mode (notify on any price drop, useful for products that rarely go on sale).

Checked periodically via scheduled GitHub Action. Integrates with existing notification system (src/infra/notifications.py).

Step 7.3 — Shopping Session Context
Create src/shopping/memory/session.py. Maintains context within a multi-turn shopping conversation. Tracks: current search topic, products discussed so far, user's stated and inferred preferences during this session, questions asked and answered, shortlisted products. This is within-conversation state — cleared when shopping topic changes. Persisted to long-term memory only if user confirms information (e.g., confirming they actually own that motherboard).

Step 7.4 — Purchase History
Create src/shopping/memory/purchase_history.py. When user confirms they bought something (detected from conversation: "I bought the Kingston one from Trendyol"), log: product name, approximate price, store, date. Uses: never recommend a product the user already owns (unless consumable), build preference model over time, enable follow-up intelligence ("you bought a B650 motherboard 3 months ago — here are compatible CPU upgrades on sale"), trigger complementary product suggestions at appropriate times ("you bought an espresso machine 2 weeks ago — you might want a quality grinder next"). Never auto-log — only when user explicitly confirms a purchase.

Phase 8: Output Formatting
Step 8.1 — Comparison Table Formatter
Create src/shopping/output/formatters.py. Takes matched products and generates formatted output: markdown comparison table (product name, price per source, rating, key specs), sorted by value score or price. Handles variable column counts (not all products found on all sources). Includes: clickable direct links to product pages (formatted per output channel), installment options column when relevant ("12 ay × 833 TL"), seller name for marketplace products (not just "Trendyol" but "Trendyol — TechSeller Store ⭐4.7"), visual indicators (🏆 best value, ⚠️ warning, 🔥 deal, 💡 suggestion), price history summary ("📉 3-month low" / "📈 price rising"). Accept an output_format parameter: Telegram markdown (character limits, limited formatting), terminal (rich tables via rich library), API JSON (fully structured).

Step 8.2 — Recommendation Summary Template
Create src/shopping/output/summary.py. Templates for the final LLM output. Structured sections: "Top Pick" (best overall recommendation with reasoning), "Best Budget Option" (if different from top pick), "Alternatives Considered" (brief list of what was evaluated and why not chosen), "Lateral Suggestions" (substitutes found, if any), "Warnings" (from Şikayetvar, bad reviews, compatibility issues), "Market Timing" (should you wait?), "Where to Buy" (specific store + link for the recommended product).

Actionable next steps at the end: "🛒 Buy now: [direct link]", "⏰ Wait: [reason and when to check back]", "👀 Watch price: [offer to set up price watch]", "🔍 Compare more: [offer to search additional sources]", "❓ Ask me: [suggest follow-up questions]".

Adapt to query complexity: simple queries get a quick 3-line recommendation, complex queries get full structured sections. Include confidence indicator: "Found consistent data across 4 sources" vs "Limited data — only 1 source, take with caution."

Step 8.3 — Visual Product Cards
Create src/shopping/output/product_cards.py. For Telegram output: generate visual product cards using Telegram's native formatting. Each card: product image (as photo message with caption), product name (bold), price with discount indication (strikethrough original + bold sale price), star rating with count, source and seller, one-line review summary, and inline keyboard buttons ("🔗 Open Link", "👀 Watch Price", "📊 Compare"). Batch multiple cards into a Telegram media group when showing comparison results.

Phase 9: Integration Points
Step 9.1 — Task Classifier Extension
Extend src/core/task_classifier.py. Add shopping intent detection with confidence scoring. Don't just binary classify — queries like "how much RAM does Chrome need?" are NOT shopping queries despite containing "RAM." Use conversation context: if the last 3 messages were about shopping, a vague message like "what about the blue one?" should be classified as shopping continuation. Detect: price inquiries ("how much is...", "fiyatı ne", "en ucuz"), comparison requests ("X vs Y", "karşılaştır"), purchase advice ("should I buy", "almak istiyorum"), deal hunting ("indirim", "kampanya"), upgrade requests ("upgrade etmek istiyorum"), compatibility checks ("uyumlu mu", "sığar mı"). Add shopping sub-intent classification directly: price_check, compare, purchase_advice, deal_hunt, research, upgrade, gift, exploration, complaint_return_help. Use keyword matching for speed and LLM classification for ambiguous cases. Sub-intent determines workflow routing.

Step 9.2 — Router Extension
Extend src/core/router.py. Add routing rules: when task_classifier returns shopping intent, route to shopping workflow (Step 6.1), quick search (Step 6.2), combo research (Step 6.3), gift recommendation (Step 6.5), or exploration (Step 6.6) based on classified sub-intent. Pass conversation history to the shopping advisor agent so it has context from prior non-shopping turns.

Step 9.3 — Orchestrator Hooks
Extend src/core/orchestrator.py. Register shopping workflow with the orchestrator's workflow engine. Add hooks:

on_shopping_query_start: log, initialize session context.

on_search_plan_generated: deduct planned requests from rate budget before execution (prevents over-committing quota), decision point for local vs remote execution.

on_results_received: merge results into session, check if enough data or need more searches.

on_shopping_complete: persist relevant info to user profile, update price history cache, offer to set price watch.

on_product_found_in_different_context: if the user is doing a non-shopping task and the system encounters a product mention, subtly note it for potential shopping relevance without interrupting ("by the way, the product you were looking at last week dropped in price").

Ensure the shopping flow respects existing orchestrator patterns: state machine transitions, error recovery, progress reporting.

Step 9.4 — Telegram Bot Commands
Extend src/app/telegram_bot.py. Add shopping-specific interactions:

Commands: /price <product> (instant price check via quick search), /watch <product> [target_price] (direct price watch setup), /deals (show currently tracked deals and price drops), /mystuff (show saved user profile), /compare <product1> vs <product2> (direct comparison).

UI: inline keyboard for clarifying questions (buttons for "Espresso", "Türk Kahvesi", "Filtre" when asked about coffee type), progress messages during search ("🔍 Akakçe'de arıyorum...", "📊 Trendyol yorumlarını okuyorum..."), formatted product cards with image thumbnails (Step 8.3), "watch this price" button on product results, "search more" button to expand results.

Shopping result message reactions: 👍 logs as positive feedback, 👎 logs as negative feedback. These passively train quality metrics.

Handle long responses by splitting into multiple messages with logical breaks.

Step 9.5 — Perplexica Integration
Create src/shopping/integrations/perplexica.py. Interface with running Perplexica instance for broad web research. Use cases: market research queries ("best espresso machine 2024 Turkey"), opinion gathering ("DDR5 worth it in 2024"), campaign discovery ("A101 bu hafta kampanyaları"), product lifecycle research ("when is RTX 5070 launching"), general questions scrapers can't answer.

Call Perplexica's local API endpoint. Parse response — Perplexica returns sourced text ideal for feeding into local LLM. Structure queries to maximize usefulness — instead of "DDR5 RAM", use "DDR5 32GB RAM fiyat karşılaştırma 2024 Türkiye" to get more relevant, recent, localized results.

Result quality assessment: extract and validate any prices mentioned (current? plausible?), identify source reliability (Technopat > random blogs), detect outdated information (article from 2022 about "current prices" is stale). Tag results with freshness and reliability score.

Step 9.6 — Notification Integration
Extend src/infra/notifications.py. Add shopping-specific notification types: price drop alert (from price watch), deal alert (if a watched category has unusual discounts detected during any search), back-in-stock alert (if previously unavailable product becomes available), price increase warning ("product you were watching went up 15%"). Deliver via existing Telegram bot. Include snooze/dismiss functionality.

Step 9.7 — Existing Researcher Agent Integration
Connect to existing src/agents/researcher.py. The shopping product researcher (Step 5.2) should extend or compose with the existing researcher rather than duplicating research capabilities. For shopping queries involving general research ("is OLED or QLED better for a bright room?"), delegate to existing researcher agent with a shopping context wrapper. For pure product data fetching (prices, reviews, specs), use shopping-specific product researcher.

Step 9.8 — Web Tools Integration
Connect to existing src/tools/web_search.py and src/tools/web_extract.py. These existing tools should be available to shopping scrapers as fallbacks. If a dedicated scraper fails and Perplexica doesn't have the answer, the generic web search + extract pipeline can get product data from any site without a dedicated scraper. This is the "long tail" solution — you can't build scrapers for every small Turkish store, but generic web extraction can get basic product info from most sites. Register as tools available to the shopping advisor agent via existing tool infrastructure.

Phase 10: Resilience & Edge Cases
Step 10.1 — Graceful Degradation Strategy
Create src/shopping/resilience/degradation.py. Define fallback chains for every data need: pricing (Akakçe → Google Custom Search → Perplexica → ask user to provide URL), reviews (Amazon PA-API → Trendyol API → Ekşi/Technopat → Perplexica → "no reviews found, here's what I know from general knowledge"), specs (Akakçe spec table → Amazon PA-API → manufacturer site via Perplexica → LLM inference from product name).

Tiered communication to user: if 1 source fails out of 5, silently proceed and mention as footnote ("⚠️ Hepsiburada verileri alınamadı"). If 2-3 sources fail, mention prominently ("Sonuçlar sınırlı — sadece Trendyol ve Akakçe verileri mevcut"). If all sources fail, suggest alternatives ("Şu an sitelere ulaşamıyorum. Perplexica üzerinden genel bir araştırma yapabilirim veya daha sonra tekrar deneyebiliriz."). Never present incomplete data as if it's complete.

Step 10.2 — Anti-Detection Monitoring
Create src/shopping/resilience/detection_monitor.py. Track per-domain: success rate over last 24 hours, last successful request timestamp, consecutive failure count, types of failures (403, 429, CAPTCHA, timeout, empty response). If a domain's success rate drops below 70%: automatically switch to remote execution for that domain. If below 30%: mark domain as blocked, skip entirely, notify user.

Automatic cooldown: when a domain enters "degraded" state, don't attempt any requests for 1 hour. After cooldown: send a single test request. If succeeds, gradually resume (2 requests, then 5, then normal). If fails, extend cooldown to 4 hours. Log cooldown events for analysis — if a domain consistently enters cooldown at certain times, adjust scraping schedule. Expose metrics via existing src/infra/metrics.py.

Step 10.3 — Response Validation (continued)
Cross-source validation: if same product shows as 5,000 TL on Trendyol but 50,000 TL on Hepsiburada, one of them is wrong. When multiple sources provide prices for the same matched product: calculate median price, flag any price that deviates more than 40% from median as suspicious, prefer the price that's closer to Akakçe's listed range. Don't silently drop suspicious prices — include them but mark them: "⚠️ Hepsiburada fiyatı diğer kaynaklardan önemli ölçüde farklı — doğrulamanız önerilir." Log all rejected data for scraper debugging.

Step 10.4 — Turkish Encoding Edge Cases
Add to src/shopping/text_utils.py. Handle: mixed encoding in responses (some sites serve ISO-8859-9, others UTF-8, some have mixed), HTML entities in product names ("&" in titles, "'" for apostrophe), inconsistent Turkish character usage in specs (some sellers write "BELLEK" instead of "Bellek", "ISLEMCI" instead of "İşlemci" — normalize all to proper Turkish), price formats embedded in HTML attributes vs text content (some sites put price in data-price attribute as integer cents, display formatted version in text). Test all scrapers against real pages with Turkish characters to catch encoding issues early.

Step 10.5 — Rate Limit Budget Manager
Create src/shopping/resilience/rate_budget.py. Global rate limit budget across all scrapers. Configure per-domain daily request limits: Akakçe (200/day), Trendyol (100/day), Hepsiburada (50/day), Amazon PA-API (unlimited but self-throttle to 1/sec), Ekşi Sözlük (50/day), forums (100/day each), Google CSE (100/day hard limit from Google). Track usage in SQLite. When budget is exhausted for a domain: fall back to cached data even if stale, or skip that source. Reset daily. Alert if a single search session would consume more than 30% of any domain's daily budget.

Predictive budget allocation: at start of each day, estimate expected usage based on recent patterns. Before executing a search plan: simulate the budget impact and warn if the plan would exhaust more than 50% of remaining daily budget for any source — give user option to proceed with fewer sources or defer. Weekly budget view for user: "This week you used 340 of 700 Akakçe requests. Budget is healthy."

Step 10.6 — Stale Data Detection
Create src/shopping/resilience/staleness.py. Detect when cached data might be dangerously stale beyond simple TTL expiry. Monitor for price volatility — if a product's price has changed every day for the past week (from price history), a 24-hour cache TTL is too long, reduce to 6 hours. Detect "flash sale" indicators in cached data — if a product was marked as having a time-limited campaign when cached, reduce TTL to 3 hours. If the user is about to make a purchase decision, always offer to refresh prices: "Bu fiyatlar 18 saat önce alındı. Satın almadan önce güncellememi ister misiniz?" Never let the user buy based on stale data without at least warning them.

Phase 11: Special Intelligence Features
Step 11.1 — Seasonal Intelligence Engine
Create src/shopping/intelligence/seasonal.py. Go beyond static calendar (Step 3.1). Track and learn: which product categories discount during which events (electronics during Black Friday, food during Ramazan, outdoor furniture in autumn clearance), how deep discounts typically go per category per event (electronics 10-25%, fashion 30-50%, groceries 5-15%), build this knowledge over time from observed price history data (Step 0.2 price_history table). When user searches for a product: check if a known discount event is within 30 days, estimate potential savings based on historical patterns, present as "wait vs buy now" recommendation with confidence level. Update patterns after each sale event with actually observed discounts.

Step 11.2 — Cross-Category Bundle Detector
Create src/shopping/intelligence/bundle_detector.py. Detects when buying products together is cheaper or smarter than separately. Types: same-store bundles (detect "birlikte al" / "set" listings), cross-store optimization (CPU cheapest on Trendyol but motherboard cheapest on Amazon — calculate total including shipping for each store vs splitting), complementary product suggestions (buying a coffee machine → suggest descaler, good beans, spare filter).

Turkish-specific bundle patterns: "al X öde Y" (buy X pay Y — common at groceries and cosmetics), "X alana Y hediye" (buy X get Y free), "set fiyatı" (set price — common for kitchen/home), "çeyiz paketi" (trousseau package — common cultural package deals for home setup). Detect store-level cart discounts — Trendyol sometimes offers "spend 500 TL get 50 TL off" across any products, meaning buying multiple unrelated items from same store can trigger a discount.

Step 11.3 — Used Market Awareness
Create src/shopping/intelligence/used_market.py. For applicable categories, check used market options. Sources: sahibinden.com (dominant, Step 1.12), dolap.com (fashion), modacruz.com (fashion), Facebook Marketplace via Perplexica.

Safety rules per category: baby products, medical devices, car safety equipment (brake pads, tires), personal protective equipment — never suggest used, actively warn against it. Electronics by component: used GPU acceptable with caveats, used SSD/HDD risky (wear level), used monitor usually fine, used keyboard/mouse fine. When used option found significantly cheaper (>40% less): include as suggestion with appropriate caveats.

Refurbished awareness: Apple Certified Refurbished, manufacturer refurbished on Amazon, Hepsiburada "yenilenmiş" section. Refurbished is the sweet spot — comes with warranty, significantly cheaper, lower risk. Present alongside new and used options when available.

Step 11.4 — Warranty & Service Intelligence
Create src/shopping/intelligence/warranty.py. Turkish consumer law specifics: mandatory 2-year warranty for all goods (Tüketici Kanunu 6502), 3-year warranty for electrical/electronic goods that cannot be repaired within service period, "ayıplı mal" (defective goods) return rights within 30 days regardless of store policy, right to choose between repair/replacement/refund within warranty period (consumer's choice, not seller's). Advise: "Even though this seller says '1-year warranty', Turkish law guarantees you 2 years minimum."

Brand service network quality from Step 3.7 knowledge base. Factor into recommendations: cheaper product with no local service may cost more long-term. Extended warranty evaluation: calculate expected failure cost (from Şikayetvar data) × failure probability vs extended warranty cost. Flag specific known issues from Şikayetvar patterns: "Samsung washing machine control board failure is common after year 3 — extended warranty recommended for this model."

Step 11.5 — Energy Cost Calculator
Create src/shopping/intelligence/energy_cost.py. For appliances, calculate total cost of ownership including energy. Extract energy rating from specs (A+++, A++, etc.), extract annual energy consumption in kWh if listed. Apply Turkish electricity tariff: tiered pricing (first 150 kWh at lower rate, above at higher rate), different rates for single-phase vs three-phase connections. Include current tariff rates (update quarterly — published by EPDK). Include water costs for washing machine/dishwasher comparisons (İSKİ rates for Istanbul, adjust by city if user location known). Include natural gas costs for gas vs electric oven comparisons.

Show tangible comparison: "Machine A: ~45 TL/month electricity. Machine B: ~62 TL/month. Over 5 years, Machine A saves you 1,020 TL — more than the 800 TL price difference." Makes abstract energy ratings concrete and actionable.

Step 11.6 — Bulk / Wholesale Detection
Create src/shopping/intelligence/bulk_detector.py. For consumables and frequently purchased items, detect bulk buying opportunities. Compare: unit price at different quantities (1kg vs 5kg rice, 1-pack vs 4-pack toothpaste), calculate price per unit at each quantity. Factor in: shelf life (don't recommend 10kg flour if single household), storage requirements (do they have space), consumption rate (if known from user profile). Detect "fake bulk deals" where the bulk price per unit is actually higher than buying singles — this happens surprisingly often on Turkish marketplace sites.

Step 11.7 — Seller Reputation Analyzer
Create src/shopping/intelligence/seller_analysis.py. For marketplace sites (Trendyol, Hepsiburada, N11, Amazon), evaluate the specific seller. Extract: seller rating, number of ratings, response time, return policy, store age. Cross-reference seller name on Şikayetvar for complaints. Flag: new sellers with very few ratings (risky), sellers with high volume but dropping ratings (quality decline), sellers with significantly lower prices than others (could be counterfeit, grey market, or missing warranty).

Track seller ratings over time across sessions. Store seller snapshots in cache: seller_id, platform, rating, review_count, observed_date. Detect: rapidly growing sellers with suspiciously fast review accumulation (potential fake reviews), sellers whose ratings drop after a sale event (sold low quality during campaign), sellers who appear across multiple platforms under similar names. Seller price pattern analysis — some sellers consistently price 5% below market then compensate with slow shipping or poor packaging.

When recommending, prefer established sellers unless price difference is substantial (>15%), and explicitly note when recommending from a newer/lower-rated seller. LLM synthesizes: "This price is 400 TL below average, and the seller has only 12 ratings over 2 months. The next cheapest option is 400 TL more but from an established seller with 4.8★ over 15,000 ratings."

Step 11.8 — Import vs Domestic Advisor
Create src/shopping/intelligence/import_domestic.py. Maintain a mapping of major brands to origin: domestic (Arçelik, Beko, Vestel, Altus, Arzum, Fakir, Sinbo), imported-but-local-production (Samsung TR, LG TR, Bosch-Siemens TR), pure import (Apple, Dyson, KitchenAid, most PC component brands). When TRY weakens: recommend domestic alternatives where they exist. When TRY strengthens: imported options become more competitive. Track USD/TRY rate (free API: exchangerate.host) and include in analysis when relevant.

Grey market and parallel import detection: many electronics sold in Turkey as "ithalatçı garantili" (importer warranty) rather than "distribütör garantili" (official distributor warranty). Consequences: no official brand service, sometimes different firmware/region, possibly missing Turkish language support, IMEI may not be registered with BTK for phones (illegal to use unregistered phones after a period). Detect clues: price significantly below market (>20% cheaper often means grey import), seller descriptions containing "ithalatçı garantili", "global versiyon", "Türkiye garantisi yoktur". Flag clearly: "⚠️ Bu ürün resmi Türkiye distribütörü garantisi taşımıyor olabilir." For phones: always check BTK registration requirement and warn.

Step 11.9 — Counterfeit / Fraud Detection
Create src/shopping/intelligence/fraud_detector.py. Turkish marketplaces have counterfeit product issues, especially in: cosmetics, perfume, branded clothing, phone accessories, memory cards/USBs (fake capacity), chargers (safety risk). Detect red flags: price too good to be true (>50% below market for branded goods), product images that look stolen/generic, seller with very few ratings selling premium brands, "A kalite" or "muadil" (equivalent/replica) keywords in title or description, missing brand authorization claims. For safety-critical products (chargers, power banks, car parts, baby products): always recommend buying from official stores or verified sellers, actively warn against suspiciously cheap options. For memory products: note that fake capacity SD cards and USB drives are common — recommend buying from authorized retailers only.

Step 11.10 — Campaign Pattern Learner
Create src/shopping/intelligence/campaign_patterns.py. Over time, learn which products/categories go on sale during which events and by how much. Data source: price history from cache, Akakçe historical data. Build per-category patterns: "RAM prices dropped average 18% during Black Friday 2023, 12% during 11.11, didn't drop during Ramazan." When user searches: cross-reference with upcoming events and learned patterns to give concrete waiting advice: "DDR5 RAM typically drops 15-20% during Black Friday. Current price: 2,800 TL. Expected Black Friday price: ~2,350 TL. Black Friday is 23 days away. Saving: ~450 TL. Worth waiting if not urgent." Initially the pattern database is empty — populate from Akakçe historical data on first use and grow over time.

Step 11.11 — Complementary Product Suggester
Create src/shopping/intelligence/complementary.py. When user buys or researches a product, suggest things they'll likely need. Two types: immediate complements (phone → case + screen protector; coffee machine → descaler + beans; printer → cables, paper, extra cartridges) and deferred complements (washing machine → dryer or drying rack eventually; laptop → external monitor for home use). LLM generates complement suggestions per product — don't hardcode. Add economic intelligence: "The printer costs 3,000 TL but replacement cartridges are 800 TL each and last ~200 pages. Consider a laser printer at 5,000 TL where toner lasts 2,000 pages — cheaper total cost if you print more than 400 pages per year."

Step 11.12 — Environmental / Efficiency Advisor
Create src/shopping/intelligence/environmental.py. For applicable categories: compare environmental impact as a secondary factor. Energy efficiency ratings and real-world impact (connected to Step 11.5), water efficiency for washing machines and dishwashers, repairability score (European repairability index where available), expected product lifespan from review data (if multiple reviewers mention failure at year 3, that's a durability signal), recyclability and disposal considerations. Frame in cost terms: "The A+++ model uses 40% less electricity — that's 600 TL savings over 5 years" rather than abstract environmental benefits. Present as secondary information, not primary.

Phase 12: Testing & Validation
Step 12.1 — Scraper Test Suite
Create tests/shopping/test_scrapers.py. For each scraper: save real HTML/JSON response as test fixture (one-time capture), test parsing logic against fixture, test price extraction with various Turkish formats, test dimension extraction from real product descriptions, test graceful failure on malformed responses.

Fixture refresh automation: a scheduled GitHub Action (monthly) that fetches fresh copies of each fixture URL and diffs against stored fixture. If diff exceeds threshold (significant structural change), alert that scraper tests need fixture updates.

Canary tests: one live request per scraper per week (extremely low volume, negligible ban risk) that verifies the scraper still returns valid results. Run on GitHub Actions. If canary fails, alert before a user encounters the failure.

Cross-scraper consistency tests: search for the same well-known product across all scrapers, verify that prices are within 20% of each other.

Integration tests that actually hit live sites (run manually, not in CI): search for a known product, verify result count > 0, verify prices in plausible range, verify all required fields populated.

Step 12.2 — Intelligence Module Test Suite
Create tests/shopping/test_intelligence.py. Test query analyzer: feed 20+ diverse queries (Turkish and English, vague and specific, single product and combo), verify intent classification and constraint extraction. Test alternative generator: for known product types, verify sensible alternatives. Test product matcher: create fake product listings with slight name variations, verify matching accuracy. Test value scorer: create products with known specs and prices, verify scoring ranks correctly. Test constraint checker: create products with known dimensions, verify dimensional filtering works. Use deterministic LLM outputs where possible (mock the LLM) for CI-compatible tests.

Adversarial test cases: intentionally ambiguous queries ("apple" — fruit or brand?), misspelled Turkish ("camsir makinesi"), impossible constraints ("200cm wide washing machine" → should report no results), harmful substitutions ("gluten-free flour" → should NOT suggest regular flour), products with similar names but different specs ("Kingston A400 240GB" vs "Kingston A400 480GB" — related but different). Bias testing: does the system consistently favor certain stores or brands due to scraper reliability differences?

Step 12.3 — End-to-End Scenario Tests
Create tests/shopping/test_scenarios.py. Define 15+ realistic scenarios with expected behavior:

Standard: "find me a DDR5 32GB RAM" → should generate alternative configs, compare across sources, check reviews. "I need a cupboard for my 60cm oven" → should ask about oven type, search with dimensional constraints, filter by dimensions. "Best coffee machine around 5000 TL" → should explore types, check Şikayetvar, consider timing. "Chicken breast cheapest per kilo" → should find prices, suggest turkey if cheaper, compare stores. "Upgrade my Ryzen 3600 setup for gaming" → should ask about current components, explore CPU-only vs platform upgrade, build combos.

Turkish-culture-specific: "Çeyiz hazırlığı" (trousseau preparation — multi-category combo), "Bayram hediyesi" (holiday gift), "Yazlık ev hazırlığı" (summer house preparation), "Ramazan alışverişi" (Ramadan grocery shopping), "Okul dönemi" (back to school).

For each scenario: define expected flow of steps, expected search queries generated, expected types of recommendations. Can run against mocked scrapers with fixture data or against live data for manual verification.

Step 12.4 — Output Quality Evaluation
Create tests/shopping/test_output_quality.py. Use local LLM as evaluator (LLM-as-judge pattern). For each test scenario: run full pipeline, feed output to evaluator LLM: "Score 1-5 on: relevance (did it answer the question?), completeness (did it consider alternatives and constraints?), accuracy (are prices and specs plausible?), helpfulness (would a real user find this useful?), Turkish market awareness (did it consider Turkey-specific factors?)". Log scores over time.

User simulation testing: create synthetic user personas (budget-conscious student, tech-enthusiast with high budget, elderly person buying first smartphone, parent shopping for baby products, someone furnishing a new apartment), run full pipeline for each persona with appropriate queries. Evaluate per persona: did the system adapt language and detail level? Did it ask appropriate questions? Did it avoid recommending outside the persona's context? Track quality per persona type.

Step 12.5 — Performance Benchmarks
Create tests/shopping/test_performance.py. Measure and track: end-to-end latency for quick search workflow (target: <30 seconds), end-to-end latency for full shopping workflow (target: <5 minutes excluding GitHub Actions wait), cache hit rate (target: >50% after first week of use), LLM inference time per intelligence module (identify bottleneck), total tokens consumed per shopping session (affects local LLM memory pressure). Set up alerts if performance degrades beyond thresholds. Track over time to catch regressions.

Phase 13: Maintenance & Evolution
Step 13.1 — Scraper Health Dashboard
Extend existing Grafana dashboard (referenced by sandbox/kutay-dashboard.json). Add panels: per-domain scraper success rate (last 24h, 7d, 30d), average response time per domain, cache hit rate, daily request budget usage vs limit, number of blocked/degraded domains, shopping session metrics (sessions per day, average products compared, most searched categories, most recommended stores), cache efficiency (hit rate by data type, storage size, stale data percentage), intelligence quality metrics (how often does user accept top recommendation vs asking for more), rate budget usage trends, knowledge freshness indicator (when was each knowledge base file last updated, highlight any >3 months stale). Alert rules: if any scraper drops below 80% success rate for 6 hours, send Telegram notification.

Step 13.2 — Knowledge Base Refresh Workflow
Create src/shopping/maintenance/knowledge_refresh.py. Semi-automated refresh of knowledge files. Monthly: run Perplexica search for "Turkish electronics market changes", "new store openings Turkey", "brand exits Turkey" and present summary for manual review. Quarterly: refresh compatibility databases (new CPU/GPU/motherboard generations).

Category-specific refresh triggers: when a scraper detects a new brand not seen before, flag for knowledge base review. When Şikayetvar complaints for a brand spike suddenly, flag for store profiles update. When a compatibility check fails because a new socket or DDR generation appears, flag for compatibility update. Create knowledge_update_queue that accumulates flags, present in weekly digest: "This week: 3 new brands detected in coffee machines, 1 compatibility gap (LGA1851 not in database), Philips complaint rate increased 40%."

On-demand: when a user query reveals a knowledge gap, log it. Keep knowledge_gaps.log — review periodically to improve coverage.

Step 13.3 — Prompt Optimization Pipeline
Create src/shopping/maintenance/prompt_tuning.py. Track which LLM prompts produce the best outputs. For each intelligence module: log prompt used, log output, log user feedback if available. Periodically review low-scoring outputs (from Step 12.4), identify prompt weaknesses, test variations. Store prompt versions using existing src/memory/prompt_versions.py. A/B test when feasible.

Category-specific prompt variants: electronics (spec-heavy, benchmark-aware), appliances (energy efficiency, service network, installation), grocery (unit pricing, freshness, nutrition), furniture (dimensional fit, material quality, assembly). Track quality per category and tune independently.

Step 13.4 — Self-Improving Substitution Map
Extend src/shopping/intelligence/substitution.py. When LLM suggests a substitution that user acts on (inferred from conversation): log the substitution pair. When user rejects: log rejection with reason. Over time: add successful substitutions to the map (Step 3.4), add rejection reasons to boundaries. Learn which substitutions work for this specific user and in general.

Step 13.5 — Scraper Auto-Repair
Create src/shopping/maintenance/auto_repair.py. When a scraper starts failing due to site changes: log the failed HTML response, send old working fixture + new broken HTML to local LLM: "The website structure changed. Identify what changed in the product listing structure — where are the product names, prices, and ratings now?" LLM suggests new CSS selectors or JSON paths. Scrapling's adaptive selectors should be the first line of defense — only escalate to LLM-assisted repair when Scrapling adaptation also fails (indicates major structural change). Don't auto-deploy fixes — present for human review.

Consider structuring scraper configs (CSS selectors, API endpoints, JSON paths) as separate data files rather than embedded in code, enabling easier updates.

Step 13.6 — System Self-Assessment
Create src/shopping/maintenance/self_assessment.py. Periodic (weekly) automated self-assessment via GitHub Actions. Run the system against 5 predefined benchmark queries with known expected outcomes. Score: did it find products on all expected sources? Are prices in plausible ranges? Did intelligence modules fire correctly? Did output meet quality thresholds? Compare week-over-week: improving, stable, or declining? If declining: generate diagnostic report identifying which component is degrading.

Step 13.7 — Feature Usage Analytics
Create src/shopping/maintenance/analytics.py. Track which intelligence features actually get used and valued. Measure: how often does substitution engine fire and user engage? How often does timing advisor say "wait" and user follow? How often does combo builder produce acted-upon results? Which scraper sources most frequently appear in final recommendation? Use analytics to prioritize maintenance effort: if grocery scrapers are never used, deprioritize. If timing advisor is used constantly, invest in improving accuracy.
