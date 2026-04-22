from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Product:
    name: str
    url: str
    source: str
    original_price: float | None = None
    discounted_price: float | None = None
    discount_percentage: float | None = None
    currency: str = "TRY"
    image_url: str | None = None
    specs: dict = field(default_factory=dict)
    rating: float | None = None
    review_count: int | None = None
    site_rank: int | None = None  # 0-indexed position in the site's own response
    availability: str = "in_stock"  # in_stock / low_stock / out_of_stock / preorder
    seller_name: str | None = None
    seller_rating: float | None = None
    seller_review_count: int | None = None
    shipping_cost: float | None = None
    shipping_time_days: int | None = None
    free_shipping: bool = False
    installment_info: dict | None = None
    warranty_months: int | None = None
    sku: str | None = None
    category_path: str | None = None
    fetched_at: str | None = None  # ISO 8601 timestamp


@dataclass
class Review:
    text: str
    source: str
    rating: float | None = None
    date: str | None = None
    author: str | None = None
    verified_purchase: bool = False
    helpful_count: int = 0
    language: str | None = None


@dataclass
class ProductMatch:
    products: list[Product] = field(default_factory=list)
    canonical_name: str = ""
    canonical_specs: dict = field(default_factory=dict)
    confidence_score: float = 0.0


@dataclass
class PriceHistoryEntry:
    price: float = 0.0
    source: str = ""
    date: str = ""
    was_campaign: bool = False


@dataclass
class ShoppingQuery:
    raw_query: str = ""
    interpreted_intent: str | None = None
    constraints: list = field(default_factory=list)
    budget: float | None = None
    category: str | None = None
    generated_searches: list = field(default_factory=list)


@dataclass
class UserConstraint:
    type: str = ""  # dimensional / compatibility / budget / dietary / electrical / temporal
    value: str = ""
    hard_or_soft: str = "hard"  # "hard" / "soft"
    source: str = "user-stated"  # "user-stated" / "system-inferred"


@dataclass
class Combo:
    products: list[ProductMatch] = field(default_factory=list)
    total_price: float = 0.0
    compatibility_notes: list[str] = field(default_factory=list)
    value_score: float = 0.0


@dataclass
class ShoppingSession:
    session_id: str = ""
    user_query: str = ""
    analyzed_intent: dict | None = None
    products_found: list[Product] = field(default_factory=list)
    recommendations_made: list[dict] = field(default_factory=list)
    user_actions: list[dict] = field(default_factory=list)
    timestamps: dict = field(default_factory=dict)
