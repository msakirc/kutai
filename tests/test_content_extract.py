# tests/test_content_extract.py
"""Tests for content_extract module — Trafilatura-based content extraction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.content_extract import extract_content, ExtractedContent


class TestExtractContent(unittest.TestCase):

    def test_returns_extracted_content_dataclass(self):
        html = """
        <html><body>
        <article>
        <h1>Best Coffee Machines 2026</h1>
        <p>We tested 15 coffee machines to find the best options for every budget.
        The DeLonghi Dinamica costs 28000 TL and scored highest in our tests.
        The Philips 3200 at 18000 TL offers the best value for money.</p>
        </article>
        </body></html>
        """
        result = extract_content(html, url="https://example.com/coffee")
        self.assertIsInstance(result, ExtractedContent)
        self.assertIn("coffee", result.text.lower())
        self.assertEqual(result.url, "https://example.com/coffee")
        self.assertGreater(result.word_count, 10)

    def test_detects_prices(self):
        html = """
        <html><body><article>
        <p>The iPhone 15 costs $799 in the US market. In Turkey, it retails
        for 45000 TL. The Samsung Galaxy S24 is priced at 35000₺.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertTrue(result.has_prices)

    def test_no_prices_detected(self):
        html = """
        <html><body><article>
        <p>Transformers use attention mechanisms to process sequential data.
        The key innovation is the self-attention layer.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertFalse(result.has_prices)

    def test_detects_reviews(self):
        html = """
        <html><body><article>
        <p>User rating: 4.5 out of 5 stars. Based on 230 reviews.</p>
        <p>Pros: Great battery life. Cons: Heavy weight.</p>
        <div class="review">Amazing product, highly recommend!</div>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertTrue(result.has_reviews)

    def test_no_reviews_detected(self):
        html = """
        <html><body><article>
        <p>Python 3.12 was released on October 2, 2023.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertFalse(result.has_reviews)

    def test_empty_html_returns_empty_content(self):
        result = extract_content("")
        self.assertEqual(result.text, "")
        self.assertEqual(result.word_count, 0)

    def test_extracts_title(self):
        html = """
        <html><head><title>Product Review Page</title></head>
        <body><article><p>Content here with enough words to be meaningful
        for the extraction to actually work properly.</p></article></body></html>
        """
        result = extract_content(html)
        self.assertEqual(result.title, "Product Review Page")

    def test_fallback_to_beautifulsoup(self):
        html = "<html><body><p>Short but valid content that should be extracted by the fallback parser at minimum.</p></body></html>"
        result = extract_content(html)
        self.assertIsInstance(result, ExtractedContent)


if __name__ == "__main__":
    unittest.main()
