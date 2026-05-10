# PRD — TruthRate MVP: Universal Review Platform with Dual-Layer Information System

## 1) Background

Current review platforms are limited to specific categories and fail to distinguish between subjective opinions and objective facts, making it difficult for users to trust the information they find. This iteration addresses the Charter's core problem of transparent, credible information access by building a universal platform that separates reviews from facts. It serves both Maya (The Truth Seeker) who needs trustworthy information for decision-making, and Carlos (The Engaged Proprietor) who needs professional tools to manage his business reputation. This is the right time because consumers increasingly demand transparency and businesses need direct response channels to maintain credibility in a crowded digital marketplace.

## 2) Objectives & Desired Outcomes

- Users feel confident making purchase and service decisions because they can clearly distinguish between community opinions and verifiable facts, with credibility indicators guiding their trust
- Contributors build meaningful reputation by providing helpful reviews and accurate facts, feeling motivated to continue participating because their impact is visible and recognized
- Business owners actively engage with their customers through the platform, responding to feedback professionally and providing their perspective, which improves business-customer relationships
- The platform maintains information quality through community-driven fact-checking and helpfulness metrics, with admin oversight ensuring harmful content is removed when necessary
- Non-goals / Boundaries: Advanced automated fact verification, AI-generated content detection, social networking features beyond reviews, e-commerce integration, and international localization beyond English

## 3) Users & Stories

- **Primary Persona:** Maya Chen (Reviewer)
  - Story A: As a reviewer, I want to search for and find reviews about any product or service, so I can make informed decisions without platform category limitations.
  - Story B: As a reviewer, I want to clearly see the difference between subjective reviews and factual claims, so I can understand what's opinion versus verified information.
  - Story C: As a reviewer, I want to quickly leave star ratings without writing full reviews, so I can contribute feedback even when I'm short on time.
  - Story D: As a reviewer, I want to submit factual claims with supporting evidence, so I can share important information that others should know about.
  - Story E: As a reviewer, I want to build my reputation through helpful contributions, so I can become a trusted voice in the community.
  - Story F: As a reviewer, I want to sign up with just a username without sharing my email publicly, so I can maintain my privacy while participating.

- **Secondary Persona:** Carlos Rodriguez (Business Owner)
  - Story G: As a business owner, I want to claim my business listing through a subscription, so I can officially represent my business on the platform.
  - Story H: As a business owner, I want to respond to both reviews and factual claims from a professional web dashboard, so I can address customer concerns and provide context.
  - Story I: As a business owner, I want to upload photos and documentation to support my responses, so I can provide evidence when addressing inaccurate claims.
  - Story J: As a business owner, I want to choose from different subscription tiers, so I can select the plan that best fits my business needs and budget.

## 4) Key Feature

- **Universal Search & Listings:** Users can search for and create listings for any business, product, or service without category restrictions, ensuring comprehensive coverage of reviewable entities
- **Dual-Layer Contribution System:** Users can submit either reviews (subjective opinions with optional star ratings) or facts (objective claims with encouraged supporting media/links), with clear UI separation between the two types
- **Credibility & Reputation System:** Reviews have helpfulness metrics voted by other users; facts have fact-checking mechanisms; individual users accumulate reputation scores based on their contribution quality (review scores, fact accuracy scores, and rating frequency). Users must achieve a minimum reputation score threshold to report facts, ensuring fact reporters understand the distinction between facts and opinions
- **Business Claiming & Subscription:** Business owners can claim listings through tiered subscription packages, gaining access to a professional web dashboard to respond to reviews and facts
- **Multi-Platform Access:** Mobile-first experience optimized for reviewers to browse and contribute on-the-go; feature-rich web interface for business owners to manage their presence professionally
- **Username-Based Authentication:** Privacy-first registration and login using usernames instead of email-based accounts, with strong data protection measures
- **Media Upload Support:** Reviewers can upload photos, videos, and include links in both reviews and fact reports; business owners can upload media and links in their responses to both reviews and facts
- **Admin Moderation Panel:** Platform administrators can manually review flagged content, remove harmful reviews or facts, and intervene in edge cases to maintain platform quality

## 5) Key Flow

- **Example:** Reviewer searches and reads reviews for a coffee shop
  - **Trigger:** Maya needs to find a good coffee shop near her office
  - **Path:** She searches "Blue Moon Cafe Austin" on her phone; the system displays the business page with separate sections for star ratings, written reviews, and factual claims; she can see helpfulness scores for reviews and fact-check statuses for claims; she reads contributions from high-reputation users first
  - **Result:** Maya understands both community sentiment and verified information, making an informed decision about visiting

- **Example:** Reviewer submits a factual claim with evidence
  - **Trigger:** Maya discovers a restaurant is using expired ingredients | **Path:** She navigates to the restaurant's page, selects "Report a Fact," writes her claim, uploads photos as evidence, and confirms she guarantees the truth; the system posts the fact with "Pending fact-check" status | **Result:** The fact appears on the restaurant's page for others to see and fact-check, and Maya's reputation score increases for contributing important information

- **Example:** Business owner claims listing and responds to review
  - **Trigger:** Carlos finds negative reviews about his restaurant that he wants to address | **Path:** He navigates to his business listing on his desktop, selects "Claim this business," chooses a subscription tier and completes payment; he gains access to the business dashboard where he can see all reviews and facts; he writes a professional response to a negative review, uploads a photo showing improvements made, and publishes | **Result:** His response appears below the original review marked as "Business Owner Response," demonstrating his commitment to customer satisfaction

- **Example:** Reviewer quickly leaves star rating
  - **Trigger:** Maya finishes a haircut and wants to provide quick feedback | **Path:** She opens the app, searches for the barbershop, and taps the star rating interface to select 5 stars without writing text; the system confirms submission | **Result:** Her rating contributes to the overall star average, and she feels good about helping others with minimal effort

- **Example:** User fact-checks another user's claim
  - **Trigger:** Maya sees a factual claim about a pharmacy that seems questionable | **Path:** She taps "Fact-check this claim," reviews the supporting evidence provided, and marks it as "Needs verification" with a brief explanation; the system updates the fact-check status | **Result:** The claim's credibility indicator changes to reflect community scrutiny, and Maya earns reputation for participating in quality control

- **Example:** Admin moderates flagged content
  - **Trigger:** A user flags a review as potentially defamatory | **Path:** Admin receives notification, reviews the flagged content in the admin panel, examines context and supporting evidence, and decides to remove the review; the system logs the action | **Result:** The review is removed from the platform, maintaining content quality and protecting against harmful claims

## 6) Competitive Analysis

- **Landscape (who is solving this problem):** Yelp serves restaurant and local business reviews for general consumers; TripAdvisor focuses on travel and hospitality reviews for travelers; Google Reviews provides cross-category business reviews integrated with search for all internet users; Amazon Reviews covers product reviews for online shoppers; niche platforms like Trustpilot serve e-commerce business reviews for online purchasers; manual approaches include asking friends or doing nothing

- **Value Thesis (each player's proposition):** Yelp offers deep local business coverage with strong community engagement but limits to specific categories; TripAdvisor provides travel-specific expertise and booking integration but restricts to hospitality; Google Reviews leverages search dominance and broad coverage but mixes reviews with business information; Amazon Reviews offers verified purchase validation but only covers products sold on platform; Trustpilot focuses on company reputation management but primarily serves e-commerce

- **Strengths / Weaknesses (experience pros/cons):** Yelp has strong community and engagement features but category restrictions frustrate users seeking reviews for non-traditional items; Google Reviews benefits from easy access through search but lacks nuanced fact-checking and business owner tools feel limited; Amazon Reviews provides purchase verification which builds trust but restricts to marketplace items and can't review services or local businesses; niche platforms offer deep expertise but require users to visit multiple sites for different needs

- **Our Differentiators (our unique points):** TruthRate embodies the Charter's Universal and Transparent brand keywords by enabling reviews for literally anything without category boundaries, while Credible and Fair principles manifest through explicit separation of subjective reviews from objective facts with reputation systems for both contributors and business owners. Our unique value lies in the dual-layer information architecture that protects truth from opinion pollution, combined with universal applicability that eliminates the need for multiple review platforms. The trade-off is that we sacrifice deep category-specific features and verified purchase validation in exchange for universal coverage and fact transparency.

- **Switching Costs & Risks (migration costs and risks):** Users accustomed to category-specific platforms may initially feel uncertain about where to find reviews without familiar navigation structures; business owners may resist paying for yet another platform subscription when they already manage multiple review sites; the dual-layer system requires users to learn new mental models for distinguishing facts from opinions; risk of fact misuse exists where users submit opinions as facts, requiring strong onboarding to establish proper usage patterns

- **Notes (reference links):** Yelp business owner dashboard patterns, Google Reviews fact-checking UI concepts, Reddit upvote/downvote credibility systems, Trustpilot subscription tier models
