# TruthRate Fact-First Style Guide

**Style Overview**:
A clean flat design for **light theme** mobile platform centered on deep forest green, using bold fact emphasis through distinct surface colors and soft shadows for elevation hierarchy. Three bright accent colors (amber, slate blue, coral) provide functional distinction for content types, creating truth-focused authority with information primacy and ad-friendly spacing.

## Colors
### Primary Colors
  - **primary-base**: `text-[#1B4D3E]` or `bg-[#1B4D3E]`
  - **primary-lighter**: `bg-[#2A6B56]`
  - **primary-darker**: `text-[#0F3329]` or `bg-[#0F3329]`

### Background Colors
- **bg-page**: `bg-[#FAF9F7]`
- **bg-container-primary**: `bg-white` - Main fact cards, content containers
- **bg-container-secondary**: `bg-[#F5F3F0]` - Secondary information cards
- **bg-container-inset**: `bg-[#F0EDE8]` - Input fields, search bars
- **bg-fact-urgent**: `bg-[#FFF4E6]` - Urgent/critical fact cards background
- **bg-fact-neutral**: `bg-[#F0F4F8]` - Neutral fact cards background
- **bg-fact-opinion**: `bg-[#FFF5F5]` - Opinion/review cards background
- **bg-bottom-navigation**: `bg-white`

### Text Colors
- **color-text-primary**: `text-[#1A1A1A]`
- **color-text-secondary**: `text-[#4A4A4A]`
- **color-text-tertiary**: `text-[#757575]`
- **color-text-quaternary**: `text-[#A3A3A3]`
- **color-text-on-dark-primary**: `text-white` - Text on primary-base, primary-darker surfaces
- **color-text-on-dark-secondary**: `text-white/80` - Text on primary-base, primary-darker surfaces
- **color-text-link**: `text-[#1B4D3E]` - Links, clickable text

### Functional Colors
Use to categorize information types and provide visual distinction for different content categories.
  - **color-urgent-default**: #F59E0B - Urgent fact highlights, critical information
  - **color-urgent-light**: #FEF3C7 - Tag/label bg for urgent content
  - **color-neutral-default**: #475569 - Neutral fact highlights
  - **color-neutral-light**: #E0E7FF - Tag/label bg for neutral facts
  - **color-opinion-default**: #EF4444 - Opinion/review indicators
  - **color-opinion-light**: #FEE2E2 - Tag/label bg for opinions
  - **color-success-default**: #10B981 - Verified facts, positive indicators
  - **color-success-light**: #D1FAE5 - Tag/label bg for verified content
  - **color-warning-default**: #F59E0B - Alert indicators
  - **color-warning-light**: #FEF3C7 - Warning banner bg

### Accent Colors
  - Secondary palette for content type categorization and visual interest. Use intentionally to maintain information hierarchy.
  - **accent-amber**: `text-[#F59E0B]` or `bg-[#F59E0B]` - Urgent facts, critical information
  - **accent-slate**: `text-[#64748B]` or `bg-[#64748B]` - Neutral facts, data points
  - **accent-coral**: `text-[#FB7185]` or `bg-[#FB7185]` - Reviews, opinions, subjective content

### Data Visualization Charts
  - Standard data colors: #1B4D3E, #2A6B56, #3D8B6F, #52AA88, #6BC9A2, #8DD9B8
  - Important data highlights: #F59E0B, #EF4444, #10B981, #64748B

## Typography
- **Font Stack**:
  - **font-family-base**: `-apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif` — For regular UI copy

- **Font Size & Weight**:
  - **Caption**: `text-xs font-normal` (12px / 400) - Metadata, timestamps, secondary labels
  - **Body small**: `text-sm font-normal` (14px / 400) - Supporting text, descriptions
  - **Body default**: `text-base font-normal` (16px / 400) - Primary content text
  - **Body emphasized**: `text-base font-semibold` (16px / 600) - Important inline text
  - **Fact Title**: `text-lg font-bold` (18px / 700) - Fact card headlines
  - **Card Title**: `text-base font-semibold` (16px / 600) - Standard card headers
  - **Page Title**: `text-xl font-bold` (20px / 700) - Screen titles
  - **Display**: `text-2xl font-bold` (24px / 700) - Featured content, key statistics

- **Line Height**: 1.5

## Border Radius
  - **Small**: 6px — Elements inside cards, tags
  - **Medium**: 12px - Input fields, buttons
  - **Large**: 16px — Fact cards, content containers
  - **Full**: full — Avatars, badges, category pills

## Layout & Spacing
  - **Spacing Scale**:
  - **Base Unit**: 4px
  - **Tight**: 8px - Closely-related elements within cards
  - **Compact**: 12px - Between card elements, list item spacing
  - **Standard**: 16px - Section spacing, card padding
  - **Relaxed**: 24px - Major section separation, content grouping

## Create Boundaries
Surface color contrast combined with soft elevation shadows to create clear information hierarchy
### Borders
  - **Case 1**: No borders for most containers. Rely on surface color contrast and shadows.
  - **Case 2**: If needed for input fields or specific emphasis
    - **Default**: 1px solid #E5E5E5. Used for inputs. `border border-[#E5E5E5]`
    - **Focused**: 2px solid #1B4D3E. Used for active input states. `border-2 border-[#1B4D3E]`

### Dividers
  - **Case 1**: Use sparingly, primarily within fact cards to separate information sections.
  - **Case 2**: `border-t border-[#E5E5E5]` for content separation within cards.

### Shadows & Effects
  - **Case 1 (No shadow)**: For inline elements, tags, and flat UI components.
  - **Case 2 (Subtle elevation)**: `shadow-[0_1px_3px_rgba(0,0,0,0.08)]` - Standard cards, list items
  - **Case 3 (Moderate elevation)**: `shadow-[0_2px_8px_rgba(0,0,0,0.10)]` - Fact cards, emphasized containers
  - **Case 4 (Pronounced elevation)**: `shadow-[0_4px_12px_rgba(0,0,0,0.12)]` - Modal dialogs, floating action buttons

## Assets
### Image
- For normal `<img>`: object-cover
- For `<img>` with:
  - Slight overlay: object-cover brightness-90
  - Heavy overlay: object-cover brightness-75

### Icon
- Use Lucide icons from Iconify.
- To ensure an aesthetic layout, each icon should be centered in a square container, typically without a background, matching the icon's size.
- Use Tailwind font size to control icon size
- Example:
  ```html
  <div class="flex items-center justify-center bg-transparent w-5 h-5">
  <iconify-icon icon="lucide:shield-check" class="text-xl"></iconify-icon>
  </div>
  ```

### Third-Party Brand Logos:
   - Use Brand Icons from Iconify.
   - Logo Example:
     Monochrome Logo: `<iconify-icon icon="simple-icons:x"></iconify-icon>`
     Colored Logo: `<iconify-icon icon="logos:google-icon"></iconify-icon>`

### User's Own Logo:
- To protect copyright, do **NOT** use real product logos as a logo for a new product, individual user, or other company products.
- **Icon-based**:
  - **Graphic**: Use a simple, relevant icon (e.g., a `shield-check` icon for fact verification, a `search` icon for discovery features).

## Page Layout - Mobile
```html
<!-- Mobile Layout Template: Adjust body width (w-[390px]) based on target device -->
<body class="w-[390px] min-h-[844px] bg-[#FAF9F7] font-['-apple-system','BlinkMacSystemFont','Segoe_UI','Helvetica_Neue','Arial',sans-serif] leading-[1.5]">

  <!-- Top Fixed Header: Contains status bar safe area and navigation bar -->
  <div class="z-10 fixed top-0 w-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.08)]">
    <!-- Default Top Safe Area -->
    <div class="h-[env(safe-area-inset-top,0px)]"></div>
    <!-- Top Navigation Bar: Standard height of 56px (h-14), remove if not needed -->
    <header class="h-14 flex items-center justify-between px-4">
      <!-- Navigation content goes here -->
    </header>
  </div>

  <!-- Top Spacer: Pushes content down to avoid overlapping with the fixed header. Adjust as needed, for example, set both `h` to 0 if a hero image is to be displayed under the status bar. -->
  <div>
    <!-- `h` should match the the top safe area height -->
    <div class="h-[env(safe-area-inset-top,0px)]"></div>
    <!-- `h` should match the navigation bar height. Adjust as needed. -->
    <div class="h-14"></div>
  </div>

  <!-- Main Scrollable Content Area  -->
  <main class="py-4 space-y-4">
    <!-- Main content goes here, use section with horizontal page padding(px-4) -->
    <section class="px-4 ...">
    </section>
  </main>

  <!-- Bottom Spacer: Avoid overlapping with the fixed bottom bars -->
  <div>
    <!-- `mt` is an additional margin to prevent layout miscalculations. `h` should match the height of the Bottom bar(Navigation, Toolbar, or Input Field). Adjust `h` if these bottom components change. -->
    <div class="mt-[16px] h-[72px]"></div>
    <!-- `h` equals to Bottom Safe Area -->
    <div class="h-[env(safe-area-inset-bottom,0px)]"></div>
    <!-- Space for Floating Action Button, remove entire div if not needed. `h` equals to the height of the Floating Action Button -->
    <div class="h-12"></div>
  </div>

  <!-- Bottom Fixed Area: Contains FAB and/or bottom navigation and/or bottom toolbar and/or bottom input dialog and bottom safe area -->
  <div class="z-10 fixed bottom-0 w-full flex flex-col">

    <!-- Floating Action Button (Optional): Remove entire div if not needed -->
    <div class="flex flex-col gap-4 px-4 pb-6 items-end">
      <button class="w-12 h-12 flex items-center justify-center bg-[#1B4D3E] rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.12)]">
        <!-- FAB content: icon only, no text -->
      </button>
    </div>

    <!-- Bottom bar(container) for Navigation/Toolbar/Input Field (bg and safe area) (Optional): Remove entire div if not needed -->
    <div class="bg-white shadow-[0_-1px_3px_rgba(0,0,0,0.08)]">
      <!-- Bottom Navigation/Toolbar/Input Field(layout) -->
      <nav class="flex justify-around py-3 px-1">
        <!-- Navigation Item: flex-1; text-[#757575](Default); text-[#1A1A1A](Active) -->
        <div class="flex flex-1 flex-col items-center gap-1">
            <div class="w-6 h-6 flex items-center justify-center">
                <iconify-icon icon="lucide:home" class="text-xl text-[#757575]"></iconify-icon>
            </div>
            <span class="text-xs font-normal text-[#757575]">Home</span>
        </div>
        <!-- Center FAB in Navigation (Optional): Remove entire div if not needed -->
        <div class="flex flex-1 flex-col items-center">
            <button class="w-12 h-12 flex items-center justify-center bg-[#1B4D3E] rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.12)]">
                <!-- FAB content: icon only, no text -->
            </button>
        </div>
      </nav>
      <!-- Default Bottom Safe Area -->
      <div class="h-[env(safe-area-inset-bottom,0px)]"></div>
    </div>
    <!-- Alternative Bottom Safe Area: Use ONLY when there's no Bottom bar -->
    <div class="h-[env(safe-area-inset-bottom,0px)]"></div>
  </div>
</body>
```

## Tailwind Component Examples (Key attributes)
**Important Note**: Use utility classes directly. Do NOT create custom CSS classes or add styles in <style> tags for the following components
### Basic

- **Button**:
  - Example 1 (Primary button):
    - button: flex items-center justify-center gap-2 bg-[#1B4D3E] text-white rounded-xl px-6 py-3 hover:bg-[#0F3329] transition
      - icon (optional)
      - span: whitespace-nowrap font-semibold
  - Example 2 (Secondary button):
    - button: flex items-center justify-center gap-2 bg-[#F5F3F0] text-[#1B4D3E] rounded-xl px-6 py-3 hover:bg-[#F0EDE8] transition
      - icon (optional)
      - span: whitespace-nowrap font-semibold
  - Example 3 (Text button):
    - button: flex items-center gap-1 text-[#1B4D3E] hover:opacity-70 transition
      - span: whitespace-nowrap font-medium

- **Tag Group (Filter Tags)**
  - container(scrollable): flex gap-2 overflow-x-auto [&::-webkit-scrollbar]:hidden
    - label (Tag item):
      - input: type="radio" name="tag1" class="sr-only peer" checked
      - div: bg-[#F5F3F0] text-[#4A4A4A] peer-checked:bg-[#1B4D3E] peer-checked:text-white rounded-full px-4 py-2 hover:opacity-80 transition whitespace-nowrap text-sm font-medium

### Data Entry
- **Progress bars/Sliders**: h-2
- **Checkbox**
  - label: flex items-center gap-3
    - input: type="checkbox" class="sr-only peer"
    - div: w-5 h-5 bg-[#F0EDE8] rounded-md flex items-center justify-center peer-checked:bg-[#1B4D3E] text-transparent peer-checked:text-white transition
      - svg(Checkmark): stroke="currentColor" stroke-width="3"
    - span: text-base text-[#1A1A1A]

- **Radio button**
  - label: flex items-center gap-3
    - input: type="radio" name="option1" class="sr-only peer"
    - div: w-5 h-5 bg-[#F0EDE8] rounded-full flex items-center justify-center peer-checked:bg-[#1B4D3E] text-transparent peer-checked:text-white transition
      - svg(dot indicator): fill="currentColor"
    - span: text-base text-[#1A1A1A]

- **Switch/Toggle**
  - label: flex items-center gap-3
    - div: relative
      - input: type="checkbox" class="sr-only peer"
      - div(Toggle track): w-12 h-7 bg-[#F0EDE8] rounded-full peer-checked:bg-[#1B4D3E] transition
      - div(Toggle thumb): absolute top-1 left-1 w-5 h-5 bg-white rounded-full shadow-[0_1px_3px_rgba(0,0,0,0.2)] peer-checked:translate-x-5 transition
    - span: text-base text-[#1A1A1A]

- **Select/Dropdown**
  - Select container: flex items-center justify-between bg-[#F0EDE8] rounded-xl px-4 py-3
    - text: text-base text-[#1A1A1A]
    - Dropdown icon(square container): w-5 h-5 flex items-center justify-center bg-transparent
      - icon: text-[#4A4A4A]

### Container
- **Card**
    - Example 1 (Fact Card - Urgent):
        - Card: bg-[#FFF4E6] rounded-2xl flex flex-col p-4 gap-3 shadow-[0_2px_8px_rgba(0,0,0,0.10)]
        - Fact badge: bg-[#F59E0B] text-white text-xs font-bold px-2 py-1 rounded-full w-fit
        - Title: text-lg font-bold text-[#1A1A1A]
        - Content: text-base text-[#4A4A4A]
    - Example 2 (Fact Card - Neutral):
        - Card: bg-[#F0F4F8] rounded-2xl flex flex-col p-4 gap-3 shadow-[0_2px_8px_rgba(0,0,0,0.10)]
        - Fact badge: bg-[#64748B] text-white text-xs font-bold px-2 py-1 rounded-full w-fit
        - Title: text-lg font-bold text-[#1A1A1A]
        - Content: text-base text-[#4A4A4A]
    - Example 3 (Opinion/Review Card):
        - Card: bg-[#FFF5F5] rounded-2xl flex flex-col p-4 gap-3 shadow-[0_1px_3px_rgba(0,0,0,0.08)]
        - Opinion badge: bg-[#FB7185] text-white text-xs font-medium px-2 py-1 rounded-full w-fit
        - Content: text-base text-[#4A4A4A]
    - Example 4 (Standard Card with Image):
        - Card: bg-white rounded-2xl flex flex-col gap-3 shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden
        - Image: w-full h-48 object-cover
        - Text area: p-4 flex flex-col gap-2
          - card-title: text-base font-semibold text-[#1A1A1A]
          - card-subtitle: text-sm text-[#4A4A4A]

- **List** (for scrollable lists, settings, etc.)
  - List container: space-y-1
    - list-item: flex items-center justify-between py-4 hover:bg-[#F5F3F0] transition rounded-lg px-2
      - Left content: flex items-center gap-3
        - icon-container (if applicable): w-5 h-5 flex items-center justify-center
          - icon: text-[#1B4D3E]
        - text: text-base text-[#1A1A1A]
      - Right content: flex items-center gap-2
        - value/badge: text-sm text-[#757575]
        - chevron-icon: w-5 h-5 text-[#A3A3A3]

## Additional Notes
- **Fact Hierarchy**: Use distinct background colors and bold typography to ensure factual information stands out from opinions and reviews
- **Color Coding System**: Amber = Urgent/Critical, Slate = Neutral/Data, Coral = Opinion/Subjective
- **Shadow Strategy**: Apply shadows to create clear elevation hierarchy - fact cards receive more prominence than opinion cards
- **Accessibility**: Maintain minimum 4.5:1 contrast ratio for all text on colored backgrounds
- **Ad-Friendly Spacing**: Standard 16px and relaxed 24px spacing provides clear visual separation for ad placements without disrupting content flow
