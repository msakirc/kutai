# TruthRate Mobile Dark Mode Style Guide

**Style Overview**:
A clean flat design optimized for dark mode, featuring bold fact emphasis with bright emerald green accents on a deep charcoal background. Surface color differentiation creates hierarchy while maintaining reduced contrast for eye comfort, with subtle glows for elevated elements. Truth-focused authority through transparency and information primacy.

## Colors
### Primary Colors
  - **primary-base**: `text-[#10D97E]` or `bg-[#10D97E]` - Bright emerald for key facts, CTAs, verification badges
  - **primary-lighter**: `text-[#3EE598]` or `bg-[#3EE598]`
  - **primary-darker**: `text-[#0DB86A]` or `bg-[#0DB86A]`
  - **primary-dim**: `text-[#10D97E]/80` or `bg-[#10D97E]/20` - Subtle highlights

### Background Colors
- **bg-page**: `bg-[#1A1D23]` - Deep charcoal with cool tint
- **bg-container-primary**: `bg-[#23272F]` - Main content cards, elevated surfaces
- **bg-container-secondary**: `bg-[#2B3039]` - Nested containers, secondary cards
- **bg-container-tertiary**: `bg-[#33373F]` - Input fields, interactive elements
- **bg-header**: `bg-[#1A1D23]/95` - Semi-transparent header with backdrop blur
- **bg-bottom-navigation**: `bg-[#23272F]`

### Text Colors
- **color-text-primary**: `text-white/95`
- **color-text-secondary**: `text-white/70`
- **color-text-tertiary**: `text-white/50`
- **color-text-quaternary**: `text-white/30`
- **color-text-on-primary**: `text-[#0A0B0D]` - Dark text on bright emerald backgrounds
- **color-text-link**: `text-[#10D97E]` - Links, clickable text

### Functional Colors
Used for fact categorization, credibility indicators, and status communication.
  - **color-success-default**: `bg-[#0DB86A]` - Verified facts, high credibility
  - **color-success-light**: `bg-[#0DB86A]/20` - Success state backgrounds
  - **color-warning-default**: `bg-[#F5A623]` - Warm gold for urgent/breaking facts
  - **color-warning-light**: `bg-[#F5A623]/20` - Warning state backgrounds
  - **color-error-default**: `bg-[#FF6B6B]` - False claims, critical alerts
  - **color-error-light**: `bg-[#FF6B6B]/20` - Error state backgrounds
  - **color-info-default**: `bg-[#4ECDC4]` - Cool cyan for neutral facts, information
  - **color-info-light**: `bg-[#4ECDC4]/20` - Info state backgrounds

### Accent Colors
  - **accent-pink**: `text-[#FF8FB3]` or `bg-[#FF8FB3]` - Soft pink for reviews, opinions, community content
  - **accent-pink-dim**: `bg-[#FF8FB3]/20` - Subtle pink backgrounds
  - **accent-cyan**: `text-[#4ECDC4]` or `bg-[#4ECDC4]` - Cool cyan for neutral analysis
  - **accent-gold**: `text-[#F5A623]` or `bg-[#F5A623]` - Warm gold for priority content

### Data Visualization Charts
  - Credibility scale: `#FF6B6B` (low) → `#F5A623` (medium) → `#10D97E` (high)
  - Fact categories: `#10D97E`, `#4ECDC4`, `#F5A623`, `#FF8FB3`, `#8B9DC3`
  - Neutral data colors: `#3F4349`, `#565B63`, `#6D737D`, `#858B97`, `#9DA3B1`

## Typography
- **Font Stack**:
  - **font-family-base**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif` — Clean sans-serif for optimal readability

- **Font Size & Weight**:
  - **Caption**: `text-xs font-normal` (12px / 400) - Timestamps, metadata, bottom navigation labels
  - **Body Small**: `text-sm font-normal` (14px / 400) - Supporting text, descriptions
  - **Body Default**: `text-base font-normal` (16px / 400) - Main content, fact descriptions
  - **Body Emphasized**: `text-base font-semibold` (16px / 600) - Emphasized facts, key information
  - **Card Title**: `text-lg font-semibold` (18px / 600) - Card headers, section titles
  - **Page Title**: `text-xl font-bold` (20px / 700) - Page headers, main headlines
  - **Fact Headline**: `text-2xl font-bold` (24px / 700) - Featured facts, breaking news
  - **Display**: `text-3xl font-bold` (30px / 700) - Hero sections, major statements

- **Line Height**: 1.5

## Border Radius
  - **Small**: 8px — Buttons, tags, small interactive elements
  - **Medium**: 12px — Input fields, badges, credibility indicators
  - **Large**: 16px — Content cards, fact containers
  - **Extra Large**: 20px — Hero cards, featured content
  - **Full**: full — Avatars, status indicators, circular badges

## Layout & Spacing
  - **Spacing Scale**:
  - **Base Unit**: 4px
  - **Tight**: 8px - Icon-text pairs, inline elements
  - **Compact**: 12px - List items, compact card content
  - **Standard**: 16px - Card padding, section spacing
  - **Comfortable**: 24px - Major section separation
  - **Spacious**: 32px - Page-level content blocks

## Create Boundaries
Boundaries are created primarily through surface color differentiation with reduced contrast for dark mode comfort. Subtle glows and shadows provide elevation hierarchy without harsh contrasts.

### Borders
  - **Case 1**: No borders for most elements - rely on surface color contrast
  - **Case 2**: Subtle borders when needed for definition
    - **Default**: `border border-white/10` - Subtle separation on inputs, dividers
    - **Emphasized**: `border border-white/20` - Active states, focused elements
    - **Accent**: `border border-[#10D97E]/40` - Primary-colored borders for verification badges

### Dividers
  - **Default**: `border-t border-white/10` or `border-b border-white/10` - Subtle content separation
  - **Emphasized**: `border-t border-white/15` - Stronger section breaks

### Shadows & Effects
  - **Case 1 (Flat)**: No shadow for standard cards and containers - rely on background color layering
  - **Case 2 (Subtle Glow)**: `shadow-[0_0_16px_rgba(16,217,126,0.15)]` - Emerald glow for verified facts, primary CTAs
  - **Case 3 (Elevated)**: `shadow-[0_4px_24px_rgba(0,0,0,0.4)]` - Modals, floating action buttons
  - **Case 4 (Accent Glow)**: `shadow-[0_0_20px_rgba(78,205,196,0.15)]` - Cyan glow for featured neutral content
  - **Backdrop Blur**: `backdrop-blur-md` - For semi-transparent headers and overlays

## Assets
### Image
- For normal `<img>`: `object-cover brightness-90 contrast-90`
- For `<img>` with:
  - Slight overlay: `object-cover brightness-80 contrast-90`
  - Heavy overlay: `object-cover brightness-60 contrast-95`

### Icon
- Use Lucide icons from Iconify for their clean, modern outline style.
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
  - **Graphic**: Use a simple, relevant icon (e.g., a `shield-check` icon for verification, a `sparkles` icon for facts).

## Page Layout - Mobile
```html
<!-- Mobile Layout Template: Adjust body width (w-[390px]) based on target device -->
<body class="w-[390px] min-h-[844px] bg-[#1A1D23] font-['-apple-system','BlinkMacSystemFont','Segoe_UI','Roboto','Helvetica_Neue','Arial',sans-serif] leading-[1.5]">

  <!-- Top Fixed Header: Contains status bar safe area and navigation bar -->
  <div class="z-10 fixed top-0 w-full bg-[#1A1D23]/95 backdrop-blur-md">
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
    <div class="h-14"></div>
  </div>

  <!-- Bottom Fixed Area: Contains FAB and/or bottom navigation and/or bottom toolbar and/or bottom input dialog and bottom safe area -->
  <div class="z-10 fixed bottom-0 w-full flex flex-col">

    <!-- Floating Action Button (Optional): Remove entire div if not needed -->
    <div class="flex flex-col gap-4 px-4 pb-6 items-end">
      <button class="w-14 h-14 flex items-center justify-center bg-[#10D97E] rounded-full shadow-[0_0_16px_rgba(16,217,126,0.15)]">
        <!-- FAB content: icon only, no text -->
      </button>
    </div>

    <!-- Bottom bar(container) for Navigation/Toolbar/Input Field (bg and safe area) (Optional): Remove entire div if not needed -->
    <div class="bg-[#23272F] border-t border-white/10">
      <!-- Bottom Navigation/Toolbar/Input Field(layout) -->
      <nav class="flex justify-around py-3 px-1">
        <!-- Navigation Item: flex-1; text-white/50(Default); text-white/95(Active) -->
        <div class="flex flex-1 flex-col items-center gap-1">
            <div class="w-6 h-6 flex items-center justify-center">
                <iconify-icon icon="lucide:home" class="text-xl text-white/50"></iconify-icon>
            </div>
            <span class="text-xs font-normal text-white/50">Home</span>
        </div>
        <!-- Center FAB in Navigation (Optional): Remove entire div if not needed -->
        <div class="flex flex-1 flex-col items-center">
            <button class="w-12 h-12 flex items-center justify-center bg-[#10D97E] rounded-full shadow-[0_0_16px_rgba(16,217,126,0.15)]">
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
  - Example 1 (Primary filled button):
    - button: `flex items-center justify-center gap-2 px-6 py-3 bg-[#10D97E] rounded-lg hover:bg-[#0DB86A] transition`
      - icon (optional)
      - span: `text-base font-semibold text-[#0A0B0D] whitespace-nowrap`
  - Example 2 (Secondary outlined button):
    - button: `flex items-center justify-center gap-2 px-6 py-3 bg-transparent border border-[#10D97E]/40 rounded-lg hover:bg-[#10D97E]/10 transition`
      - icon (optional)
      - span: `text-base font-semibold text-[#10D97E] whitespace-nowrap`
  - Example 3 (Text button):
    - button: `flex items-center gap-2`
      - span: `text-base font-semibold text-[#10D97E] whitespace-nowrap`
  - Example 4 (Icon button):
    - button: `w-10 h-10 flex items-center justify-center bg-[#23272F] rounded-lg hover:bg-[#2B3039] transition`
      - icon

- **Tag Group (Filter Tags)**
  - container(scrollable): `flex gap-2 overflow-x-auto [&::-webkit-scrollbar]:hidden`
    - label (Tag item 1):
      - input: `type="radio" name="factCategory" class="sr-only peer" checked`
      - div: `px-4 py-2 bg-[#23272F] text-white/70 rounded-full peer-checked:bg-[#10D97E] peer-checked:text-[#0A0B0D] hover:bg-[#2B3039] transition whitespace-nowrap text-sm font-semibold`

### Data Entry
- **Progress bars/Sliders**:
  - Credibility meter: `h-2 bg-[#33373F] rounded-full`
    - Fill: `h-2 rounded-full` with gradient based on score (low: `bg-[#FF6B6B]`, medium: `bg-[#F5A623]`, high: `bg-[#10D97E]`)

- **Checkbox**
  - label: `flex items-center gap-3`
    - input: `type="checkbox" class="sr-only peer"`
    - div: `w-5 h-5 bg-[#33373F] rounded-md flex items-center justify-center peer-checked:bg-[#10D97E] text-transparent peer-checked:text-[#0A0B0D] border border-white/10 peer-checked:border-transparent transition`
      - svg(Checkmark): `<svg class="w-3 h-3" viewBox="0 0 12 12" fill="none"><path d="M2 6L5 9L10 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`
    - span: `text-base text-white/95`

- **Radio button**
  - label: `flex items-center gap-3`
    - input: `type="radio" name="option1" class="sr-only peer"`
    - div: `w-5 h-5 bg-[#33373F] rounded-full flex items-center justify-center peer-checked:bg-[#10D97E] border border-white/10 peer-checked:border-transparent transition`
      - svg(dot indicator): `<svg class="w-2.5 h-2.5 text-transparent peer-checked:text-[#0A0B0D]" viewBox="0 0 8 8"><circle cx="4" cy="4" r="3" fill="currentColor"/></svg>`
    - span: `text-base text-white/95`

- **Switch/Toggle**
  - label: `flex items-center gap-3`
    - div: `relative`
      - input: `type="checkbox" class="sr-only peer"`
      - div(Toggle track): `w-12 h-7 bg-[#33373F] rounded-full peer-checked:bg-[#10D97E] transition border border-white/10 peer-checked:border-transparent`
      - div(Toggle thumb): `absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full peer-checked:translate-x-5 transition shadow-[0_2px_8px_rgba(0,0,0,0.3)]`
    - span: `text-base text-white/95`

- **Select/Dropdown**
  - Select container: `flex items-center justify-between px-4 py-3 bg-[#33373F] rounded-lg border border-white/10`
    - text: `text-base text-white/95`
    - Dropdown icon(square container): `w-5 h-5 flex items-center justify-center bg-transparent`
      - icon: `<iconify-icon icon="lucide:chevron-down" class="text-lg text-white/70"></iconify-icon>`

### Container
- **Card**
    - Example 1 (Fact Card - Vertical with credibility badge):
        - Card: `bg-[#23272F] rounded-2xl flex flex-col p-4 gap-3`
        - Credibility Badge: `flex items-center gap-2 px-3 py-1.5 bg-[#10D97E]/20 rounded-full w-fit`
          - icon: `<iconify-icon icon="lucide:shield-check" class="text-base text-[#10D97E]"></iconify-icon>`
          - text: `text-sm font-semibold text-[#10D97E]`
        - Fact headline: `text-lg font-semibold text-white/95`
        - Description: `text-sm text-white/70`
        - Metadata: `flex items-center gap-3 text-xs text-white/50`
    - Example 2 (Review Card - Horizontal with avatar):
        - Card: `bg-[#23272F] rounded-2xl flex gap-3 p-4`
        - Avatar: `w-12 h-12 rounded-full bg-[#33373F]`
        - Content area: `flex flex-col gap-2 flex-1`
          - User name: `text-base font-semibold text-white/95`
          - Review text: `text-sm text-white/70`
          - Timestamp: `text-xs text-white/50`
    - Example 3 (Featured Fact - Image-focused with glow):
        - Card: `flex flex-col gap-3 shadow-[0_0_16px_rgba(16,217,126,0.15)]`
        - Image: `rounded-2xl w-full object-cover brightness-90 contrast-90`
        - Text area: `flex flex-col gap-2`
          - Category badge: `px-3 py-1 bg-[#F5A623]/20 text-[#F5A623] text-xs font-semibold rounded-full w-fit`
          - Headline: `text-xl font-bold text-white/95`
          - Source: `text-sm text-white/50`
    - Example 4 (Credibility Summary Card):
        - Card: `bg-[#23272F] rounded-2xl flex flex-col gap-4 p-4`
        - Score Display: `flex items-center justify-between`
          - Label: `text-base font-semibold text-white/95`
          - Score: `text-2xl font-bold text-[#10D97E]`
        - Progress bar section
        - Details: `text-sm text-white/70`

- **List** (for scrollable lists, settings, etc.)
  - List container: `space-y-1`
    - list-item: `flex items-center justify-between py-4 hover:bg-[#23272F] transition rounded-lg px-2`
      - Left content: `flex items-center gap-3`
        - icon-container: `w-10 h-10 flex items-center justify-center bg-[#23272F] rounded-lg`
          - icon: `<iconify-icon icon="lucide:bookmark" class="text-xl text-white/70"></iconify-icon>`
        - text area: `flex flex-col gap-0.5`
          - title: `text-base font-semibold text-white/95`
          - subtitle: `text-sm text-white/50`
      - Right content: `flex items-center gap-2`
        - badge (if applicable): `px-2 py-1 bg-[#10D97E]/20 text-[#10D97E] text-xs font-semibold rounded-full`
        - chevron-icon: `w-5 h-5 flex items-center justify-center`
          - icon: `<iconify-icon icon="lucide:chevron-right" class="text-lg text-white/30"></iconify-icon>`

## Additional Notes
- **Dark Mode Optimization**: All color choices prioritize reduced eye strain with carefully balanced contrast ratios (minimum WCAG AA compliance)
- **Credibility Visual Language**: Emerald green consistently represents verified/high credibility, warm gold for urgent priority, cool cyan for neutral analysis, soft pink for community opinions
- **Glow Effects**: Use sparingly on verified content and primary CTAs to create subtle hierarchy without harsh shadows
- **Fact Emphasis**: Bold typography and bright emerald accents draw immediate attention to verified information
- **Transparency Through Design**: Clear visual indicators (badges, progress bars, color coding) make credibility assessment immediate and intuitive

<colors_extraction>
#10D97E
#3EE598
#0DB86A
#10D97E33
#10D97ECC
#1A1D23
#23272F
#2B3039
#33373F
#1A1D23F2
#FFFFFFF2
#FFFFFFB3
#FFFFFF80
#FFFFFF4D
#0A0B0D
#0DB86A
#0DB86A33
#F5A623
#F5A62333
#FF6B6B
#FF6B6B33
#4ECDC4
#4ECDC433
#FF8FB3
#FF8FB333
#3F4349
#565B63
#6D737D
#858B97
#9DA3B1
#8B9DC3
#EFD46E
#FFFFFF1A
#FFFFFF33
#10D97E66
#FFFFFF26
rgba(16,217,126,0.15)
rgba(0,0,0,0.4)
rgba(78,205,196,0.15)
rgba(0,0,0,0.3)
</colors_extraction>
