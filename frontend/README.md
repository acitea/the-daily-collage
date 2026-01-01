# The Daily Collage - Frontend

Interactive React application for visualizing real-time news vibes.

## Features

- **Interactive Canvas**: Click on elements in the visualization to explore related news articles
- **Timeline Navigation**: Wayback-style timeline to view historical snapshots
- **Location Selector**: Browse vibes from different cities
- **Signal Analysis**: Bar chart and filterable headline list for all news categories
- **Responsive Design**: Built with Mantine UI and Tailwind CSS

## Technology Stack

- **React 19** + **TypeScript**
- **Vite** - Fast build tool
- **Mantine UI** - Component library
- **Tailwind CSS** - Utility-first styling
- **TanStack Query** - Data fetching and caching
- **Recharts** - Data visualization
- **Vitest** - Unit testing

## Setup

1. Install dependencies:
   ```bash
   pnpm install
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env to set VITE_API_BASE_URL
   ```

3. Start development server:
   ```bash
   pnpm dev
   ```

4. Run tests:
   ```bash
   pnpm test
   ```

## Project Structure

```
src/
├── components/          # React components
│   ├── VibeCanvas.tsx       # Interactive image with hitboxes
│   ├── LocationHeader.tsx   # City selector
│   ├── TimelineSelector.tsx # Historical navigation
│   └── SignalsPanel.tsx     # Charts and headlines
├── hooks/               # Custom React hooks
│   └── useVibeData.ts       # API data fetching
├── types/               # TypeScript definitions
│   └── vibe.ts
├── config/              # Configuration
│   └── api.ts
├── test/                # Test files
└── App.tsx              # Main application
```

## Environment Variables

- `VITE_API_BASE_URL` - Backend API URL (default: `http://localhost:8000`)

## API Integration

The frontend expects the following API endpoints:

- `GET /api/vibe/{location}/current` - Current vibe data
- `GET /api/vibe/{location}/historical?timestamp={iso8601}` - Historical snapshot
- `GET /api/locations` - Available locations

### Expected Response Format

```typescript
{
  location: string;
  timestamp: string;
  time_window: string;
  image_url: string;
  cached: boolean;
  hitboxes: Array<{
    x: number;
    y: number;
    w: number;
    h: number;
    category: string;
    tag: string;
    articles: Array<{
      title: string;
      url: string;
      source: string;
      published_at?: string;
    }>;
  }>;
  signals: Array<{
    category: string;
    score: number;
    tag: string;
    articles: Article[];
  }>;
}
```

## Development

### Running the Dev Server

```bash
pnpm dev
```

Access at `http://localhost:5173`

### Building for Production

```bash
pnpm build
```

### Running Tests

```bash
# Run tests in watch mode
pnpm test

# Run tests with UI
pnpm test:ui

# Generate coverage report
pnpm test:coverage
```

### Linting

```bash
pnpm lint
```

## Component Documentation

### VibeCanvas

Displays the vibe visualization image with interactive hitboxes. Clicking a hitbox opens a modal with related articles.

**Props:**
- `imageUrl: string` - URL of the vibe image
- `hitboxes: Hitbox[]` - Array of clickable regions
- `alt?: string` - Image alt text

### LocationHeader

Horizontal scrollable list of city buttons for location selection.

**Props:**
- `locations: Location[]` - Available locations
- `selectedLocation: string` - Currently selected location ID
- `onLocationChange: (id: string) => void` - Selection callback

### TimelineSelector

Wayback-style timeline with day/week/month ticks for navigating historical snapshots.

**Props:**
- `selectedTimestamp: string | null` - Currently selected time
- `onTimestampSelect: (timestamp: string | null) => void` - Selection callback
- `currentTimestamp: string` - Latest available timestamp

### SignalsPanel

Displays signal intensity bar chart and filterable headline list.

**Props:**
- `signals: SignalData[]` - Signal data with articles

## Testing Strategy

- **Unit Tests**: Component rendering and basic interactions (Vitest + React Testing Library)
- **Integration Tests**: Data flow from hooks to components
- **E2E Tests**: (Future) Full user flows with Playwright/Cypress

## Performance Considerations

- **React Query Caching**: Reduces redundant API calls
- **Lazy Loading**: Future enhancement for images
- **Optimistic Updates**: Timeline navigation feels instant
- **CSS-in-JS Minimization**: Tailwind for utilities, Mantine for components

## Future Enhancements

- [ ] E2E tests with Playwright
- [ ] React Router for multi-page navigation
- [ ] Dark mode support
- [ ] Image zoom/pan functionality
- [ ] Export/share functionality
- [ ] Animation transitions between timeline states
- [ ] Mobile-optimized touch interactions

## License

Part of The Daily Collage project.
