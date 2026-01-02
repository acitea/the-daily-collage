/**
 * Mock data for testing the frontend without a backend.
 * Provides realistic vibe data for 31 Dec 0000 and 1 Jan 1200.
 */

import type { VibeResponse } from '../types/vibe';

export const mockVibes: Record<string, VibeResponse> = {
  '0000-12-31T00:00:00Z': {
    location: 'Stockholm',
    timestamp: '0000-12-31T00:00:00Z',
    time_window: '0000-12-31 00:00 to 0000-12-31 05:59',
    image_url: 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22800%22 height=%22600%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22800%22 height=%22600%22/%3E%3Crect fill=%22%2387ceeb%22 width=%22800%22 height=%22300%22/%3E%3Crect fill=%22%23d4af37%22 y=%22300%22 width=%22800%22 height=%22150%22/%3E%3Crect fill=%22%23228b22%22 y=%22450%22 width=%22800%22 height=%22150%22/%3E%3Ccircle cx=%22700%22 cy=%22100%22 r=%2240%22 fill=%22%23ffd700%22/%3E%3Crect x=%22100%22 y=%22400%22 width=%2250%22 height=%22100%22 fill=%22%23654321%22/%3E%3Crect x=%22100%22 y=%22380%22 width=%2250%22 height=%2220%22 fill=%22%23a52a2a%22/%3E%3Crect x=%22200%22 y=%22420%22 width=%2240%22 height=%2280%22 fill=%22%23696969%22/%3E%3Crect x=%22200%22 y=%22400%22 width=%2240%22 height=%2220%22 fill=%22%23a52a2a%22/%3E%3Crect x=%22300%22 y=%22410%22 width=%2260%22 height=%2290%22 fill=%22%23808080%22/%3E%3Crect x=%22300%22 y=%22390%22 width=%2260%22 height=%2220%22 fill=%22%23a52a2a%22/%3E%3C/svg%3E',
    cached: true,
    hitboxes: [
      {
        x: 700,
        y: 60,
        w: 80,
        h: 80,
        category: 'weather_temp',
        tag: 'sunny',
        articles: [
          {
            title: 'New Year\'s Eve Weather: Clear skies expected',
            url: 'https://example.com/article1',
            source: 'Swedish News Agency',
            published_at: '0000-12-31T08:00:00Z'
          }
        ]
      },
      {
        x: 100,
        y: 380,
        w: 50,
        h: 120,
        category: 'politics',
        tag: 'government_building',
        articles: [
          {
            title: 'Government to announce policies in New Year',
            url: 'https://example.com/article2',
            source: 'Stockholm Times',
            published_at: '0000-12-30T15:30:00Z'
          }
        ]
      },
      {
        x: 200,
        y: 400,
        w: 40,
        h: 100,
        category: 'transportation',
        tag: 'traffic_light',
        articles: [
          {
            title: 'New Year traffic management plan in effect',
            url: 'https://example.com/article3',
            source: 'Transport Authority',
            published_at: '0000-12-31T10:00:00Z'
          }
        ]
      },
      {
        x: 300,
        y: 390,
        w: 60,
        h: 110,
        category: 'festivals',
        tag: 'celebration',
        articles: [
          {
            title: 'New Year\'s Eve celebrations planned across Stockholm',
            url: 'https://example.com/article4',
            source: 'Events Stockholm',
            published_at: '0000-12-31T12:00:00Z'
          }
        ]
      }
    ],
    signals: [
      {
        category: 'emergencies',
        score: 0.1,
        tag: 'none',
        articles: []
      },
      {
        category: 'crime',
        score: 0.2,
        tag: 'minor',
        articles: [
          {
            title: 'Holiday period sees slight increase in petty crime',
            url: 'https://example.com/crime1',
            source: 'Police Report',
            published_at: '0000-12-31T09:00:00Z'
          }
        ]
      },
      {
        category: 'festivals',
        score: 0.95,
        tag: 'new_year_eve',
        articles: [
          {
            title: 'Stockholm prepares for massive New Year celebrations',
            url: 'https://example.com/festival1',
            source: 'Events Stockholm',
            published_at: '0000-12-31T12:00:00Z'
          },
          {
            title: 'Record attendance expected for New Year\'s Eve',
            url: 'https://example.com/festival2',
            source: 'Tourism Board',
            published_at: '0000-12-31T11:30:00Z'
          }
        ]
      },
      {
        category: 'transportation',
        score: 0.6,
        tag: 'heavy_traffic',
        articles: [
          {
            title: 'New Year traffic expected to peak at 6 PM',
            url: 'https://example.com/transport1',
            source: 'Traffic Authority',
            published_at: '0000-12-31T10:00:00Z'
          }
        ]
      },
      {
        category: 'weather_temp',
        score: 0.8,
        tag: 'cold',
        articles: [
          {
            title: 'Clear and cold weather perfect for New Year celebrations',
            url: 'https://example.com/weather1',
            source: 'SMHI Weather Service',
            published_at: '0000-12-31T08:00:00Z'
          }
        ]
      },
      {
        category: 'weather_wet',
        score: -0.95,
        tag: 'no_rain',
        articles: []
      },
      {
        category: 'sports',
        score: 0.3,
        tag: 'hockey_league',
        articles: [
          {
            title: 'SHL hockey matches scheduled for New Year week',
            url: 'https://example.com/sports1',
            source: 'Sports News',
            published_at: '0000-12-30T14:00:00Z'
          }
        ]
      },
      {
        category: 'economics',
        score: 0.1,
        tag: 'stable',
        articles: [
          {
            title: 'Stock market closed for New Year holidays',
            url: 'https://example.com/econ1',
            source: 'Financial Times',
            published_at: '0000-12-31T16:00:00Z'
          }
        ]
      },
      {
        category: 'politics',
        score: 0.4,
        tag: 'new_year_address',
        articles: [
          {
            title: 'Prime Minister to deliver New Year address',
            url: 'https://example.com/politics1',
            source: 'Government News',
            published_at: '0000-12-31T13:00:00Z'
          }
        ]
      }
    ]
  },

  '1200-01-01T12:00:00Z': {
    location: 'Stockholm',
    timestamp: '1200-01-01T12:00:00Z',
    time_window: '1200-01-01 12:00 to 1200-01-01 17:59',
    image_url: 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22800%22 height=%22600%22%3E%3Crect fill=%22%23b0b0b0%22 width=%22800%22 height=%22600%22/%3E%3Crect fill=%22%238b8b8b%22 width=%22800%22 height=%22300%22/%3E%3Crect fill=%22%23d4a574%22 y=%22300%22 width=%22800%22 height=%22150%22/%3E%3Crect fill=%22%23556b2f%22 y=%22450%22 width=%22800%22 height=%22150%22/%3E%3Ccircle cx=%22700%22 cy=%22120%22 r=%2235%22 fill=%22%23e0e0e0%22 opacity=%220.7%22/%3E%3Crect x=%22120%22 y=%22370%22 width=%2260%22 height=%22130%22 fill=%22%23444444%22/%3E%3Crect x=%22120%22 y=%22350%22 width=%2260%22 height=%2220%22 fill=%22%23d4a574%22/%3E%3Crect x=%22220%22 y=%22390%22 width=%2250%22 height=%22110%22 fill=%22%23555555%22/%3E%3Crect x=%22220%22 y=%22370%22 width=%2250%22 height=%2220%22 fill=%22%23d4a574%22/%3E%3Crect x=%22330%22 y=%22380%22 width=%2270%22 height=%22120%22 fill=%22%23666666%22/%3E%3Crect x=%22330%22 y=%22360%22 width=%2270%22 height=%2220%22 fill=%22%23d4a574%22/%3E%3Cpath d=%22M 450 200 L 460 250 L 470 200 L 480 250 L 490 200 L 500 250 L 510 200%22 stroke=%22%23999999%22 stroke-width=%222%22 fill=%22none%22/%3E%3C/svg%3E',
    cached: false,
    hitboxes: [
      {
        x: 700,
        y: 85,
        w: 70,
        h: 70,
        category: 'weather_temp',
        tag: 'overcast',
        articles: [
          {
            title: 'Medieval January: Overcast skies dominate',
            url: 'https://example.com/article5',
            source: 'Historical Weather Records',
            published_at: '1200-01-01T11:00:00Z'
          }
        ]
      },
      {
        x: 120,
        y: 350,
        w: 60,
        h: 150,
        category: 'politics',
        tag: 'castle',
        articles: [
          {
            title: 'Royal court convenes after New Year recess',
            url: 'https://example.com/article6',
            source: 'Medieval Chronicles',
            published_at: '1200-01-01T09:00:00Z'
          }
        ]
      },
      {
        x: 220,
        y: 370,
        w: 50,
        h: 130,
        category: 'emergencies',
        tag: 'fire_watch',
        articles: [
          {
            title: 'Winter fire watches established in wooden districts',
            url: 'https://example.com/article7',
            source: 'Town Guard',
            published_at: '1200-01-01T08:00:00Z'
          }
        ]
      },
      {
        x: 330,
        y: 360,
        w: 70,
        h: 140,
        category: 'festivals',
        tag: 'epiphany',
        articles: [
          {
            title: 'Epiphany celebrations continue throughout the week',
            url: 'https://example.com/article8',
            source: 'Church Records',
            published_at: '1200-01-01T10:00:00Z'
          }
        ]
      },
      {
        x: 450,
        y: 200,
        w: 60,
        h: 50,
        category: 'weather_wet',
        tag: 'light_snow',
        articles: [
          {
            title: 'Light snow flurries in afternoon',
            url: 'https://example.com/article9',
            source: 'Monastic Records',
            published_at: '1200-01-01T12:00:00Z'
          }
        ]
      }
    ],
    signals: [
      {
        category: 'emergencies',
        score: 0.5,
        tag: 'fire_watch',
        articles: [
          {
            title: 'Increased fire patrols due to winter conditions',
            url: 'https://example.com/emerg1',
            source: 'Town Guard',
            published_at: '1200-01-01T08:00:00Z'
          }
        ]
      },
      {
        category: 'crime',
        score: 0.3,
        tag: 'minor_theft',
        articles: [
          {
            title: 'Winter brings seasonal crime challenges',
            url: 'https://example.com/crime2',
            source: 'Guard Reports',
            published_at: '1200-01-01T07:30:00Z'
          }
        ]
      },
      {
        category: 'festivals',
        score: 0.85,
        tag: 'epiphany',
        articles: [
          {
            title: 'Epiphany feast celebrated in medieval Stockholm',
            url: 'https://example.com/festival3',
            source: 'Church Records',
            published_at: '1200-01-01T10:00:00Z'
          },
          {
            title: 'Three Kings procession through town center',
            url: 'https://example.com/festival4',
            source: 'Historical Records',
            published_at: '1200-01-01T11:30:00Z'
          }
        ]
      },
      {
        category: 'transportation',
        score: 0.2,
        tag: 'slow_travel',
        articles: [
          {
            title: 'Winter roads treacherous for travelers',
            url: 'https://example.com/transport2',
            source: 'Merchant Guild',
            published_at: '1200-01-01T09:00:00Z'
          }
        ]
      },
      {
        category: 'weather_temp',
        score: -0.7,
        tag: 'freezing',
        articles: [
          {
            title: 'Bitter cold grips Stockholm throughout the day',
            url: 'https://example.com/weather2',
            source: 'Monastic Records',
            published_at: '1200-01-01T06:00:00Z'
          }
        ]
      },
      {
        category: 'weather_wet',
        score: 0.4,
        tag: 'light_snow',
        articles: [
          {
            title: 'Snow flurries expected throughout afternoon',
            url: 'https://example.com/weather3',
            source: 'Monastic Records',
            published_at: '1200-01-01T12:00:00Z'
          }
        ]
      },
      {
        category: 'sports',
        score: 0.1,
        tag: 'hunting',
        articles: [
          {
            title: 'Winter hunting season continues',
            url: 'https://example.com/sports2',
            source: 'Hunter\'s Guild',
            published_at: '1200-12-31T14:00:00Z'
          }
        ]
      },
      {
        category: 'economics',
        score: 0.2,
        tag: 'trade_slowdown',
        articles: [
          {
            title: 'Winter trade winds down due to harsh weather',
            url: 'https://example.com/econ2',
            source: 'Merchant Guild',
            published_at: '1200-01-01T08:00:00Z'
          }
        ]
      },
      {
        category: 'politics',
        score: 0.6,
        tag: 'court_sessions',
        articles: [
          {
            title: 'Royal court convenes for winter council',
            url: 'https://example.com/politics2',
            source: 'Court Chronicles',
            published_at: '1200-01-01T09:00:00Z'
          },
          {
            title: 'New year edicts issued by royal decree',
            url: 'https://example.com/politics3',
            source: 'Court Records',
            published_at: '1200-01-01T10:30:00Z'
          }
        ]
      }
    ]
  }
};

export default mockVibes;
