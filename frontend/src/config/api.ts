export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const ENDPOINTS = {
  vibe: (cacheKey: string) => `/api/vibe/${cacheKey}`,
  locations: '/api/supported-locations',
  signalCategories: '/api/signal-categories',
} as const;
