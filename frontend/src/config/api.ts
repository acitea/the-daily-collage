export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const ENDPOINTS = {
  currentVibe: (location: string) => `/api/vibe/${location}/current`,
  historicalVibe: (location: string, timestamp: string) => `/api/vibe/${location}/historical?timestamp=${timestamp}`,
  locations: '/api/locations',
} as const;
