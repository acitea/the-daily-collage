import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { API_BASE_URL, ENDPOINTS } from '../config/api';
import { getCurrentCacheKey, generateCacheKey } from '../utils/vibeHash';
import type { VibeResponse, Location } from '../types/vibe';
import mockVibes from '../mocks/vibeData';

// Enable mock mode by setting this to true or checking an env var
const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === 'true';

export const useCurrentVibe = (location: string) => {
  return useQuery<VibeResponse>({
    queryKey: ['vibe', 'current', location],
    queryFn: async () => {
      if (USE_MOCK_DATA) {
        // Return mock data for current timestamp (31 Dec 0000)
        const mockData = mockVibes['0000-12-31T00:00:00Z'];
        return mockData;
      }
      // Generate cache_key directly from current time
      const cacheKey = getCurrentCacheKey(location);
      
      // Fetch vibe using cache_key
      const { data } = await axios.get(
        `${API_BASE_URL}${ENDPOINTS.vibe(cacheKey)}`
      );
      return data;
    },
    enabled: !!location,
  });
};

export const useHistoricalVibe = (location: string, timestamp: string | null) => {
  return useQuery<VibeResponse>({
    queryKey: ['vibe', 'historical', location, timestamp],
    queryFn: async () => {
      if (USE_MOCK_DATA) {
        // Check if we have mock data for this exact timestamp
        const mockData = mockVibes[timestamp as keyof typeof mockVibes];
        if (mockData) {
          return mockData;
        }
        // Fallback to a default mock vibe
        return mockVibes['0000-12-31T00:00:00Z'];
      }
      // Generate cache_key directly from the provided timestamp
      const cacheKey = generateCacheKey(location, timestamp!);
      
      // Fetch vibe using cache_key
      const { data } = await axios.get(
        `${API_BASE_URL}${ENDPOINTS.vibe(cacheKey)}`
      );
      return data;
    },
    enabled: !!location && !!timestamp,
  });
};

export const useLocations = () => {
  return useQuery<Location[]>({
    queryKey: ['locations'],
    queryFn: async () => {
      if (USE_MOCK_DATA) {
        return [{ name: 'Stockholm', code: 'sthlm' }];
      }
      const { data } = await axios.get(`${API_BASE_URL}${ENDPOINTS.locations}`);
      return data;
    },
  });
};
