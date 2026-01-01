import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { API_BASE_URL, ENDPOINTS } from '../config/api';
import type { VibeResponse, Location } from '../types/vibe';

export const useCurrentVibe = (location: string) => {
  return useQuery<VibeResponse>({
    queryKey: ['vibe', 'current', location],
    queryFn: async () => {
      const { data } = await axios.get(
        `${API_BASE_URL}${ENDPOINTS.currentVibe(location)}`
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
      const { data } = await axios.get(
        `${API_BASE_URL}${ENDPOINTS.historicalVibe(location, timestamp!)}`
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
      const { data } = await axios.get(`${API_BASE_URL}${ENDPOINTS.locations}`);
      return data;
    },
  });
};
