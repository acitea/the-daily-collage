/**
 * Utility for generating cache keys matching the backend's VibeHash format.
 * 
 * Cache key format: {city}_{YYYY-MM-DD}_{HH-HH}
 * Example: stockholm_2026-01-03_12-18
 */

const WINDOW_DURATION_HOURS = 6;

/**
 * Generate a cache key for a given city and timestamp.
 * 
 * The timestamp represents the END of the time window or a point within it.
 * For example, selecting Jan 11 at 06:00 will return the window 00:00-06:00.
 * Selecting Jan 11 at 00:00 will return the previous window (Jan 10 18:00-24:00).
 * 
 * @param city - City name (e.g., 'stockholm')
 * @param timestamp - JavaScript Date object or ISO timestamp string
 * @returns Cache key string (e.g., 'stockholm_2026-01-03_12-18')
 */
export function generateCacheKey(city: string, timestamp: Date | string): string {
  // Parse timestamp if it's a string
  let date = typeof timestamp === 'string' ? new Date(timestamp) : new Date(timestamp);
  
  // Normalize city name
  const cityNormalized = city.toLowerCase().replace(/\s+/g, '_');
  
  const hour = date.getHours();
  const minute = date.getMinutes();
  
  // If we're exactly at a window boundary (hour is divisible by 6 and minutes are 0),
  // we want the PREVIOUS window, not the one starting now
  let windowIndex = Math.floor(hour / WINDOW_DURATION_HOURS);
  
  if (hour % WINDOW_DURATION_HOURS === 0 && minute === 0 && hour !== 0) {
    // At a boundary (06:00, 12:00, 18:00), use the previous window
    windowIndex = windowIndex - 1;
  } else if (hour === 0 && minute === 0) {
    // Special case: midnight (00:00) - use the previous day's last window (18-24)
    date = new Date(date);
    date.setDate(date.getDate() - 1);
    windowIndex = 3; // 18-24 window
  }
  
  // Get date in YYYY-MM-DD format using local time (not UTC)
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const dateStr = `${year}-${month}-${day}`;
  
  const windowStart = windowIndex * WINDOW_DURATION_HOURS;
  const windowEnd = (windowIndex + 1) * WINDOW_DURATION_HOURS;  
  
  // Format window string (e.g., "00-06", "12-18")
  const windowStr = `${String(windowStart).padStart(2, '0')}-${String(windowEnd).padStart(2, '0')}`;
  
  // Combine parts
  return `${cityNormalized}_${dateStr}_${windowStr}`;
}

/**
 * Get the most recent completed window cache key for a city.
 * 
 * Returns the previous window since the current window hasn't completed yet.
 * For example, if it's 14:00 (in the 12-18 window), returns the 06-12 window.
 */
export function getCurrentCacheKey(city: string): string {
  const now = new Date();
  const hour = now.getHours();
  
  // Calculate which window we're currently in
  const currentWindowIndex = Math.floor(hour / WINDOW_DURATION_HOURS);
  
  // Get the previous completed window
  const targetDate = new Date(now);
  let targetWindowIndex = currentWindowIndex - 1;
  
  // If we're in the first window (00-06), get yesterday's last window (18-24)
  if (targetWindowIndex < 0) {
    targetDate.setDate(targetDate.getDate() - 1);
    targetWindowIndex = 3; // 18-24 window
  }
  
  // Set the time to the start of the target window
  targetDate.setHours(targetWindowIndex * WINDOW_DURATION_HOURS, 0, 0, 0);
  
  return generateCacheKey(city, targetDate);
}
