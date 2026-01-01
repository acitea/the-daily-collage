export interface Hitbox {
  x: number;
  y: number;
  w: number;
  h: number;
  category: string;
  tag: string;
  articles: Article[];
}

export interface Article {
  title: string;
  url: string;
  source: string;
  published_at?: string;
}

export interface SignalData {
  category: string;
  score: number;
  tag: string;
  articles: Article[];
}

export interface VibeResponse {
  location: string;
  timestamp: string;
  time_window: string;
  image_url: string;
  hitboxes: Hitbox[];
  signals: SignalData[];
  cached: boolean;
}

export interface Location {
  id: string;
  name: string;
  country: string;
}

export interface TimelinePoint {
  timestamp: string;
  date: Date;
  label: string;
  type: 'day' | 'week' | 'month';
}
