export interface TimelinePoint {
  timestamp: string;
  date: Date;
  label: string;
  type: 'day' | 'week' | 'month' | 'interval';
}

export const getWeekNumber = (date: Date): number => {
  const tempDate = new Date(date);
  tempDate.setHours(0, 0, 0, 0);
  tempDate.setDate(tempDate.getDate() + 4 - (tempDate.getDay() || 7));
  const yearStart = new Date(tempDate.getFullYear(), 0, 1);
  const weekNum = Math.ceil(((tempDate.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return weekNum;
};

export const generateTimelinePoints = (weekStart: 'sunday' | 'monday'): TimelinePoint[] => {
  const points: TimelinePoint[] = [];
  const now = new Date();

  for (let i = 30; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    date.setHours(0, 0, 0, 0);

    const dayOfMonth = date.getDate();
    let dayOfWeek = date.getDay();

    if (weekStart === 'monday') {
      dayOfWeek = (dayOfWeek + 6) % 7;
    }

    let type: 'day' | 'week' | 'month' = 'day';
    if (dayOfMonth === 1) {
      type = 'month';
    } else if (dayOfWeek === 0) {
      type = 'week';
    }

    points.push({
      timestamp: date.toISOString(),
      date,
      label:
        type === 'month'
          ? date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
          : type === 'week'
            ? `W${getWeekNumber(date)}`
            : date.getDate().toString(),
      type,
    });

    // Add 6-hour interval tickers
    for (let hour = 6; hour < 24; hour += 6) {
      const intervalDate = new Date(date);
      intervalDate.setHours(hour, 0, 0, 0);

      points.push({
        timestamp: intervalDate.toISOString(),
        date: intervalDate,
        label: '',
        type: 'interval',
      });
    }
  }

  return points;
};

export const getTickHeight = (type: 'day' | 'week' | 'month' | 'interval'): string => {
  switch (type) {
    case 'month':
      return 'h-12';
    case 'week':
      return 'h-10';
    case 'interval':
      return 'h-6';
    default:
      return 'h-8';
  }
};

export const getTickWidth = (type: 'day' | 'week' | 'month' | 'interval'): string => {
  switch (type) {
    case 'month':
      return 'w-2';
    case 'week':
      return 'w-1.5';
    case 'interval':
      return 'w-1';
    default:
      return 'w-1';
  }
};

export const getHoverTickWidth = (type: 'day' | 'week' | 'month' | 'interval'): string => {
  switch (type) {
    case 'month':
      return 'hover:w-3';
    case 'week':
      return 'hover:w-2.5';
    case 'interval':
      return 'hover:w-2.5';
    default:
      return 'hover:w-2';
  }
};

export const roundToNearestInterval = (date: Date): Date => {
  const hours = date.getHours();
  const validHours = [0, 6, 12, 18];
  const nearest = validHours.reduce((prev, curr) =>
    Math.abs(curr - hours) < Math.abs(prev - hours) ? curr : prev
  );
  const rounded = new Date(date);
  rounded.setHours(nearest, 0, 0, 0);
  return rounded;
};
