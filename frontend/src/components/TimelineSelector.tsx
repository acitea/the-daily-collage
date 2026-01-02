import { ScrollArea, Text, SegmentedControl, Group } from '@mantine/core';
import { useMemo, useState } from 'react';
interface TimelinePoint {
  timestamp: string;
  date: Date;
  label: string;
  type: 'day' | 'week' | 'month' | 'interval';
}

interface TimelineSelectorProps {
  selectedTimestamp: string | null;
  onTimestampSelect: (timestamp: string | null) => void;
  currentTimestamp: string;
}

export const TimelineSelector = ({
  selectedTimestamp,
  onTimestampSelect,
  currentTimestamp,
}: TimelineSelectorProps) => {
  const [weekStart, setWeekStart] = useState<'sunday' | 'monday'>('sunday');

  // Helper function to get ISO week number
  const getWeekNumber = (date: Date): number => {
    const tempDate = new Date(date);
    tempDate.setHours(0, 0, 0, 0);
    tempDate.setDate(tempDate.getDate() + 4 - (tempDate.getDay() || 7));
    const yearStart = new Date(tempDate.getFullYear(), 0, 1);
    const weekNum = Math.ceil(((tempDate.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
    return weekNum;
  };

  const timelinePoints = useMemo(() => {
    const points: TimelinePoint[] = [];
    const now = new Date();

    // Generate timeline: last 30 days with day/week/month markers
    // Also include 6-hour interval tickers between dates
    // Going backwards from oldest to newest
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      date.setHours(0, 0, 0, 0);

      const dayOfMonth = date.getDate();
      let dayOfWeek = date.getDay();

      // Adjust day of week based on selected week start
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

      // Add 6-hour interval tickers (0.25, 0.5, 0.75 days forward)
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
  }, [weekStart]);

  const getTickHeight = (type: 'day' | 'week' | 'month' | 'interval') => {
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

  const getTickWidth = (type: 'day' | 'week' | 'month' | 'interval') => {
    switch (type) {
      case 'month':
        return 'w-2';
      case 'week':
        return 'w-1.5';
      case 'interval':
        return 'w-0.5';
      default:
        return 'w-1';
    }
  };

  const getHoverTickWidth = (type: 'day' | 'week' | 'month' | 'interval') => {
    switch (type) {
      case 'month':
        return 'hover:w-3';
      case 'week':
        return 'hover:w-2.5';
      case 'interval':
        return 'hover:w-1.5';
      default:
        return 'hover:w-2';
    }
  };

  return (
    <div className="w-full border-t border-b border-gray-300 bg-white py-4 px-4">
      <div className="max-w-4xl mx-auto">
        <Group justify="space-between" align="center" mb="md">
          <Text size="sm" fw={600} c="dark" className="">
            Timeline (Last 30 Days)
          </Text>
          <SegmentedControl
            value={weekStart}
            onChange={(value) => setWeekStart(value as 'sunday' | 'monday')}
            data={[
              { label: 'Sun', value: 'sunday' },
              { label: 'Mon', value: 'monday' },
            ]}
            size="xs"
          />
        </Group>

        <ScrollArea>
          <div className="flex gap-1 pb-4 min-w-max">
            {timelinePoints.map((point, idx) => {
              const isSelected = selectedTimestamp === point.timestamp;
              const isCurrent = point.timestamp === currentTimestamp;
              const dayOfWeekName = point.date.toLocaleDateString('en-US', { weekday: 'short' });
              const isInterval = point.type === 'interval';

              // Determine text color based on state - using Mantine color system
              const labelColor = isSelected ? 'black' : 'dark';
              const fontWeight = isSelected ? 700 : 400;

              return (
                <div
                  key={idx}
                  className="flex flex-col items-center justify-between cursor-pointer min-h-24"
                  onClick={() => onTimestampSelect(point.timestamp)}
                  title={point.date.toLocaleDateString()}
                >
                  {!isInterval && (
                    <>
                      {/* Row 1: Month/Week Label */}
                      <div className="h-4 flex items-center">
                        {(point.type === 'week' || point.type === 'month') && (
                          <Text
                            size="xs"
                            fw={point.type === 'month' ? 600 : 500}
                            c={labelColor}
                            className="whitespace-nowrap"
                          >
                            {point.label}
                          </Text>
                        )}
                      </div>

                      {/* Row 2: Day of Week (always shown, muted) */}
                      <div className="h-4 flex items-center">
                        <Text 
                          size="xs" 
                          fw={fontWeight}
                          c="dimmed"
                          className="whitespace-nowrap"
                        >
                          {dayOfWeekName}
                        </Text>
                      </div>
                    </>
                  )}
                  {isInterval && (
                    <>
                      {/* Empty space for Row 1 */}
                      <div className="h-4" />
                      {/* Empty space for Row 2 */}
                      <div className="h-4" />
                    </>
                  )}

                  {/* Row 3: Tick Bar */}
                  <div
                    className={`
                      ${getTickWidth(point.type)} 
                      ${getTickHeight(point.type)} 
                      rounded-sm
                      transition-all 
                      duration-150
                      ${
                        isSelected
                          ? 'bg-black opacity-100 shadow-sm'
                          : isCurrent
                            ? 'bg-gray-800 opacity-80'
                            : isInterval
                              ? 'bg-gray-300 opacity-50'
                              : 'bg-gray-400 opacity-60'
                      }
                      ${getHoverTickWidth(point.type)}
                      hover:opacity-100
                      hover:bg-gray-900
                      hover:shadow-md
                    `}
                  />

                  {!isInterval && (
                    <>
                      {/* Row 4: Date (always shown, bottom-aligned) */}
                      <div className="h-4 flex items-end">
                        <Text 
                          size="xs" 
                          fw={fontWeight} 
                          c={labelColor}
                          className="whitespace-nowrap"
                        >
                          {point.date.getDate()}
                        </Text>
                      </div>
                    </>
                  )}
                  {isInterval && (
                    <>
                      {/* Empty space for Row 4 */}
                      <div className="h-4" />
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </ScrollArea>

        <Text size="xs" c="dimmed" className="mt-3">
          Click a tick to view historical snapshot • Current view updates every 6 hours
        </Text>
      </div>
    </div>
  );
};
