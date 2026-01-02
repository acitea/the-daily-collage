import { ScrollArea, Text } from '@mantine/core';
import { useMemo } from 'react';
import type { TimelinePoint } from '../types/vibe';

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
  const timelinePoints = useMemo(() => {
    const points: TimelinePoint[] = [];
    const now = new Date();

    // Generate timeline: last 30 days with day/week/month markers
    // Going backwards from oldest to newest
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      date.setHours(0, 0, 0, 0);

      const dayOfMonth = date.getDate();
      const dayOfWeek = date.getDay();

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
              ? `Week ${Math.ceil(dayOfMonth / 7)}`
              : date.getDate().toString(),
        type,
      });
    }

    return points;
  }, []);

  const getTickHeight = (type: 'day' | 'week' | 'month') => {
    switch (type) {
      case 'month':
        return 'h-12';
      case 'week':
        return 'h-10';
      default:
        return 'h-8';
    }
  };

  const getTickWidth = (type: 'day' | 'week' | 'month') => {
    switch (type) {
      case 'month':
        return 'w-2';
      case 'week':
        return 'w-1.5';
      default:
        return 'w-1';
    }
  };

  return (
    <div className="w-full border-t border-b border-gray-300 bg-white py-4 px-4">
      <div className="max-w-4xl mx-auto">
        <Text size="sm" fw={600} mb="md" className="text-gray-700">
          Timeline (Last 30 Days)
        </Text>

        <ScrollArea>
          <div className="flex gap-3 pb-4 min-w-max">
            {timelinePoints.map((point, idx) => {
              const isSelected = selectedTimestamp === point.timestamp;
              const isCurrent = point.timestamp === currentTimestamp;

              return (
                <div
                  key={idx}
                  className="flex flex-col items-center cursor-pointer"
                  onClick={() => onTimestampSelect(point.timestamp)}
                  title={point.date.toLocaleDateString()}
                >
                  {/* Fixed label height area - ensures all ticks align to same baseline */}
                  <div className="h-5 flex items-center mb-2">
                    {(point.type === 'week' || point.type === 'month') && (
                      <Text
                        size="xs"
                        fw={point.type === 'month' ? 600 : 500}
                        className={`whitespace-nowrap ${
                          isSelected
                            ? 'text-black'
                            : isCurrent
                              ? 'text-gray-700'
                              : 'text-gray-500'
                        }`}
                      >
                        {point.label}
                      </Text>
                    )}
                  </div>

                  {/* Tick Bar - All aligned at same vertical level */}
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
                            : 'bg-gray-400 opacity-60'
                      }
                      hover:w-3
                      hover:opacity-100
                      hover:bg-gray-900
                      hover:shadow-md
                    `}
                  />

                  {/* Day label - Only for day type, positioned below */}
                  {point.type === 'day' && (
                    <div className="h-5 flex items-center mt-2">
                      <Text
                        size="xs"
                        fw={400}
                        className={`whitespace-nowrap ${
                          isSelected ? 'text-black' : 'text-gray-500'
                        }`}
                      >
                        {point.label}
                      </Text>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </ScrollArea>

        <Text size="xs" c="dimmed" className="mt-3 text-gray-600">
          Click a tick to view historical snapshot • Current view updates every 6 hours
        </Text>
      </div>
    </div>
  );
};
