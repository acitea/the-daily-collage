import { ScrollArea, Text, Group, Badge } from '@mantine/core';
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
    for (let i = 0; i <= 30; i++) {
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
        label: type === 'month' 
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
        return 'h-8';
      case 'week':
        return 'h-6';
      default:
        return 'h-4';
    }
  };

  const getTickWidth = (type: 'day' | 'week' | 'month') => {
    switch (type) {
      case 'month':
        return 'w-1';
      case 'week':
        return 'w-0.5';
      default:
        return 'w-px';
    }
  };

  return (
    <div className="border-b border-gray-200 bg-gray-50 py-4">
      <Group justify="space-between" className="px-4 mb-2">
        <Text size="sm" fw={500}>
          Timeline
        </Text>
        <Group gap="xs">
          <Badge
            color={selectedTimestamp === null ? 'blue' : 'gray'}
            variant={selectedTimestamp === null ? 'filled' : 'light'}
            className="cursor-pointer"
            onClick={() => onTimestampSelect(null)}
          >
            Current
          </Badge>
        </Group>
      </Group>
      
      <ScrollArea className="px-4">
        <div className="flex items-end gap-1 pb-2 min-w-max">
          {timelinePoints.map((point, idx) => {
            const isSelected = selectedTimestamp === point.timestamp;
            const isCurrent = point.timestamp === currentTimestamp;
            
            return (
              <div
                key={idx}
                className="flex flex-col items-center gap-1 cursor-pointer hover:opacity-70 transition-opacity"
                onClick={() => onTimestampSelect(point.timestamp)}
                title={point.date.toLocaleDateString()}
              >
                <div
                  className={`${getTickWidth(point.type)} ${getTickHeight(point.type)} ${
                    isSelected
                      ? 'bg-blue-600'
                      : isCurrent
                      ? 'bg-green-600'
                      : 'bg-gray-400'
                  }`}
                />
                {point.type !== 'day' && (
                  <Text
                    size="xs"
                    c={isSelected ? 'blue' : 'dimmed'}
                    fw={point.type === 'month' ? 600 : 400}
                  >
                    {point.label}
                  </Text>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
      
      <Text size="xs" c="dimmed" className="px-4 mt-2">
        Click a tick to view historical snapshot • Current view updates every 6 hours
      </Text>
    </div>
  );
};
