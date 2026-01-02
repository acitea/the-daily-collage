import { ScrollArea, Text, SegmentedControl, Group, Button, Stack } from '@mantine/core';
import { useMemo, useState } from 'react';
import { DateTimePickerModal } from './TimelineSelector/DateTimePickerModal';
import {
  generateTimelinePoints,
  getTickHeight,
  getTickWidth,
  getHoverTickWidth,
} from './TimelineSelector/helpers';

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
  const [datePickerOpen, setDatePickerOpen] = useState(false);

  const timelinePoints = useMemo(() => generateTimelinePoints(weekStart), [weekStart]);

  const handleNowClick = () => {
    // Find the most recent (latest) timestamp
    const latestPoint = timelinePoints[timelinePoints.length - 1];
    onTimestampSelect(latestPoint.timestamp);
  };

  return (
    <div className="w-full border-t border-b border-gray-300 bg-white py-4 px-4">
      <div className="max-w-4xl mx-auto">
        <Group justify="space-between" align="flex-end" mb="md">
          <Text size="sm" fw={600} c="dark">
            Timeline (Last 30 Days)
          </Text>
          <Group gap="md" align="flex-end">
            <Button size="xs" variant="light" color="dark" onClick={handleNowClick}>
              Now
            </Button>
            <Button
              size="xs"
              variant="light"
              color="dark"
              onClick={() => setDatePickerOpen(true)}
            >
              Go to Date
            </Button>
            <Stack gap={4} align="center">
              <Text size="xs" c="dimmed" fw={500}>
                Week Start
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
            </Stack>
          </Group>
        </Group>

        {/* Currently Selected Date/Time Display */}
        <div className="mb-4 text-center py-3 bg-gray-50 rounded border border-gray-200">
          <Text size="lg" fw={700} c="dark" className="font-serif">
            {selectedTimestamp
              ? new Date(selectedTimestamp).toLocaleDateString('en-US', {
                  weekday: 'long',
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                }) +
                ' at ' +
                new Date(selectedTimestamp).toLocaleTimeString('en-US', {
                  hour: '2-digit',
                  minute: '2-digit',
                  hour12: false,
                })
              : 'Current View (Latest)'}
          </Text>
        </div>

        <DateTimePickerModal
          opened={datePickerOpen}
          onClose={() => setDatePickerOpen(false)}
          initialValue={selectedTimestamp}
          onSubmit={onTimestampSelect}
        />

        <ScrollArea>
          <div className="flex gap-1 pb-4 min-w-max">
            {timelinePoints.map((point, idx) => {
              const isSelected = selectedTimestamp === point.timestamp;
              const isCurrent = point.timestamp === currentTimestamp;
              const dayOfWeekName = point.date.toLocaleDateString('en-US', { weekday: 'short' });
              const isInterval = point.type === 'interval';

              // Determine text color based on state
              const labelColor = isSelected ? 'black' : 'dark';
              const fontWeight = isSelected ? 700 : 400;

              return (
                <div
                  key={idx}
                  className="flex flex-col items-center justify-between cursor-pointer min-h-24"
                  onClick={() => onTimestampSelect(point.timestamp)}
                  title={point.date.toLocaleDateString() + ' ' + point.date.toLocaleTimeString()}
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

                      {/* Row 2: Day of Week */}
                      <div className="h-4 flex items-center">
                        <Text size="xs" fw={fontWeight} c="dimmed" className="whitespace-nowrap">
                          {dayOfWeekName}
                        </Text>
                      </div>
                    </>
                  )}
                  {isInterval && (
                    <>
                      <div className="h-4" />
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
                      {/* Row 4: Date */}
                      <div className="h-4 flex items-end">
                        <Text size="xs" fw={fontWeight} c={labelColor} className="whitespace-nowrap">
                          {point.date.getDate()}
                        </Text>
                      </div>
                    </>
                  )}
                  {isInterval && (
                    <>
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
