import { useMemo } from 'react';
import { Text, Group } from '@mantine/core';
import { ResponsiveBar } from '@nivo/bar';
import type { SignalData } from '../types/vibe';

interface SignalsChartProps {
  signals: SignalData[];
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  emergencies: '#fca5a5',
  crime: '#fbb6ce',
  festivals: '#a7f3d0',
  transportation: '#93c5fd',
  weather_temp: '#fed7aa',
  weather_wet: '#bae6fd',
  sports: '#c4b5fd',
  economics: '#fde68a',
  politics: '#fbcfe8',
};

export const SignalsChart = ({ signals, selectedCategory, onSelectCategory }: SignalsChartProps) => {
  const chartData = useMemo(() => {
    return signals.map((signal) => ({
      category: signal.category,
      score: signal.score,
      tag: signal.tag,
      color: CATEGORY_COLORS[signal.category] || '#64748b',
    }));
  }, [signals]);

  const legendItems = useMemo(() => {
    return signals.map((signal) => ({
      category: signal.category,
      color: CATEGORY_COLORS[signal.category] || '#64748b',
    }));
  }, [signals]);

  return (
    <div className="space-y-4">
      {/* Chart */}
      <div style={{ height: '300px', position: 'relative' }}>
        <div style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', zIndex: 10 }}>
          <Text size="xs" fw={500} className="text-gray-600" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
            Strong Negative
          </Text>
        </div>
        <div style={{ position: 'absolute', right: '30px', top: '50%', transform: 'translateY(-50%)', zIndex: 10 }}>
          <Text size="xs" fw={500} className="text-gray-600" style={{ writingMode: 'vertical-rl' }}>
            Strong Positive
          </Text>
        </div>
        <ResponsiveBar
          data={chartData}
          keys={['score']}
          indexBy="category"
          layout="horizontal"
          margin={{ top: 10, right: 30, bottom: 50, left: 10 }}
          padding={0.3}
          valueScale={{ type: 'linear', min: -1, max: 1 }}
          colors={(bar) => {
            const item = chartData.find((d) => d.category === bar.indexValue);
            return item?.color || '#64748b';
          }}
          borderColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
          axisTop={null}
          axisRight={null}
          axisBottom={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: '',
            legendPosition: 'middle',
            legendOffset: 40,
            tickValues: [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
          }}
          axisLeft={null}
          enableGridY={false}
          enableGridX={true}
          gridXValues={[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]}
          enableLabel={false}
          onClick={(node) => {
            const category = node.indexValue as string;
            if (selectedCategory === category) {
              onSelectCategory(null);
            } else {
              onSelectCategory(category);
            }
          }}
          tooltip={({ indexValue, value }) => (
            <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
              <Text size="sm" fw={600} className="text-gray-900">
                {indexValue}
              </Text>
              <Text size="sm" className="text-gray-700">
                Score: {Number(value).toFixed(2)}
              </Text>
            </div>
          )}
          theme={{
            axis: {
              ticks: {
                text: {
                  fontSize: 11,
                  fill: '#6b7280',
                },
              },
            },
            grid: {
              line: {
                stroke: '#e5e7eb',
                strokeWidth: 1,
              },
            },
          }}
        />
      </div>

      <Text size="xs" c="dimmed" className="text-center text-gray-600 mt-2">
        Click a bar to filter headlines by category
      </Text>

      {/* Color-coded legend */}
      <div>
        <div className="flex flex-wrap gap-4 justify-center">
          {legendItems.map((item) => (
            <Group key={item.category} gap="xs">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: item.color }}
              />
              <Text size="xs" className="text-gray-700 capitalize">
                {item.category.replace('_', ' ')}
              </Text>
            </Group>
          ))}
        </div>
      </div>
    </div>
  );
};
