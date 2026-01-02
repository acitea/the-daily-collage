import { useMemo } from 'react';
import { Text, Group } from '@mantine/core';
import { BarChart } from '@mantine/charts';
import type { SignalData } from '../types/vibe';

interface SignalsChartProps {
  signals: SignalData[];
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  emergencies: '#dc2626',
  crime: '#7c2d12',
  festivals: '#16a34a',
  transportation: '#0284c7',
  weather_temp: '#ea580c',
  weather_wet: '#0369a1',
  sports: '#7c3aed',
  economics: '#ea580c',
  politics: '#06b6d4',
};

export const SignalsChart = ({ signals }: SignalsChartProps) => {
  const chartData = useMemo(() => {
    return signals.map((signal) => ({
      category: signal.category,
      score: signal.score,
      tag: signal.tag,
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
      <div>
        <BarChart
          h={300}
          data={chartData}
          dataKey="category"
          series={signals.map((signal) => ({
            name: signal.category,
            color: CATEGORY_COLORS[signal.category] || '#64748b',
          }))}
          orientation="vertical"
          yAxisProps={{ width: 100 }}
          withTooltip
        />

        <Text size="xs" c="dimmed" mt="md" className="text-center text-gray-600">
          Click a bar to filter headlines by category
        </Text>
      </div>

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
