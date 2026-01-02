import { useState, useMemo } from 'react';
import { Text, Badge, Group, Button, Stack, Anchor, Card } from '@mantine/core';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { IconExternalLink } from '@tabler/icons-react';
import type { SignalData } from '../types/vibe';

interface SignalsPanelProps {
  signals: SignalData[];
}

const CATEGORY_COLORS: Record<string, string> = {
  emergencies: '#7c2d12',
  crime: '#7c2d12',
  festivals: '#6b7280',
  transportation: '#374151',
  weather_temp: '#78716c',
  weather_wet: '#57534e',
  sports: '#4b5563',
  economics: '#1f2937',
  politics: '#374151',
};

export const SignalsPanel = ({ signals }: SignalsPanelProps) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const chartData = useMemo(() => {
    return signals.map((signal) => ({
      category: signal.category,
      score: Math.abs(signal.score),
      tag: signal.tag,
      color: CATEGORY_COLORS[signal.category] || '#64748b',
    }));
  }, [signals]);

  const filteredArticles = useMemo(() => {
    if (!selectedCategory) {
      return signals.flatMap((signal) =>
        signal.articles.map((article) => ({
          ...article,
          category: signal.category,
          tag: signal.tag,
        }))
      );
    }

    const signal = signals.find((s) => s.category === selectedCategory);
    return signal
      ? signal.articles.map((article) => ({
          ...article,
          category: signal.category,
          tag: signal.tag,
        }))
      : [];
  }, [signals, selectedCategory]);

  return (
    <div className="grid lg:grid-cols-2 gap-8">
      {/* Bar Chart */}
      <Card shadow="sm" padding="lg" radius="md" className="border border-gray-300">
        <Group justify="space-between" mb="md">
          <Text size="lg" fw={600} className="font-serif text-gray-900">
            Signal Intensities
          </Text>
          {selectedCategory && (
            <Button
              size="xs"
              variant="default"
              className="bg-gray-800 text-white hover:bg-gray-900"
              onClick={() => setSelectedCategory(null)}
            >
              Clear Filter
            </Button>
          )}
        </Group>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="category"
              angle={-45}
              textAnchor="end"
              height={100}
              tick={{ fontSize: 12, fill: '#4b5563' }}
            />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12, fill: '#4b5563' }} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-2 border border-gray-300 rounded shadow-lg">
                      <Text size="sm" fw={600} className="text-gray-900">
                        {data.category}
                      </Text>
                      <Text size="sm" className="text-gray-700">
                        Score: {data.score.toFixed(2)}
                      </Text>
                      <Text size="sm" c="dimmed" className="text-gray-600">
                        {data.tag}
                      </Text>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar
              dataKey="score"
              onClick={(data: any) => setSelectedCategory(data.category)}
              className="cursor-pointer"
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.color}
                  opacity={
                    selectedCategory === null || selectedCategory === entry.category
                      ? 1
                      : 0.3
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <Text size="xs" c="dimmed" mt="md" className="text-gray-600">
          Click a bar to filter headlines by category
        </Text>
      </Card>

      {/* Headlines List */}
      <Card shadow="sm" padding="lg" radius="md" className="border border-gray-300">
        <Group justify="space-between" mb="md">
          <div>
            <Text size="lg" fw={600} className="font-serif text-gray-900">
              Headlines
            </Text>
            {selectedCategory && (
              <Badge color="dark" mt="xs" className="bg-gray-800">
                {selectedCategory}
              </Badge>
            )}
          </div>
          <Badge variant="light" className="bg-gray-100 text-gray-800">
            {filteredArticles.length} articles
          </Badge>
        </Group>

        <Stack gap="md" className="max-h-[400px] overflow-y-auto">
          {filteredArticles.length > 0 ? (
            filteredArticles.map((article, idx) => (
              <div
                key={idx}
                className="p-3 border border-gray-200 rounded hover:bg-gray-50 transition-colors"
              >
                <Group justify="space-between" align="start" mb="xs">
                  <Badge
                    size="xs"
                    color="gray"
                    className="bg-gray-200 text-gray-800"
                  >
                    {article.category}
                  </Badge>
                  <Text size="xs" c="dimmed" className="text-gray-600">
                    {article.tag}
                  </Text>
                </Group>

                <Anchor
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-2"
                >
                  <Text size="sm" className="flex-1 text-gray-900 font-medium">
                    {article.title}
                  </Text>
                  <IconExternalLink size={14} className="mt-0.5 flex-shrink-0 text-gray-500" />
                </Anchor>

                <Text size="xs" c="dimmed" mt="xs" className="text-gray-600">
                  {article.source}
                  {article.published_at &&
                    ` • ${new Date(article.published_at).toLocaleDateString()}`}
                </Text>
              </div>
            ))
          ) : (
            <Text c="dimmed" ta="center" className="text-gray-500">
              No headlines available
            </Text>
          )}
        </Stack>
      </Card>
    </div>
  );
};
