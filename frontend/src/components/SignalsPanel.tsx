import { useState, useMemo } from 'react';
import { Text, Badge, Group, Button, Stack, Anchor, Card } from '@mantine/core';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { IconExternalLink } from '@tabler/icons-react';
import type { SignalData } from '../types/vibe';

interface SignalsPanelProps {
  signals: SignalData[];
}

const CATEGORY_COLORS: Record<string, string> = {
  emergencies: '#ef4444',
  crime: '#f97316',
  festivals: '#a855f7',
  transportation: '#3b82f6',
  weather_temp: '#eab308',
  weather_wet: '#06b6d4',
  sports: '#10b981',
  economics: '#8b5cf6',
  politics: '#ec4899',
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
      // Show all articles from all signals
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
    <div className="grid lg:grid-cols-2 gap-6">
      {/* Bar Chart */}
      <Card shadow="sm" padding="lg" radius="md" withBorder>
        <Group justify="space-between" mb="md">
          <Text size="lg" fw={600}>
            Signal Intensities
          </Text>
          {selectedCategory && (
            <Button
              size="xs"
              variant="light"
              onClick={() => setSelectedCategory(null)}
            >
              Clear Filter
            </Button>
          )}
        </Group>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="category"
              angle={-45}
              textAnchor="end"
              height={100}
              tick={{ fontSize: 12 }}
            />
            <YAxis domain={[0, 1]} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
                      <Text size="sm" fw={600}>
                        {data.category}
                      </Text>
                      <Text size="sm">Score: {data.score.toFixed(2)}</Text>
                      <Text size="sm" c="dimmed">
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

        <Text size="xs" c="dimmed" mt="md">
          Click a bar to filter headlines by category
        </Text>
      </Card>

      {/* Headlines List */}
      <Card shadow="sm" padding="lg" radius="md" withBorder>
        <Group justify="space-between" mb="md">
          <div>
            <Text size="lg" fw={600}>
              Headlines
            </Text>
            {selectedCategory && (
              <Badge color={CATEGORY_COLORS[selectedCategory]} mt="xs">
                {selectedCategory}
              </Badge>
            )}
          </div>
          <Badge variant="light">{filteredArticles.length} articles</Badge>
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
                    color={CATEGORY_COLORS[article.category]}
                  >
                    {article.category}
                  </Badge>
                  <Text size="xs" c="dimmed">
                    {article.tag}
                  </Text>
                </Group>
                
                <Anchor
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-2"
                >
                  <Text size="sm" className="flex-1">
                    {article.title}
                  </Text>
                  <IconExternalLink size={14} className="mt-0.5 flex-shrink-0" />
                </Anchor>
                
                <Text size="xs" c="dimmed" mt="xs">
                  {article.source}
                  {article.published_at &&
                    ` • ${new Date(article.published_at).toLocaleDateString()}`}
                </Text>
              </div>
            ))
          ) : (
            <Text c="dimmed" ta="center">
              No headlines available
            </Text>
          )}
        </Stack>
      </Card>
    </div>
  );
};
