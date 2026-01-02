import { useState, useMemo } from 'react';
import { Text, Anchor, Stack } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import { SignalsChart } from './SignalsChart';
import type { SignalData } from '../types/vibe';

interface SignalsPanelProps {
  signals: SignalData[];
}

export const SignalsPanel = ({ signals }: SignalsPanelProps) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Get filtered articles based on selected category
  const filteredArticles = useMemo(() => {
    if (!selectedCategory) {
      return signals.flatMap((signal) =>
        signal.articles.map((article) => ({
          ...article,
          category: signal.category,
          tag: signal.tag,
          score: signal.score,
        }))
      );
    }

    const signal = signals.find((s) => s.category === selectedCategory);
    return signal
      ? signal.articles.map((article) => ({
          ...article,
          category: signal.category,
          tag: signal.tag,
          score: signal.score,
        }))
      : [];
  }, [signals, selectedCategory]);

  return (
    <div className="space-y-6">
      {/* Signals Chart */}
      <SignalsChart 
        signals={signals} 
        selectedCategory={selectedCategory}
        onSelectCategory={setSelectedCategory}
      />

      {/* Headlines Table */}
      <div>
        <Text size="lg" fw={600} className="font-serif text-gray-900 mb-4">
          Headlines
        </Text>

        {filteredArticles.length > 0 ? (
          <Stack gap="sm">
            {filteredArticles.map((article, idx) => (
              <div
                key={idx}
                className="p-4 border border-gray-200 rounded hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start gap-4 justify-between">
                  {/* Headline and link on the left */}
                  <Anchor
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 flex-1 min-w-0"
                  >
                    <Text size="sm" fw={500} className="flex-1 text-gray-900 break-words">
                      {article.title}
                    </Text>
                    <IconExternalLink
                      size={16}
                      className="mt-0.5 flex-shrink-0 text-gray-500"
                    />
                  </Anchor>

                  {/* Category chip and score on the right */}
                  <div className="flex items-center gap-3 ml-4 flex-shrink-0">
                    <div className="px-3 py-1 rounded-full text-xs font-medium bg-gray-200 text-gray-800 whitespace-nowrap">
                      {article.category}
                    </div>
                    <Text size="sm" fw={600} className="text-gray-900 whitespace-nowrap">
                      {article.score > 0 ? '+' : ''}{article.score.toFixed(2)}
                    </Text>
                  </div>
                </div>

                <Text size="xs" c="dimmed" mt="xs" className="text-gray-600">
                  {article.source}
                  {article.published_at &&
                    ` • ${new Date(article.published_at).toLocaleDateString()}`}
                </Text>
              </div>
            ))}
          </Stack>
        ) : (
          <Text c="dimmed" ta="center" className="text-gray-500 py-8">
            No headlines available
          </Text>
        )}
      </div>
    </div>
  );
};
