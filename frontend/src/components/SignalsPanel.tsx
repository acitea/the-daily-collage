import { useState, useMemo } from 'react';
import { Text, Anchor, Table } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import { SignalsChart } from './SignalsChart';
import type { SignalData } from '../types/vibe';

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
        {filteredArticles.length > 0 ? (
          <Table striped highlightOnHover withColumnBorders>
            <Table.Thead className="border-b-2 border-gray-300">
              <Table.Tr>
                <Table.Th className="text-lg font-serif font-semibold text-gray-900 py-4">Headline</Table.Th>
                <Table.Th className="text-lg font-serif font-semibold text-gray-900 text-right py-4">Category</Table.Th>
                <Table.Th className="text-lg font-serif font-semibold text-gray-900 text-right py-4" style={{ width: '100px' }}>Score</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filteredArticles.map((article, idx) => (
                <Table.Tr key={idx}>
                  <Table.Td>
                    <Anchor
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-gray-900 hover:text-blue-600"
                    >
                      <span>{article.title}</span>
                      <span className='inline-block'><IconExternalLink size={14} className="flex-shrink-0" /></span>
                    </Anchor>
                  </Table.Td>
                  <Table.Td className="text-right">
                    <span 
                      className="inline-block px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap"
                      style={{ 
                        backgroundColor: CATEGORY_COLORS[article.category] || '#e5e7eb',
                        color: '#1f2937'
                      }}
                    >
                      {article.category}
                    </span>
                  </Table.Td>
                  <Table.Td className="text-right">
                    <Text size="sm" fw={600} className="text-gray-900">
                      {article.score > 0 ? '+' : ''}{article.score.toFixed(2)}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        ) : (
          <Text c="dimmed" ta="center" className="text-gray-500 py-8">
            No headlines available
          </Text>
        )}
      </div>
    </div>
  );
};
