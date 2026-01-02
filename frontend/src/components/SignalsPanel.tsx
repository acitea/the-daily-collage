import { useState, useMemo } from 'react';
import { Collapse } from '@mantine/core';
import { IconChevronDown, IconChevronUp } from '@tabler/icons-react';
import { SignalsChart } from './SignalsChart';
import { HeadlinesTable } from './HeadlinesTable';
import type { SignalData } from '../types/vibe';

interface SignalsPanelProps {
  signals: SignalData[];
}

export const SignalsPanel = ({ signals }: SignalsPanelProps) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [isChartOpen, setIsChartOpen] = useState(true);

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
      {/* Collapsible Signals Chart Section */}
      <div>
        <div
          onClick={() => setIsChartOpen(!isChartOpen)}
          className="flex items-center gap-2 cursor-pointer group relative mb-3"
        >
          {isChartOpen ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
          <span className="text-lg font-serif font-semibold text-gray-900">
            Signal Intensity Chart
          </span>
          <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gray-900 transition-all duration-300 ease-out group-hover:w-full"></span>
        </div>
        <Collapse in={isChartOpen}>
          <SignalsChart 
            signals={signals} 
            selectedCategory={selectedCategory}
            onSelectCategory={setSelectedCategory}
          />
        </Collapse>
      </div>

      {/* Headlines Table */}
      <HeadlinesTable articles={filteredArticles} />
    </div>
  );
};
