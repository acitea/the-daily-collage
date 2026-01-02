import { useState } from 'react';
import { Loader, Alert, Card, Text } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';
import { LocationHeader } from './components/LocationHeader';
import { TimelineSelector } from './components/TimelineSelector';
import { VibeCanvas } from './components/VibeCanvas';
import { SignalsPanel } from './components/SignalsPanel';
import { useCurrentVibe, useHistoricalVibe } from './hooks/useVibeData';

// Mock locations data - will be replaced by API call
const MOCK_LOCATIONS = [
  { id: 'stockholm', name: 'Stockholm', country: 'Sweden' },
  { id: 'gothenburg', name: 'Gothenburg', country: 'Sweden' },
  { id: 'malmo', name: 'Malmö', country: 'Sweden' },
];

function App() {
  const [selectedLocation, setSelectedLocation] = useState('stockholm');
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set());

  const handleSelectCategory = (category: string) => {
    setSelectedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  const {
    data: currentData,
    isLoading: isLoadingCurrent,
    error: currentError,
  } = useCurrentVibe(selectedLocation);

  const {
    data: historicalData,
    isLoading: isLoadingHistorical,
    error: historicalError,
  } = useHistoricalVibe(selectedLocation, selectedTimestamp);

  const isHistoricalMode = selectedTimestamp !== null;
  const activeData = isHistoricalMode ? historicalData : currentData;
  const isLoading = isHistoricalMode ? isLoadingHistorical : isLoadingCurrent;
  const error = isHistoricalMode ? historicalError : currentError;

  return (
    <div className="min-h-screen bg-white">
      {/* Header with logo and title */}
      <header className="bg-black text-white py-8 border-b border-gray-300">
        <div className="max-w-7xl mx-auto px-4">
          <Text size="xl" fw={700} className="text-center font-serif">
            THE DAILY COLLAGE
          </Text>
          <Text size="sm" className="text-center text-gray-400 mt-2 tracking-wider">
            REAL-TIME NEWS VIBES VISUALIZED
          </Text>
        </div>
      </header>

      {/* Location selector */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4">
          <div className="py-3">
            <Text size="xs" fw={600} className="text-gray-600 uppercase tracking-widest mb-3">
              Select Location
            </Text>
            <LocationHeader
              locations={MOCK_LOCATIONS}
              selectedLocation={selectedLocation}
              onLocationChange={setSelectedLocation}
            />
          </div>
        </div>
      </div>

      {/* Timeline selector */}
      <div className="bg-gray-50 border-b border-gray-300">
        <TimelineSelector
          selectedTimestamp={selectedTimestamp}
          onTimestampSelect={setSelectedTimestamp}
          currentTimestamp={currentData?.timestamp || new Date().toISOString()}
        />
      </div>

      {/* Main content */}
      <div className="bg-white py-8">
        <div className="max-w-4xl mx-auto px-4">
        {isLoading && (
          <div className="flex justify-center items-center py-20">
            <Loader size="xl" />
          </div>
        )}

        {error && (
          <Alert
            icon={<IconAlertCircle size={16} />}
            title="Error Loading Vibe"
            color="red"
            className="mb-6"
          >
            {error instanceof Error ? error.message : 'Failed to load vibe data'}
          </Alert>
        )}

        {activeData && !isLoading && (
          <div className="space-y-6">
            {/* Vibe canvas */}
            <div>
              <VibeCanvas
                imageUrl={activeData.image_url}
                hitboxes={activeData.hitboxes}
                alt={`${activeData.location} vibe`}
                selectedCategories={selectedCategories}
                onSelectCategory={handleSelectCategory}
              />
              <Text size="xs" c="dimmed" mt="md" className="text-center">
                Click on elements in the image to view related news articles
              </Text>
            </div>

            {/* Signals and headlines */}
            <SignalsPanel signals={activeData.signals} selectedCategories={selectedCategories} onSelectCategory={handleSelectCategory} />
          </div>
        )}

        {!activeData && !isLoading && !error && (
          <Card shadow="sm" padding="xl" radius="md" withBorder>
            <Text ta="center" c="dimmed">
              Select a location to view the vibe
            </Text>
          </Card>
        )}
      </div>

      </div>
      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 border-t border-gray-700 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4">
          <Text size="sm" ta="center" className="text-gray-500">
            The Daily Collage — A proof-of-concept news visualization system
          </Text>
          <Text size="xs" ta="center" mt="xs" className="text-gray-600">
            Updates every 6 hours • Powered by GDELT & Stability AI
          </Text>
        </div>
      </footer>
    </div>
  );
}

export default App;

