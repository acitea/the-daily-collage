import { Modal, Group, Button, Stack, Select } from '@mantine/core';
import { DateInput } from '@mantine/dates';
import { useState, useEffect } from 'react';
import { roundToNearestInterval } from './helpers';

interface DateTimePickerModalProps {
  opened: boolean;
  onClose: () => void;
  initialValue: string | null;
  onSubmit: (timestamp: string) => void;
}

const TIME_OPTIONS = [
  { value: '0', label: '00:00 (Midnight)' },
  { value: '6', label: '06:00' },
  { value: '12', label: '12:00 (Noon)' },
  { value: '18', label: '18:00' },
];

export const DateTimePickerModal = ({
  opened,
  onClose,
  initialValue,
  onSubmit,
}: DateTimePickerModalProps) => {
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);
  const [selectedHour, setSelectedHour] = useState<string>('0');

  // Calculate min and max dates (last 30 days)
  const maxDate = new Date();
  const minDate = new Date();
  minDate.setDate(minDate.getDate() - 30);

  useEffect(() => {
    if (opened && initialValue) {
      const date = new Date(initialValue);
      setSelectedDate(date);
      setSelectedHour(date.getHours().toString());
    } else if (opened) {
      const rounded = roundToNearestInterval(new Date());
      setSelectedDate(rounded);
      setSelectedHour(rounded.getHours().toString());
    }
  }, [opened, initialValue]);

  const handleSubmit = () => {
    if (selectedDate) {
      const finalDate = new Date(selectedDate);
      finalDate.setHours(parseInt(selectedHour), 0, 0, 0);
      onSubmit(finalDate.toISOString());
      onClose();
    }
  };

  return (
    <Modal opened={opened} onClose={onClose} title="Select Date and Time" size="auto">
      <Stack gap="md">
        <DateInput
          value={selectedDate}
          onChange={(value) => setSelectedDate(value ? new Date(value) : null)}
          label="Date"
          placeholder="Pick a date"
          clearable
          valueFormat="YYYY-MM-DD"
          minDate={minDate}
          maxDate={maxDate}
        />
        <Select
          label="Time"
          placeholder="Select time"
          data={TIME_OPTIONS}
          value={selectedHour}
          onChange={(value) => value && setSelectedHour(value)}
        />
        <Group justify="flex-end" mt="md">
          <Button variant="light" color="dark" onClick={onClose}>
            Cancel
          </Button>
          <Button color="dark" onClick={handleSubmit} disabled={!selectedDate}>
            Go
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
};
