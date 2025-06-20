'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrainingStatusResponse } from '../context/TrainingContext';

// The data prop expects our specific TrainingStatusResponse type, plus our added 'time' property.
interface SingleMetricChartProps {
  data: (TrainingStatusResponse & { time: string })[];
  title: string;
  dataKey: keyof TrainingStatusResponse;
  strokeColor: string;
}

export const CognitiveStateChart = ({ data, title, dataKey, strokeColor }: SingleMetricChartProps) => {
  
  // --- Smart Scale Logic ---
  // Calculate the dynamic domain for the Y-axis based on the visible data.
  const calculateDomain = () => {
    if (data.length === 0) {
      return [0, 1]; // Default domain if no data
    }
    
    let min: number | null = null;
    let max: number | null = null;

    for (const point of data) {
      const value = point[dataKey];
      if (typeof value === 'number' && isFinite(value)) { // Ensure value is a finite number
        if (min === null || value < min) min = value;
        if (max === null || value > max) max = value;
      }
    }

    if (min === null || max === null) {
      return [0, 1]; // Return default if no numeric data found for this key
    }

    // Add 10% padding to the top and bottom for better visual clearance
    const range = max - min;
    const padding = range === 0 ? 0.2 : range * 0.1; // Add slightly more padding for flat lines

    return [min - padding, max + padding];
  };

  const domain = calculateDomain();

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ textAlign: 'center', marginTop: 0, marginBottom: '1rem', color: '#4a4a4a', fontFamily: 'Arial, sans-serif' }}>
        {title}
      </h3>
      <div style={{ flexGrow: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" fontSize="0.75rem" tick={{ fill: '#666' }} />
            <YAxis 
              fontSize="0.75rem" 
              tick={{ fill: '#666' }} 
              domain={domain as [number, number]}
              tickFormatter={(value) => value.toFixed(3)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                border: '1px solid #ccc',
                borderRadius: '8px',
              }}
              formatter={(value: number) => typeof value === 'number' ? value.toFixed(4) : 'N/A'}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line 
              type="monotone" 
              dataKey={dataKey} 
              stroke={strokeColor} 
              strokeWidth={2} 
              name={title} 
              dot={false} 
              connectNulls={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};