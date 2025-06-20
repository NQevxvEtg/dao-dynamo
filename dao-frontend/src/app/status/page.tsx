// dao-frontend/src/app/status/page.tsx
'use client';

import { useState, useEffect, ReactNode } from 'react';
import { useTrainingStatus, TrainingStatusResponse } from '../context/TrainingContext';
import { CognitiveStateChart } from '../components/CognitiveStateChart';

// Define the colors for each chart for consistency
const COLORS = {
  focus: '#8884d8',
  confidence: '#82ca9d',
  metaError: '#ca8282',
  // stress: '#ffc658', // Removed as requested
  amplitude: '#ff8042',
  drift: '#00C49F', // Color for State Drift
  growth: '#0088FE',
  trainLoss: '#3498db',
  valLoss: '#9b59b6',
  clLoss: '#e74c3c',
  currentAmplitude: '#8a2be2', // New color for Current Amplitude
};

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;

export default function StatusPage() {
  const { trainingStatus } = useTrainingStatus();
  
  // State to hold the complete history fetched from the backend
  const [fullHistory, setFullHistory] = useState<(TrainingStatusResponse & { time: string })[]>([]);
  // State for the data that is actually rendered in the charts
  const [displayHistory, setDisplayHistory] = useState<(TrainingStatusResponse & { time: string })[]>([]);
  // State for the user-configurable window size
  const [windowSize, setWindowSize] = useState(150);
  
  const [isLoading, setIsLoading] = useState(false);

  // Initial Data Fetch
  useEffect(() => {
    const fetchHistory = async () => {
      setIsLoading(true);
      try {
        if (!BACKEND_API_URL) return;
        const res = await fetch(`${BACKEND_API_URL}/cognitive_state_history`);
        if (res.ok) {
          const data: TrainingStatusResponse[] = await res.json();
          const formattedData = data.map(d => ({
            ...d,
            time: d.timestamp 
              ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
              : 'N/A',
          }));
          // Store the entire fetched history
          setFullHistory(formattedData);
        }
      } catch (error) {
        console.error("Failed to fetch cognitive state history:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // Live Data Update - Appends new data to the full history
  useEffect(() => {
    if (trainingStatus) {
      const now = new Date();
      const newStatus = {
        ...trainingStatus,
        time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      };
      setFullHistory(prevFullHistory => [...prevFullHistory, newStatus]);
    }
  }, [trainingStatus]);

  // This effect runs whenever the full history or the window size changes,
  // creating the sliced data for display.
  useEffect(() => {
    // Ensure windowSize is a positive number
    const effectiveWindowSize = Math.max(1, windowSize);
    if (fullHistory.length > effectiveWindowSize) {
      setDisplayHistory(fullHistory.slice(fullHistory.length - effectiveWindowSize));
    } else {
      setDisplayHistory(fullHistory);
    }
  }, [fullHistory, windowSize]);


  // Database Control Handlers
  const handleExport = async () => {
    try {
      if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
      window.open(`${BACKEND_API_URL}/export_cognitive_state`, '_blank');
    } catch (error) {
      alert(`Failed to export history: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear the entire cognitive state history? This action cannot be undone.')) {
      return;
    }
    setIsLoading(true);
    try {
      if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
      const res = await fetch(`${BACKEND_API_URL}/clear_cognitive_state`, { method: 'DELETE' });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
        setFullHistory([]); // Clear the history state
      } else {
        throw new Error(data.detail || "Failed to clear history.");
      }
    } catch (error) {
      alert(`Error clearing history: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleWindowChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    // Set to 1 if input is empty or less than 1, otherwise use the value
    setWindowSize(isNaN(value) || value < 1 ? 1 : value);
  };

  const renderChartContainer = (child: ReactNode) => (
    <div style={{
      width: '100%',
      height: '280px',
      backgroundColor: 'var(--card-background)',
      borderRadius: '8px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
      padding: '1rem',
      border: '1px solid var(--card-border)'
    }}>
      {child}
    </div>
  );

  return (
    <main style={{
      padding: '1rem',
    }}>
      <h1 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>dao&apos;s Real-Time Status</h1>
      
      {/* --- Control Bar --- */}
      <div style={{
        width: '100%', 
        maxWidth: '1200px', 
        margin: '0 auto 1.5rem', 
        display: 'flex', 
        justifyContent: 'flex-end', 
        alignItems: 'center',
        gap: '1rem', 
        flexWrap: 'wrap'
      }}>
        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
          <label htmlFor="window-size-input" style={{color: 'var(--foreground-subtle)', fontSize: '0.9rem'}}>Data Points:</label>
          <input
            id="window-size-input"
            type="number"
            value={windowSize}
            onChange={handleWindowChange}
            min="1"
            style={{
              width: '80px',
              padding: '0.5rem',
              backgroundColor: 'var(--card-background)',
              color: 'var(--foreground)',
              border: '1px solid var(--card-border)',
              borderRadius: '4px'
            }}
          />
        </div>
        <button onClick={handleExport} disabled={isLoading} style={{padding: '0.5rem 1rem', cursor: 'pointer', border: `1px solid var(--button-secondary-border)`, borderRadius: '4px', backgroundColor: 'var(--button-secondary-bg)', color: 'var(--button-secondary-text)'}}>Export History</button>
        <button onClick={handleClear} disabled={isLoading} style={{padding: '0.5rem 1rem', cursor: 'pointer', backgroundColor: 'var(--button-danger-bg)', color: 'var(--button-danger-text)', border: 'none', borderRadius: '4px'}}>Clear History</button>
      </div>

      {/* --- Charts Grid --- */}
      <div className="grid w-full max-w-6xl grid-cols-1 gap-6 md:grid-cols-2 mx-auto">
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Focus" dataKey="focus" strokeColor={COLORS.focus}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Confidence" dataKey="confidence" strokeColor={COLORS.confidence}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Meta-Error" dataKey="meta_error" strokeColor={COLORS.metaError}/>)}
        {/* Replaced Cognitive Stress with State Drift and Current Amplitude */}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Novelty / Surprise (State Drift)" dataKey="state_drift" strokeColor={COLORS.drift}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Resonator Amplitude (Current)" dataKey="current_amplitude" strokeColor={COLORS.currentAmplitude}/>)}
        {/* Original Resonator Amplitude chart was "target_amplitude", now we explicitly added "current_amplitude" as well */}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Resonator Amplitude (Target)" dataKey="target_amplitude" strokeColor={COLORS.amplitude}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Long-Term Growth (Base Focus)" dataKey="base_focus" strokeColor={COLORS.growth}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Training Loss" dataKey="train_loss" strokeColor={COLORS.trainLoss}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Validation Loss" dataKey="val_loss" strokeColor={COLORS.valLoss}/>)}
        {renderChartContainer(<CognitiveStateChart data={displayHistory} title="Interactive Loss" dataKey="continuous_learning_loss" strokeColor={COLORS.clLoss}/>)}
      </div>

      {trainingStatus && trainingStatus.is_training_active && (
        <div style={{
          marginTop: '2rem',
          backgroundColor: 'var(--card-background)',
          borderRadius: '8px',
          padding: '1rem 1.5rem',
          width: '100%',
          maxWidth: '1200px',
          margin: '2rem auto 0',
          fontSize: '0.9rem',
          color: 'var(--foreground)',
          border: '1px solid var(--card-border)'
        }}>
          <h3 style={{marginTop: 0}}>Training In Progress</h3>
          <strong>Status Message:</strong> {trainingStatus.message}
        </div>
      )}
    </main>
  );
}