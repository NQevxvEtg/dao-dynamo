'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// This is the shape of the training status data received from the backend
export interface TrainingStatusResponse {
  timestamp?: string;
  is_training_active: boolean;
  message: string;
  
  // Core Metrics
  focus: number;
  confidence: number;
  meta_error: number;
  curiosity: number;
  
  // Training-specific Metrics
  current_epoch?: number;
  current_batch?: number;
  total_batches_in_epoch?: number;
  train_loss?: number | null;
  val_loss?: number | null;
  best_val_loss?: number | null;
  
  // Advanced Metrics for Charting
  cognitive_stress?: number | null;
  target_amplitude?: number | null;
  current_amplitude?: number | null;
  target_frequency?: number | null;
  current_frequency?: number | null;
  base_focus?: number | null;
  base_curiosity?: number | null;
  state_drift?: number | null;
  continuous_learning_loss?: number | null;
}

// Define the shape of the context state
interface TrainingContextType {
  trainingStatus: TrainingStatusResponse | null;
  isConnected: boolean;
}

// Create the context
const TrainingContext = createContext<TrainingContextType>({
  trainingStatus: null,
  isConnected: false,
});

// Create a custom hook for easy consumption of the context
export const useTrainingStatus = () => useContext(TrainingContext);

// Create the Provider component
export const TrainingProvider = ({ children }: { children: ReactNode }) => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusResponse | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;

  useEffect(() => {
    if (!BACKEND_API_URL) {
      console.error("BACKEND_API_URL is not defined. Cannot connect to WebSocket.");
      return;
    }

    const wsUrl = BACKEND_API_URL.replace('http', 'ws') + '/ws/training_updates';
    let ws: WebSocket;
    let reconnectTimeout: NodeJS.Timeout;

    function connect() {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Global WebSocket connection opened for training status.');
        setIsConnected(true);
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
      };

      ws.onmessage = (event) => {
        try {
          const data: TrainingStatusResponse = JSON.parse(event.data);
          setTrainingStatus(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log(`Global WebSocket connection closed. Code: ${event.code}`);
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 5000);
      };

      ws.onerror = (error) => {
        console.error('Global WebSocket error:', error);
        ws.close();
      };
    }

    connect();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) {
        ws.onclose = null; 
        ws.close();
      }
    };
  }, [BACKEND_API_URL]);

  const value = { trainingStatus, isConnected };

  return (
    <TrainingContext.Provider value={value}>
      {children}
    </TrainingContext.Provider>
  );
};