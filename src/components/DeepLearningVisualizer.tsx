"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle, Brain, Zap } from 'lucide-react';

interface DeepLearningVisualizerProps {
    algorithmId: string;
}

interface Layer {
    type: 'input' | 'hidden' | 'conv' | 'pool' | 'lstm' | 'attention' | 'output';
    neurons: number;
    activation?: string;
    state: 'default' | 'active' | 'complete';
}

interface TrainingMetrics {
    epoch: number;
    loss: number;
    accuracy: number;
    valLoss?: number;
    valAccuracy?: number;
}

export const DeepLearningVisualizer = ({ algorithmId }: DeepLearningVisualizerProps) => {
    const [layers, setLayers] = useState<Layer[]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [step, setStep] = useState<'idle' | 'forward' | 'backward' | 'training' | 'complete'>('idle');
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [epochs, setEpochs] = useState(10);
    const [learningRate, setLearningRate] = useState(0.01);
    const [trainingHistory, setTrainingHistory] = useState<TrainingMetrics[]>([]);
    const [activeLayer, setActiveLayer] = useState(-1);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Generate network architecture based on algorithm
    const generateArchitecture = useCallback(() => {
        let newLayers: Layer[] = [];

        switch (algorithmId) {
            case 'neural-network':
                newLayers = [
                    { type: 'input', neurons: 4, state: 'default' },
                    { type: 'hidden', neurons: 8, activation: 'ReLU', state: 'default' },
                    { type: 'hidden', neurons: 6, activation: 'ReLU', state: 'default' },
                    { type: 'output', neurons: 3, activation: 'Softmax', state: 'default' }
                ];
                break;
            case 'cnn':
                newLayers = [
                    { type: 'input', neurons: 28, state: 'default' },
                    { type: 'conv', neurons: 32, activation: 'ReLU', state: 'default' },
                    { type: 'pool', neurons: 16, state: 'default' },
                    { type: 'conv', neurons: 64, activation: 'ReLU', state: 'default' },
                    { type: 'pool', neurons: 8, state: 'default' },
                    { type: 'hidden', neurons: 128, activation: 'ReLU', state: 'default' },
                    { type: 'output', neurons: 10, activation: 'Softmax', state: 'default' }
                ];
                break;
            case 'rnn-lstm':
                newLayers = [
                    { type: 'input', neurons: 10, state: 'default' },
                    { type: 'lstm', neurons: 64, state: 'default' },
                    { type: 'lstm', neurons: 32, state: 'default' },
                    { type: 'hidden', neurons: 16, activation: 'ReLU', state: 'default' },
                    { type: 'output', neurons: 1, activation: 'Sigmoid', state: 'default' }
                ];
                break;
            case 'transformer':
                newLayers = [
                    { type: 'input', neurons: 512, state: 'default' },
                    { type: 'attention', neurons: 8, state: 'default' },
                    { type: 'hidden', neurons: 2048, activation: 'GELU', state: 'default' },
                    { type: 'attention', neurons: 8, state: 'default' },
                    { type: 'hidden', neurons: 2048, activation: 'GELU', state: 'default' },
                    { type: 'output', neurons: 1000, activation: 'Softmax', state: 'default' }
                ];
                break;
            default:
                newLayers = [
                    { type: 'input', neurons: 4, state: 'default' },
                    { type: 'hidden', neurons: 8, activation: 'ReLU', state: 'default' },
                    { type: 'output', neurons: 2, activation: 'Softmax', state: 'default' }
                ];
        }

        setLayers(newLayers);
        setTrainingHistory([]);
        setCurrentEpoch(0);
        setStep('idle');
        setActiveLayer(-1);
    }, [algorithmId]);

    useEffect(() => {
        generateArchitecture();
    }, [generateArchitecture]);

    // Simulate training
    const runTraining = async () => {
        setIsTraining(true);
        setTrainingHistory([]);
        setStep('training');

        let loss = 2.5 + Math.random() * 0.5;
        let accuracy = 0.1 + Math.random() * 0.1;

        for (let epoch = 1; epoch <= epochs; epoch++) {
            setCurrentEpoch(epoch);

            // Forward pass animation
            setStep('forward');
            for (let i = 0; i < layers.length; i++) {
                setActiveLayer(i);
                const newLayers = [...layers];
                newLayers[i].state = 'active';
                setLayers([...newLayers]);
                await sleep(150);
                newLayers[i].state = 'complete';
                setLayers([...newLayers]);
            }

            // Backward pass animation
            setStep('backward');
            for (let i = layers.length - 1; i >= 0; i--) {
                setActiveLayer(i);
                const newLayers = [...layers];
                newLayers[i].state = 'active';
                setLayers([...newLayers]);
                await sleep(100);
                newLayers[i].state = 'default';
                setLayers([...newLayers]);
            }

            // Update metrics
            const decay = Math.exp(-epoch * learningRate * 5);
            loss = loss * (0.7 + decay * 0.3) + (Math.random() - 0.5) * 0.1;
            accuracy = Math.min(0.99, accuracy + (1 - accuracy) * (0.15 + Math.random() * 0.1));

            const metrics: TrainingMetrics = {
                epoch,
                loss: Math.max(0.01, loss),
                accuracy,
                valLoss: Math.max(0.01, loss * (1.1 + Math.random() * 0.2)),
                valAccuracy: Math.max(0, accuracy - Math.random() * 0.05)
            };

            setTrainingHistory(prev => [...prev, metrics]);
            setStep('training');
            await sleep(200);
        }

        setStep('complete');
        setActiveLayer(-1);
        setIsTraining(false);
    };

    const getLayerColor = (layer: Layer) => {
        if (layer.state === 'active') return '#FBBF24';
        if (layer.state === 'complete') return '#10B981';

        switch (layer.type) {
            case 'input': return '#3B82F6';
            case 'conv': return '#8B5CF6';
            case 'pool': return '#EC4899';
            case 'lstm': return '#F97316';
            case 'attention': return '#EF4444';
            case 'output': return '#10B981';
            default: return '#6366F1';
        }
    };

    const getLayerLabel = (layer: Layer) => {
        switch (layer.type) {
            case 'input': return 'Input';
            case 'conv': return 'Conv2D';
            case 'pool': return 'MaxPool';
            case 'lstm': return 'LSTM';
            case 'attention': return 'Attention';
            case 'output': return 'Output';
            default: return 'Dense';
        }
    };

    const algorithmName = {
        'neural-network': 'Neural Network',
        'cnn': 'CNN',
        'rnn-lstm': 'RNN/LSTM',
        'transformer': 'Transformer'
    }[algorithmId] || 'Deep Learning';

    const latestMetrics = trainingHistory[trainingHistory.length - 1];
    const maxLoss = Math.max(...trainingHistory.map(m => m.loss), 3);

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    <Brain className="w-5 h-5 inline mr-2 text-purple-600" />
                    {algorithmName} Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generateArchitecture}
                        disabled={isTraining}
                    >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        Reset
                    </Button>
                    <Button
                        size="sm"
                        onClick={runTraining}
                        disabled={isTraining}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isTraining ? 'Training...' : 'Train'}
                    </Button>
                </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Epochs</span>
                        <span className="font-mono text-purple-600">{epochs}</span>
                    </div>
                    <Slider
                        value={[epochs]}
                        onValueChange={(v) => setEpochs(v[0])}
                        min={5}
                        max={30}
                        step={5}
                        disabled={isTraining}
                    />
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Learning Rate</span>
                        <span className="font-mono text-blue-600">{learningRate}</span>
                    </div>
                    <Slider
                        value={[learningRate * 100]}
                        onValueChange={(v) => setLearningRate(v[0] / 100)}
                        min={1}
                        max={10}
                        step={1}
                        disabled={isTraining}
                    />
                </div>
            </div>

            {/* Status */}
            <div className="text-sm text-center">
                {step === 'idle' && <span className="text-gray-500">Ready to train</span>}
                {step === 'forward' && <span className="text-blue-600">⏩ Forward Pass (Layer {activeLayer + 1}/{layers.length})</span>}
                {step === 'backward' && <span className="text-orange-600">⏪ Backward Pass (Gradient)</span>}
                {step === 'training' && <span className="text-purple-600">🔄 Epoch {currentEpoch}/{epochs}</span>}
                {step === 'complete' && <span className="text-emerald-600">✅ Training Complete!</span>}
            </div>

            {/* Network Architecture */}
            <div className="relative h-40 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden p-4">
                <div className="flex items-center justify-between h-full">
                    {layers.map((layer, idx) => {
                        const layerHeight = Math.min(100, 20 + (layer.neurons / 20) * 80);
                        const isActive = activeLayer === idx;

                        return (
                            <div key={idx} className="flex flex-col items-center gap-1">
                                {/* Layer block */}
                                <div
                                    className={`rounded-lg transition-all duration-200 flex items-center justify-center ${isActive ? 'ring-2 ring-yellow-400 ring-offset-2' : ''}`}
                                    style={{
                                        width: layer.type === 'attention' ? 50 : 40,
                                        height: layerHeight,
                                        backgroundColor: getLayerColor(layer),
                                        opacity: layer.state === 'default' ? 0.7 : 1
                                    }}
                                >
                                    <span className="text-white text-[10px] font-bold writing-vertical">
                                        {layer.neurons}
                                    </span>
                                </div>
                                {/* Layer label */}
                                <span className="text-[9px] text-gray-500 text-center">
                                    {getLayerLabel(layer)}
                                </span>
                                {layer.activation && (
                                    <span className="text-[8px] text-gray-400">
                                        {layer.activation}
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Connection lines */}
                <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: 0 }}>
                    {layers.slice(0, -1).map((_, idx) => {
                        const x1 = (idx + 0.5) / layers.length * 100 + 5;
                        const x2 = (idx + 1.5) / layers.length * 100 + 5;
                        return (
                            <line
                                key={`conn-${idx}`}
                                x1={`${x1}%`}
                                y1="50%"
                                x2={`${x2}%`}
                                y2="50%"
                                stroke={activeLayer === idx || activeLayer === idx + 1 ? '#FBBF24' : '#D1D5DB'}
                                strokeWidth="2"
                                strokeDasharray={activeLayer === idx ? "5,3" : "0"}
                            />
                        );
                    })}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-xs flex-wrap">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-blue-500" />
                    <span className="text-gray-600">Input</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-indigo-500" />
                    <span className="text-gray-600">Dense</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-violet-500" />
                    <span className="text-gray-600">Conv</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-orange-500" />
                    <span className="text-gray-600">LSTM</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-rose-500" />
                    <span className="text-gray-600">Attention</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-emerald-500" />
                    <span className="text-gray-600">Output</span>
                </div>
            </div>

            {/* Training Progress */}
            {trainingHistory.length > 0 && (
                <div className="space-y-4">
                    {/* Loss Chart */}
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <div className="text-xs font-bold text-gray-500 uppercase mb-2">Training Loss</div>
                        <div className="h-24 flex items-end gap-1">
                            {trainingHistory.map((m, i) => (
                                <div
                                    key={i}
                                    className="flex-1 bg-gradient-to-t from-rose-500 to-rose-300 rounded-t transition-all duration-200"
                                    style={{ height: `${(m.loss / maxLoss) * 100}%` }}
                                    title={`Epoch ${m.epoch}: ${m.loss.toFixed(4)}`}
                                />
                            ))}
                        </div>
                        <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                            <span>Epoch 1</span>
                            <span>Epoch {epochs}</span>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg border border-emerald-200 text-center">
                            <div className="text-[10px] text-emerald-600 uppercase font-bold mb-1">Accuracy</div>
                            <div className="text-xl font-bold text-emerald-700">
                                {((latestMetrics?.accuracy || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-3 bg-gradient-to-br from-rose-50 to-red-50 rounded-lg border border-rose-200 text-center">
                            <div className="text-[10px] text-rose-600 uppercase font-bold mb-1">Loss</div>
                            <div className="text-xl font-bold text-rose-700">
                                {latestMetrics?.loss.toFixed(4)}
                            </div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Val Accuracy</div>
                            <div className="text-xl font-bold text-gray-700">
                                {((latestMetrics?.valAccuracy || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Val Loss</div>
                            <div className="text-xl font-bold text-gray-700">
                                {latestMetrics?.valLoss?.toFixed(4)}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Epoch Progress Bar */}
            {isTraining && (
                <div className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-500">
                        <span>Progress</span>
                        <span>{currentEpoch}/{epochs} epochs</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-purple-500 to-violet-500 transition-all duration-300"
                            style={{ width: `${(currentEpoch / epochs) * 100}%` }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};
