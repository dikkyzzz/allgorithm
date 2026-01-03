"use client";

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Calculator, RotateCcw, Play, BarChart3 } from 'lucide-react';

interface StatsVisualizerProps {
    algorithmId: string;
}

export const StatsVisualizer = ({ algorithmId }: StatsVisualizerProps) => {
    const [inputData, setInputData] = useState('12, 7, 9, 14, 7, 8, 11, 7, 10, 15');
    const [results, setResults] = useState<Record<string, number | string> | null>(null);
    const [isRunning, setIsRunning] = useState(false);

    // Bayesian specific
    const [priorA, setPriorA] = useState(0.3);
    const [likelihoodBA, setLikelihoodBA] = useState(0.8);
    const [likelihoodBNotA, setLikelihoodBNotA] = useState(0.2);

    // Markov specific
    const [markovSteps, setMarkovSteps] = useState(10);
    const [markovHistory, setMarkovHistory] = useState<string[]>([]);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const parseData = (): number[] => {
        return inputData
            .split(',')
            .map(s => parseFloat(s.trim()))
            .filter(n => !isNaN(n));
    };

    const calculateMean = (data: number[]): number => {
        return data.reduce((a, b) => a + b, 0) / data.length;
    };

    const calculateMedian = (data: number[]): number => {
        const sorted = [...data].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const calculateMode = (data: number[]): number | string => {
        const freq: Record<number, number> = {};
        data.forEach(n => freq[n] = (freq[n] || 0) + 1);
        const maxFreq = Math.max(...Object.values(freq));
        const modes = Object.entries(freq).filter(([, f]) => f === maxFreq).map(([n]) => parseFloat(n));
        return modes.length === data.length ? 'No mode' : modes.join(', ');
    };

    const calculateStdDev = (data: number[]): number => {
        const mean = calculateMean(data);
        const variance = data.reduce((sum, n) => sum + Math.pow(n - mean, 2), 0) / data.length;
        return Math.sqrt(variance);
    };

    const calculateCorrelation = (x: number[], y: number[]): number => {
        const n = x.length;
        const meanX = calculateMean(x);
        const meanY = calculateMean(y);
        const num = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
        const denX = Math.sqrt(x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0));
        const denY = Math.sqrt(y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0));
        return num / (denX * denY);
    };

    // Bayesian Inference calculation
    const calculateBayesian = () => {
        const pB = likelihoodBA * priorA + likelihoodBNotA * (1 - priorA);
        const posteriorAB = (likelihoodBA * priorA) / pB;

        setResults({
            'Prior P(A)': priorA,
            'P(B|A)': likelihoodBA,
            'P(B|¬A)': likelihoodBNotA,
            'P(B)': pB,
            'Posterior P(A|B)': posteriorAB,
            'Update Factor': posteriorAB / priorA
        });
    };

    // Markov Chain simulation
    const runMarkovChain = async () => {
        setIsRunning(true);
        setMarkovHistory([]);

        const states = ['A', 'B', 'C'];
        const transitionMatrix = [
            [0.7, 0.2, 0.1],  // From A
            [0.3, 0.4, 0.3],  // From B
            [0.2, 0.3, 0.5]   // From C
        ];

        let currentState = 0; // Start at A
        const history: string[] = [states[currentState]];

        for (let i = 0; i < markovSteps; i++) {
            await sleep(200);

            const random = Math.random();
            let cumProb = 0;
            for (let j = 0; j < states.length; j++) {
                cumProb += transitionMatrix[currentState][j];
                if (random < cumProb) {
                    currentState = j;
                    break;
                }
            }

            history.push(states[currentState]);
            setMarkovHistory([...history]);
        }

        // Calculate stationary distribution
        const counts = { A: 0, B: 0, C: 0 };
        history.forEach(s => counts[s as keyof typeof counts]++);

        setResults({
            'Steps': markovSteps,
            'Final State': states[currentState],
            'P(A) Observed': counts.A / history.length,
            'P(B) Observed': counts.B / history.length,
            'P(C) Observed': counts.C / history.length
        });

        setIsRunning(false);
    };

    const calculate = () => {
        if (algorithmId === 'bayesian-inference') {
            calculateBayesian();
            return;
        }

        if (algorithmId === 'markov-chain') {
            runMarkovChain();
            return;
        }

        const data = parseData();
        if (data.length === 0) return;

        switch (algorithmId) {
            case 'descriptive-stats':
                setResults({
                    'Mean': calculateMean(data),
                    'Median': calculateMedian(data),
                    'Mode': calculateMode(data),
                    'Count': data.length,
                    'Sum': data.reduce((a, b) => a + b, 0),
                    'Min': Math.min(...data),
                    'Max': Math.max(...data)
                });
                break;
            case 'standard-deviation':
                const mean = calculateMean(data);
                const stdDev = calculateStdDev(data);
                setResults({
                    'Mean': mean,
                    'Variance': Math.pow(stdDev, 2),
                    'Std Dev (σ)': stdDev,
                    'Sample Std Dev': Math.sqrt(data.reduce((sum, n) => sum + Math.pow(n - mean, 2), 0) / (data.length - 1))
                });
                break;
            case 'correlation':
                const mid = Math.floor(data.length / 2);
                const x = data.slice(0, mid);
                const y = data.slice(mid, mid + x.length);
                if (x.length > 1 && y.length === x.length) {
                    const r = calculateCorrelation(x, y);
                    setResults({
                        'X values': x.join(', '),
                        'Y values': y.join(', '),
                        'Correlation (r)': r,
                        'R-squared': Math.pow(r, 2),
                        'Strength': r > 0.7 ? 'Strong +' : r > 0.3 ? 'Moderate +' : r > -0.3 ? 'Weak' : r > -0.7 ? 'Moderate -' : 'Strong -'
                    });
                }
                break;
            default:
                setResults({
                    'Mean': calculateMean(data),
                    'Std Dev': calculateStdDev(data)
                });
        }
    };

    const algorithmName = {
        'descriptive-stats': 'Descriptive Statistics',
        'standard-deviation': 'Standard Deviation',
        'correlation': 'Correlation Analysis',
        'bayesian-inference': 'Bayesian Inference',
        'markov-chain': 'Markov Chain'
    }[algorithmId] || 'Statistics';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-900">
                    <BarChart3 className="w-5 h-5 inline mr-2 text-teal-600" />
                    {algorithmName}
                </h3>
            </div>

            {/* Bayesian Inference UI */}
            {algorithmId === 'bayesian-inference' && (
                <div className="space-y-4">
                    <div className="p-4 bg-teal-50 rounded-lg border border-teal-200">
                        <p className="text-sm text-teal-700 mb-2">
                            <strong>Bayes' Theorem:</strong> P(A|B) = P(B|A) × P(A) / P(B)
                        </p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Prior P(A)</span>
                                <span className="font-mono text-teal-600">{priorA.toFixed(2)}</span>
                            </div>
                            <Slider value={[priorA * 100]} onValueChange={(v) => setPriorA(v[0] / 100)} min={5} max={95} step={5} />
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600">P(B|A)</span>
                                <span className="font-mono text-blue-600">{likelihoodBA.toFixed(2)}</span>
                            </div>
                            <Slider value={[likelihoodBA * 100]} onValueChange={(v) => setLikelihoodBA(v[0] / 100)} min={5} max={95} step={5} />
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600">P(B|¬A)</span>
                                <span className="font-mono text-orange-600">{likelihoodBNotA.toFixed(2)}</span>
                            </div>
                            <Slider value={[likelihoodBNotA * 100]} onValueChange={(v) => setLikelihoodBNotA(v[0] / 100)} min={5} max={95} step={5} />
                        </div>
                    </div>
                    <Button onClick={calculate} className="w-full bg-[#004040] hover:bg-[#003030]">
                        <Calculator className="w-4 h-4 mr-2" />
                        Calculate Posterior
                    </Button>
                </div>
            )}

            {/* Markov Chain UI */}
            {algorithmId === 'markov-chain' && (
                <div className="space-y-4">
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <p className="text-sm text-purple-700 mb-2"><strong>Transition Matrix:</strong></p>
                        <div className="grid grid-cols-4 gap-1 text-xs font-mono text-center">
                            <div></div><div className="font-bold">A</div><div className="font-bold">B</div><div className="font-bold">C</div>
                            <div className="font-bold">A</div><div className="bg-white rounded p-1">0.7</div><div className="bg-white rounded p-1">0.2</div><div className="bg-white rounded p-1">0.1</div>
                            <div className="font-bold">B</div><div className="bg-white rounded p-1">0.3</div><div className="bg-white rounded p-1">0.4</div><div className="bg-white rounded p-1">0.3</div>
                            <div className="font-bold">C</div><div className="bg-white rounded p-1">0.2</div><div className="bg-white rounded p-1">0.3</div><div className="bg-white rounded p-1">0.5</div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Simulation Steps</span>
                            <span className="font-mono text-purple-600">{markovSteps}</span>
                        </div>
                        <Slider value={[markovSteps]} onValueChange={(v) => setMarkovSteps(v[0])} min={5} max={50} step={5} disabled={isRunning} />
                    </div>
                    <Button onClick={calculate} disabled={isRunning} className="w-full bg-[#004040] hover:bg-[#003030]">
                        <Play className="w-4 h-4 mr-2" />
                        {isRunning ? 'Simulating...' : 'Run Simulation'}
                    </Button>

                    {/* Markov Chain History */}
                    {markovHistory.length > 0 && (
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <div className="text-xs text-gray-500 uppercase mb-2">State Sequence</div>
                            <div className="flex flex-wrap gap-1">
                                {markovHistory.map((state, i) => (
                                    <span
                                        key={i}
                                        className={`px-2 py-1 rounded text-xs font-bold ${state === 'A' ? 'bg-blue-100 text-blue-700' :
                                                state === 'B' ? 'bg-emerald-100 text-emerald-700' :
                                                    'bg-amber-100 text-amber-700'
                                            }`}
                                    >
                                        {state}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Standard Data Input */}
            {!['bayesian-inference', 'markov-chain'].includes(algorithmId) && (
                <div className="space-y-2">
                    <label className="text-sm text-gray-600">
                        Enter numbers (comma-separated):
                    </label>
                    <div className="flex gap-2">
                        <Input
                            value={inputData}
                            onChange={(e) => setInputData(e.target.value)}
                            placeholder="e.g., 1, 2, 3, 4, 5"
                            className="flex-1"
                        />
                        <Button onClick={calculate} className="bg-[#004040] hover:bg-[#003030]">
                            <Calculator className="w-4 h-4 mr-1" />
                            Calculate
                        </Button>
                        <Button variant="outline" onClick={() => { setResults(null); setInputData(''); }}>
                            <RotateCcw className="w-4 h-4" />
                        </Button>
                    </div>
                    {algorithmId === 'correlation' && (
                        <p className="text-xs text-gray-500">
                            For correlation: enter an even number of values. First half = X, second half = Y.
                        </p>
                    )}
                </div>
            )}

            {/* Results */}
            {results && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(results).map(([key, value]) => (
                        <div key={key} className="p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">{key}</div>
                            <div className="text-lg font-bold text-[#004040]">
                                {typeof value === 'number' ? value.toFixed(4) : value}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Data Visualization - bar chart */}
            {!['bayesian-inference', 'markov-chain'].includes(algorithmId) && parseData().length > 0 && (
                <div className="space-y-2">
                    <div className="text-xs text-gray-500 uppercase">Data Distribution</div>
                    <div className="h-32 bg-gray-50 rounded-lg border border-gray-100 p-4 flex items-end justify-center gap-1">
                        {parseData().map((val, idx) => {
                            const max = Math.max(...parseData());
                            return (
                                <div
                                    key={idx}
                                    className="bg-gradient-to-t from-teal-600 to-teal-400 rounded-t flex-1 max-w-8 transition-all duration-300 hover:from-teal-700 hover:to-teal-500"
                                    style={{ height: `${(val / max) * 100}%` }}
                                    title={val.toString()}
                                />
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};
