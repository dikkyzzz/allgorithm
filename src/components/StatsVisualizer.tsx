"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Calculator, RotateCcw } from 'lucide-react';

interface StatsVisualizerProps {
    algorithmId: string;
}

export const StatsVisualizer = ({ algorithmId }: StatsVisualizerProps) => {
    const [inputData, setInputData] = useState('12, 7, 9, 14, 7, 8, 11, 7, 10, 15');
    const [results, setResults] = useState<Record<string, number | string> | null>(null);

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

    const calculate = () => {
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
                    'Std Dev (Population)': stdDev,
                    'Std Dev (Sample)': Math.sqrt(data.reduce((sum, n) => sum + Math.pow(n - mean, 2), 0) / (data.length - 1))
                });
                break;
            case 'correlation':
                // Split data into two halves for correlation demo
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
                        'Relationship': r > 0.7 ? 'Strong positive' : r > 0.3 ? 'Moderate positive' : r > -0.3 ? 'Weak/None' : r > -0.7 ? 'Moderate negative' : 'Strong negative'
                    });
                } else {
                    setResults({ 'Error': 'Need even number of values (first half = X, second half = Y)' });
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
        'descriptive-stats': 'Mean/Median/Mode',
        'standard-deviation': 'Standard Deviation',
        'correlation': 'Correlation'
    }[algorithmId] || 'Statistics';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-900">
                    {algorithmName} Calculator
                </h3>
            </div>

            {/* Input */}
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
                    <Button
                        onClick={calculate}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Calculator className="w-4 h-4 mr-1" />
                        Calculate
                    </Button>
                    <Button
                        variant="outline"
                        onClick={() => { setResults(null); setInputData(''); }}
                    >
                        <RotateCcw className="w-4 h-4" />
                    </Button>
                </div>
            </div>

            {algorithmId === 'correlation' && (
                <p className="text-xs text-gray-500">
                    For correlation: enter an even number of values. First half will be X, second half will be Y.
                </p>
            )}

            {/* Results */}
            {results && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(results).map(([key, value]) => (
                        <div key={key} className="p-4 bg-gray-50 rounded-lg border border-gray-100 text-center">
                            <div className="text-xs text-gray-500 uppercase mb-1">{key}</div>
                            <div className="text-xl font-bold text-[#004040]">
                                {typeof value === 'number' ? value.toFixed(4) : value}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Visualization - bar chart for data distribution */}
            {parseData().length > 0 && (
                <div className="space-y-2">
                    <div className="text-xs text-gray-500 uppercase">Data Visualization</div>
                    <div className="h-32 bg-gray-50 rounded-lg border border-gray-100 p-4 flex items-end justify-center gap-1">
                        {parseData().map((val, idx) => {
                            const max = Math.max(...parseData());
                            return (
                                <div
                                    key={idx}
                                    className="bg-gradient-to-t from-[#004040] to-[#006060] rounded-t flex-1 max-w-8 transition-all duration-300 hover:from-[#006060] hover:to-[#008080]"
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
