"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle, TrendingUp } from 'lucide-react';

interface RegressionVisualizerProps {
    algorithmId: string;
}

interface DataPoint {
    x: number;
    y: number;
    predicted?: number;
    residual?: number;
    state: 'default' | 'processing' | 'fitted';
}

interface RegressionLine {
    slope: number;
    intercept: number;
    r2: number;
    mse: number;
}

export const RegressionVisualizer = ({ algorithmId }: RegressionVisualizerProps) => {
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'fitting' | 'complete'>('idle');
    const [regressionLine, setRegressionLine] = useState<RegressionLine | null>(null);
    const [showResiduals, setShowResiduals] = useState(true);
    const [animationProgress, setAnimationProgress] = useState(0);
    const [noiseLevel, setNoiseLevel] = useState(15);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Generate sample regression data
    const generateData = useCallback(() => {
        const points: DataPoint[] = [];
        const trueSlope = 0.6 + Math.random() * 0.4;
        const trueIntercept = 10 + Math.random() * 20;

        for (let i = 0; i < 30; i++) {
            const x = 5 + Math.random() * 85;
            const noise = (Math.random() - 0.5) * noiseLevel * 2;
            const y = trueSlope * x + trueIntercept + noise;

            points.push({
                x,
                y: Math.max(5, Math.min(95, y)),
                state: 'default'
            });
        }

        setDataPoints(points);
        setRegressionLine(null);
        setStep('idle');
        setAnimationProgress(0);
    }, [noiseLevel]);

    useEffect(() => {
        generateData();
    }, [generateData]);

    // Calculate linear regression using least squares
    const calculateRegression = (points: DataPoint[]): RegressionLine => {
        const n = points.length;
        const sumX = points.reduce((s, p) => s + p.x, 0);
        const sumY = points.reduce((s, p) => s + p.y, 0);
        const sumXY = points.reduce((s, p) => s + p.x * p.y, 0);
        const sumX2 = points.reduce((s, p) => s + p.x * p.x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Calculate R²
        const meanY = sumY / n;
        const ssTotal = points.reduce((s, p) => s + Math.pow(p.y - meanY, 2), 0);
        const ssResidual = points.reduce((s, p) => s + Math.pow(p.y - (slope * p.x + intercept), 2), 0);
        const r2 = 1 - ssResidual / ssTotal;

        const mse = ssResidual / n;

        return { slope, intercept, r2, mse };
    };

    const runRegression = async () => {
        setIsRunning(true);
        setStep('fitting');
        setAnimationProgress(0);

        const pts = [...dataPoints];

        // Animate the fitting process
        for (let i = 0; i <= 20; i++) {
            setAnimationProgress(i * 5);

            // Calculate regression with current points
            const currentPoints = pts.slice(0, Math.max(3, Math.floor(pts.length * (i / 20))));
            if (currentPoints.length >= 3) {
                const line = calculateRegression(currentPoints);
                setRegressionLine(line);
            }

            await sleep(100);
        }

        // Final calculation with all points
        const finalLine = calculateRegression(pts);
        setRegressionLine(finalLine);

        // Calculate residuals and predictions
        pts.forEach(p => {
            p.predicted = finalLine.slope * p.x + finalLine.intercept;
            p.residual = p.y - p.predicted;
            p.state = 'fitted';
        });

        setDataPoints([...pts]);
        setStep('complete');
        setAnimationProgress(100);
        setIsRunning(false);
    };

    const getPointColor = (point: DataPoint) => {
        if (point.state === 'fitted' && point.residual !== undefined) {
            return Math.abs(point.residual) > 10 ? '#F59E0B' : '#10B981';
        }
        return '#3B82F6';
    };

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    <TrendingUp className="w-5 h-5 inline mr-2 text-blue-600" />
                    Linear Regression Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generateData}
                        disabled={isRunning}
                    >
                        <Shuffle className="w-4 h-4 mr-1" />
                        New Data
                    </Button>
                    <Button
                        size="sm"
                        onClick={runRegression}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Fitting...' : 'Fit Line'}
                    </Button>
                </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Noise Level</span>
                        <span className="font-mono text-orange-600">{noiseLevel}</span>
                    </div>
                    <Slider
                        value={[noiseLevel]}
                        onValueChange={(v) => setNoiseLevel(v[0])}
                        min={5}
                        max={30}
                        step={5}
                        disabled={isRunning}
                    />
                </div>
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="showResiduals"
                        checked={showResiduals}
                        onChange={(e) => setShowResiduals(e.target.checked)}
                        className="rounded"
                    />
                    <label htmlFor="showResiduals" className="text-sm text-gray-600">
                        Show Residuals
                    </label>
                </div>
            </div>

            {/* Visualization */}
            <div className="relative h-72 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    {/* Grid */}
                    <defs>
                        <pattern id="regGrid" width="10" height="10" patternUnits="userSpaceOnUse">
                            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#E5E7EB" strokeWidth="0.2" />
                        </pattern>
                    </defs>
                    <rect width="100" height="100" fill="url(#regGrid)" />

                    {/* Regression line */}
                    {regressionLine && (
                        <line
                            x1={0}
                            y1={100 - regressionLine.intercept}
                            x2={100}
                            y2={100 - (regressionLine.slope * 100 + regressionLine.intercept)}
                            stroke="#EF4444"
                            strokeWidth="1.5"
                            className="transition-all duration-300"
                        />
                    )}

                    {/* Residual lines */}
                    {showResiduals && regressionLine && dataPoints.map((point, idx) => (
                        point.predicted !== undefined && (
                            <line
                                key={`res-${idx}`}
                                x1={point.x}
                                y1={100 - point.y}
                                x2={point.x}
                                y2={100 - point.predicted}
                                stroke="#94A3B8"
                                strokeWidth="0.5"
                                strokeDasharray="1,1"
                                opacity="0.6"
                            />
                        )
                    ))}

                    {/* Data points */}
                    {dataPoints.map((point, idx) => (
                        <circle
                            key={idx}
                            cx={point.x}
                            cy={100 - point.y}
                            r={2.5}
                            fill={getPointColor(point)}
                            stroke="white"
                            strokeWidth="0.5"
                            className="transition-all duration-200"
                        />
                    ))}
                </svg>

                {/* Progress indicator */}
                {isRunning && (
                    <div className="absolute bottom-2 left-2 right-2">
                        <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-200"
                                style={{ width: `${animationProgress}%` }}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs flex-wrap">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-gray-600">Data Points</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-red-500" />
                    <span className="text-gray-600">Regression Line</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-amber-500" />
                    <span className="text-gray-600">High Residual</span>
                </div>
            </div>

            {/* Results */}
            {step === 'complete' && regressionLine && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200 text-center">
                        <div className="text-[10px] text-blue-600 uppercase font-bold mb-1">R² Score</div>
                        <div className="text-xl font-bold text-blue-700">{regressionLine.r2.toFixed(4)}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">MSE</div>
                        <div className="text-xl font-bold text-gray-700">{regressionLine.mse.toFixed(2)}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Slope (β₁)</div>
                        <div className="text-xl font-bold text-gray-700">{regressionLine.slope.toFixed(3)}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Intercept (β₀)</div>
                        <div className="text-xl font-bold text-gray-700">{regressionLine.intercept.toFixed(2)}</div>
                    </div>
                </div>
            )}

            {/* Equation display */}
            {step === 'complete' && regressionLine && (
                <div className="p-3 bg-gray-900 rounded-lg text-center">
                    <code className="text-emerald-400 font-mono text-sm">
                        ŷ = {regressionLine.slope.toFixed(3)}x + {regressionLine.intercept.toFixed(2)}
                    </code>
                </div>
            )}
        </div>
    );
};
