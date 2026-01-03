"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle, Layers } from 'lucide-react';

interface PCAVisualizerProps {
    algorithmId: string;
}

interface DataPoint {
    x: number;
    y: number;
    projected?: { x: number; y: number };
    state: 'default' | 'processing' | 'projected';
}

interface PCAResult {
    pc1: { x: number; y: number };
    pc2: { x: number; y: number };
    variance1: number;
    variance2: number;
    totalVariance: number;
    eigenvalue1: number;
    eigenvalue2: number;
}

export const PCAVisualizer = ({ algorithmId }: PCAVisualizerProps) => {
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'centering' | 'computing' | 'projecting' | 'complete'>('idle');
    const [pcaResult, setPcaResult] = useState<PCAResult | null>(null);
    const [showProjections, setShowProjections] = useState(true);
    const [nComponents, setNComponents] = useState(2);
    const [dataSpread, setDataSpread] = useState(20);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Generate correlated 2D data
    const generateData = useCallback(() => {
        const points: DataPoint[] = [];
        const centerX = 50;
        const centerY = 50;
        const angle = Math.PI / 4 + (Math.random() - 0.5) * 0.5; // ~45 degrees with some randomness

        for (let i = 0; i < 40; i++) {
            // Generate along the principal axis with some noise
            const t = (Math.random() - 0.5) * 60;
            const noise = (Math.random() - 0.5) * dataSpread;

            const x = centerX + t * Math.cos(angle) + noise * Math.sin(angle);
            const y = centerY + t * Math.sin(angle) - noise * Math.cos(angle);

            points.push({
                x: Math.max(5, Math.min(95, x)),
                y: Math.max(5, Math.min(95, y)),
                state: 'default'
            });
        }

        setDataPoints(points);
        setPcaResult(null);
        setStep('idle');
    }, [dataSpread]);

    useEffect(() => {
        generateData();
    }, [generateData]);

    // Simple PCA calculation
    const calculatePCA = (points: DataPoint[]): PCAResult => {
        const n = points.length;

        // Calculate mean
        const meanX = points.reduce((s, p) => s + p.x, 0) / n;
        const meanY = points.reduce((s, p) => s + p.y, 0) / n;

        // Center the data
        const centered = points.map(p => ({ x: p.x - meanX, y: p.y - meanY }));

        // Calculate covariance matrix
        const covXX = centered.reduce((s, p) => s + p.x * p.x, 0) / n;
        const covYY = centered.reduce((s, p) => s + p.y * p.y, 0) / n;
        const covXY = centered.reduce((s, p) => s + p.x * p.y, 0) / n;

        // Calculate eigenvalues using characteristic equation
        const trace = covXX + covYY;
        const det = covXX * covYY - covXY * covXY;
        const discriminant = Math.sqrt(trace * trace / 4 - det);

        const eigenvalue1 = trace / 2 + discriminant;
        const eigenvalue2 = trace / 2 - discriminant;

        // Calculate eigenvectors
        let pc1 = { x: 1, y: 0 };
        let pc2 = { x: 0, y: 1 };

        if (covXY !== 0) {
            pc1 = { x: eigenvalue1 - covYY, y: covXY };
            pc2 = { x: eigenvalue2 - covYY, y: covXY };
        } else if (covXX >= covYY) {
            pc1 = { x: 1, y: 0 };
            pc2 = { x: 0, y: 1 };
        } else {
            pc1 = { x: 0, y: 1 };
            pc2 = { x: 1, y: 0 };
        }

        // Normalize eigenvectors
        const norm1 = Math.sqrt(pc1.x * pc1.x + pc1.y * pc1.y);
        const norm2 = Math.sqrt(pc2.x * pc2.x + pc2.y * pc2.y);
        pc1 = { x: pc1.x / norm1, y: pc1.y / norm1 };
        pc2 = { x: pc2.x / norm2, y: pc2.y / norm2 };

        // Calculate variance explained
        const totalVar = eigenvalue1 + eigenvalue2;
        const variance1 = (eigenvalue1 / totalVar) * 100;
        const variance2 = (eigenvalue2 / totalVar) * 100;

        return {
            pc1,
            pc2,
            variance1,
            variance2,
            totalVariance: variance1 + variance2,
            eigenvalue1,
            eigenvalue2
        };
    };

    const runPCA = async () => {
        setIsRunning(true);
        setStep('centering');

        const pts = [...dataPoints];
        await sleep(500);

        setStep('computing');
        await sleep(700);

        const result = calculatePCA(pts);
        setPcaResult(result);

        setStep('projecting');

        // Calculate projections
        const meanX = pts.reduce((s, p) => s + p.x, 0) / pts.length;
        const meanY = pts.reduce((s, p) => s + p.y, 0) / pts.length;

        for (let i = 0; i < pts.length; i++) {
            pts[i].state = 'processing';
            setDataPoints([...pts]);
            await sleep(50);

            // Project onto PC1
            const centered = { x: pts[i].x - meanX, y: pts[i].y - meanY };
            const projection1 = centered.x * result.pc1.x + centered.y * result.pc1.y;

            pts[i].projected = {
                x: meanX + projection1 * result.pc1.x,
                y: meanY + projection1 * result.pc1.y
            };
            pts[i].state = 'projected';
            setDataPoints([...pts]);
        }

        setStep('complete');
        setIsRunning(false);
    };

    const getPointColor = (point: DataPoint) => {
        if (point.state === 'processing') return '#FBBF24';
        if (point.state === 'projected') return '#8B5CF6';
        return '#3B82F6';
    };

    // Calculate mean for drawing principal components
    const meanX = dataPoints.length > 0 ? dataPoints.reduce((s, p) => s + p.x, 0) / dataPoints.length : 50;
    const meanY = dataPoints.length > 0 ? dataPoints.reduce((s, p) => s + p.y, 0) / dataPoints.length : 50;

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    <Layers className="w-5 h-5 inline mr-2 text-violet-600" />
                    PCA Visualization
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
                        onClick={runPCA}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Computing...' : 'Run PCA'}
                    </Button>
                </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Data Spread (Noise)</span>
                        <span className="font-mono text-violet-600">{dataSpread}</span>
                    </div>
                    <Slider
                        value={[dataSpread]}
                        onValueChange={(v) => setDataSpread(v[0])}
                        min={5}
                        max={40}
                        step={5}
                        disabled={isRunning}
                    />
                </div>
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="showProjections"
                        checked={showProjections}
                        onChange={(e) => setShowProjections(e.target.checked)}
                        className="rounded"
                    />
                    <label htmlFor="showProjections" className="text-sm text-gray-600">
                        Show Projections onto PC1
                    </label>
                </div>
            </div>

            {/* Status */}
            <div className="text-sm text-center">
                {step === 'idle' && <span className="text-gray-500">Ready to compute PCA</span>}
                {step === 'centering' && <span className="text-amber-600">📊 Centering data...</span>}
                {step === 'computing' && <span className="text-blue-600">🔢 Computing covariance matrix & eigenvectors...</span>}
                {step === 'projecting' && <span className="text-violet-600">📐 Projecting points onto principal components...</span>}
                {step === 'complete' && <span className="text-emerald-600">✅ PCA Complete!</span>}
            </div>

            {/* Visualization */}
            <div className="relative h-72 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    {/* Grid */}
                    <defs>
                        <pattern id="pcaGrid" width="10" height="10" patternUnits="userSpaceOnUse">
                            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#E5E7EB" strokeWidth="0.2" />
                        </pattern>
                        <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                            <polygon points="0 0, 6 2, 0 4" fill="#7C3AED" />
                        </marker>
                        <marker id="arrowhead2" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                            <polygon points="0 0, 6 2, 0 4" fill="#F97316" />
                        </marker>
                    </defs>
                    <rect width="100" height="100" fill="url(#pcaGrid)" />

                    {/* Mean point */}
                    <circle
                        cx={meanX}
                        cy={100 - meanY}
                        r={3}
                        fill="none"
                        stroke="#374151"
                        strokeWidth="1"
                        strokeDasharray="2,2"
                    />

                    {/* Principal Components as arrows */}
                    {pcaResult && (
                        <>
                            {/* PC1 - primary direction (violet) */}
                            <line
                                x1={meanX - pcaResult.pc1.x * 40}
                                y1={100 - (meanY - pcaResult.pc1.y * 40)}
                                x2={meanX + pcaResult.pc1.x * 40}
                                y2={100 - (meanY + pcaResult.pc1.y * 40)}
                                stroke="#7C3AED"
                                strokeWidth="2"
                                markerEnd="url(#arrowhead)"
                            />

                            {/* PC2 - secondary direction (orange) */}
                            {nComponents === 2 && (
                                <line
                                    x1={meanX - pcaResult.pc2.x * 20}
                                    y1={100 - (meanY - pcaResult.pc2.y * 20)}
                                    x2={meanX + pcaResult.pc2.x * 20}
                                    y2={100 - (meanY + pcaResult.pc2.y * 20)}
                                    stroke="#F97316"
                                    strokeWidth="1.5"
                                    markerEnd="url(#arrowhead2)"
                                />
                            )}
                        </>
                    )}

                    {/* Projection lines */}
                    {showProjections && dataPoints.map((point, idx) => (
                        point.projected && (
                            <line
                                key={`proj-${idx}`}
                                x1={point.x}
                                y1={100 - point.y}
                                x2={point.projected.x}
                                y2={100 - point.projected.y}
                                stroke="#C4B5FD"
                                strokeWidth="0.5"
                                opacity="0.6"
                            />
                        )
                    ))}

                    {/* Data points */}
                    {dataPoints.map((point, idx) => (
                        <g key={idx}>
                            {/* Original point */}
                            <circle
                                cx={point.x}
                                cy={100 - point.y}
                                r={2.5}
                                fill={getPointColor(point)}
                                stroke="white"
                                strokeWidth="0.5"
                                className="transition-all duration-200"
                            />
                            {/* Projected point (smaller) */}
                            {showProjections && point.projected && (
                                <circle
                                    cx={point.projected.x}
                                    cy={100 - point.projected.y}
                                    r={1.5}
                                    fill="#DDD6FE"
                                    stroke="#7C3AED"
                                    strokeWidth="0.5"
                                />
                            )}
                        </g>
                    ))}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs flex-wrap">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-gray-600">Original Data</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-violet-600" />
                    <span className="text-gray-600">PC1 (Primary)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-orange-500" />
                    <span className="text-gray-600">PC2 (Secondary)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full border-2 border-violet-500 bg-violet-100" />
                    <span className="text-gray-600">Projected</span>
                </div>
            </div>

            {/* Variance Explained */}
            {step === 'complete' && pcaResult && (
                <div className="space-y-4">
                    {/* Variance bars */}
                    <div className="space-y-2">
                        <div className="text-xs font-bold text-gray-500 uppercase text-center">Variance Explained</div>
                        <div className="flex gap-2 items-center">
                            <span className="text-xs text-gray-500 w-10">PC1</span>
                            <div className="flex-1 h-6 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-violet-500 to-violet-600 rounded-full transition-all duration-500 flex items-center justify-end px-2"
                                    style={{ width: `${pcaResult.variance1}%` }}
                                >
                                    <span className="text-xs text-white font-bold">{pcaResult.variance1.toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-2 items-center">
                            <span className="text-xs text-gray-500 w-10">PC2</span>
                            <div className="flex-1 h-6 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-orange-400 to-orange-500 rounded-full transition-all duration-500 flex items-center justify-end px-2"
                                    style={{ width: `${pcaResult.variance2}%` }}
                                >
                                    <span className="text-xs text-white font-bold">{pcaResult.variance2.toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Stats grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-gradient-to-br from-violet-50 to-purple-50 rounded-lg border border-violet-200 text-center">
                            <div className="text-[10px] text-violet-600 uppercase font-bold mb-1">PC1 Variance</div>
                            <div className="text-xl font-bold text-violet-700">{pcaResult.variance1.toFixed(1)}%</div>
                        </div>
                        <div className="p-3 bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg border border-orange-200 text-center">
                            <div className="text-[10px] text-orange-600 uppercase font-bold mb-1">PC2 Variance</div>
                            <div className="text-xl font-bold text-orange-700">{pcaResult.variance2.toFixed(1)}%</div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">λ₁ (Eigenvalue)</div>
                            <div className="text-xl font-bold text-gray-700">{pcaResult.eigenvalue1.toFixed(2)}</div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">λ₂ (Eigenvalue)</div>
                            <div className="text-xl font-bold text-gray-700">{pcaResult.eigenvalue2.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
