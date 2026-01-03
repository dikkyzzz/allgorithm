"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle } from 'lucide-react';

interface ClassificationVisualizerProps {
    algorithmId: string;
}

interface DataPoint {
    x: number;
    y: number;
    actualClass: 0 | 1;
    predictedClass?: 0 | 1;
    state: 'default' | 'processing' | 'classified' | 'correct' | 'incorrect';
}

interface ConfusionMatrix {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
}

export const ClassificationVisualizer = ({ algorithmId }: ClassificationVisualizerProps) => {
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'training' | 'classifying' | 'complete'>('idle');
    const [iteration, setIteration] = useState(0);
    const [trainTestSplit, setTrainTestSplit] = useState(70);
    const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix | null>(null);
    const [decisionBoundary, setDecisionBoundary] = useState<{ slope: number, intercept: number } | null>(null);
    const [kNeighbors, setKNeighbors] = useState(3);
    const [maxDepth, setMaxDepth] = useState(5);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Generate sample 2D classification data
    const generateData = useCallback(() => {
        const points: DataPoint[] = [];

        // Class 0 - centered around (25, 65)
        for (let i = 0; i < 20; i++) {
            points.push({
                x: 15 + Math.random() * 30,
                y: 50 + Math.random() * 35,
                actualClass: 0,
                state: 'default'
            });
        }

        // Class 1 - centered around (70, 35)
        for (let i = 0; i < 20; i++) {
            points.push({
                x: 55 + Math.random() * 35,
                y: 15 + Math.random() * 40,
                actualClass: 1,
                state: 'default'
            });
        }

        // Some overlapping points for realism
        for (let i = 0; i < 10; i++) {
            points.push({
                x: 35 + Math.random() * 30,
                y: 30 + Math.random() * 35,
                actualClass: Math.random() > 0.5 ? 1 : 0,
                state: 'default'
            });
        }

        setDataPoints(points.sort(() => Math.random() - 0.5));
        setConfusionMatrix(null);
        setDecisionBoundary(null);
        setStep('idle');
        setIteration(0);
    }, []);

    useEffect(() => {
        generateData();
    }, [generateData]);

    // Simple linear classifier (for visualization purposes)
    const classifyPoint = (x: number, y: number, slope: number, intercept: number): 0 | 1 => {
        // Decision boundary: y = slope * x + intercept
        // Points above the line are class 0, below are class 1
        return y > slope * x + intercept ? 0 : 1;
    };

    // Calculate decision boundary based on algorithm
    const calculateDecisionBoundary = (trainData: DataPoint[]) => {
        // Simple linear separator for visualization
        const class0 = trainData.filter(p => p.actualClass === 0);
        const class1 = trainData.filter(p => p.actualClass === 1);

        const mean0 = {
            x: class0.reduce((s, p) => s + p.x, 0) / class0.length,
            y: class0.reduce((s, p) => s + p.y, 0) / class0.length
        };
        const mean1 = {
            x: class1.reduce((s, p) => s + p.x, 0) / class1.length,
            y: class1.reduce((s, p) => s + p.y, 0) / class1.length
        };

        // Midpoint
        const midX = (mean0.x + mean1.x) / 2;
        const midY = (mean0.y + mean1.y) / 2;

        // Perpendicular slope
        const dirSlope = (mean1.y - mean0.y) / (mean1.x - mean0.x);
        const perpSlope = -1 / dirSlope;

        // y - midY = perpSlope * (x - midX)
        // y = perpSlope * x - perpSlope * midX + midY
        const intercept = -perpSlope * midX + midY;

        return { slope: perpSlope, intercept };
    };

    // KNN classification
    const knnClassify = (point: DataPoint, trainData: DataPoint[], k: number): 0 | 1 => {
        const distances = trainData.map(p => ({
            point: p,
            dist: Math.sqrt(Math.pow(p.x - point.x, 2) + Math.pow(p.y - point.y, 2))
        }));
        distances.sort((a, b) => a.dist - b.dist);

        const neighbors = distances.slice(0, k);
        const votes = neighbors.reduce((sum, n) => sum + n.point.actualClass, 0);
        return votes > k / 2 ? 1 : 0;
    };

    const runClassification = async () => {
        setIsRunning(true);
        setConfusionMatrix(null);
        setStep('training');

        const pts = [...dataPoints];
        const splitIdx = Math.floor(pts.length * trainTestSplit / 100);
        const trainData = pts.slice(0, splitIdx);
        const testData = pts.slice(splitIdx);

        // Mark training data
        trainData.forEach(p => p.state = 'classified');
        setDataPoints([...pts]);
        await sleep(500);

        setIteration(1);

        // Calculate decision boundary for linear classifiers
        if (['logistic-regression', 'svm'].includes(algorithmId)) {
            const boundary = calculateDecisionBoundary(trainData);

            // Animate boundary drawing
            for (let i = 0; i <= 10; i++) {
                setDecisionBoundary({
                    slope: boundary.slope,
                    intercept: boundary.intercept * (i / 10)
                });
                await sleep(100);
            }
            setDecisionBoundary(boundary);
        }

        setStep('classifying');
        await sleep(300);

        // Classify test data
        let tp = 0, tn = 0, fp = 0, fn = 0;

        for (let i = splitIdx; i < pts.length; i++) {
            pts[i].state = 'processing';
            setDataPoints([...pts]);
            await sleep(150);

            let predicted: 0 | 1;

            switch (algorithmId) {
                case 'knn':
                    predicted = knnClassify(pts[i], trainData, kNeighbors);
                    break;
                case 'logistic-regression':
                case 'svm':
                    if (decisionBoundary) {
                        predicted = classifyPoint(pts[i].x, pts[i].y, decisionBoundary.slope, decisionBoundary.intercept);
                    } else {
                        const boundary = calculateDecisionBoundary(trainData);
                        setDecisionBoundary(boundary);
                        predicted = classifyPoint(pts[i].x, pts[i].y, boundary.slope, boundary.intercept);
                    }
                    break;
                default:
                    // Decision tree / Random Forest - use a simple rule-based classification
                    const xThreshold = 45;
                    const yThreshold = 45;
                    predicted = (pts[i].x > xThreshold || pts[i].y < yThreshold) ? 1 : 0;
            }

            pts[i].predictedClass = predicted;
            pts[i].state = predicted === pts[i].actualClass ? 'correct' : 'incorrect';

            // Update confusion matrix
            if (pts[i].actualClass === 1 && predicted === 1) tp++;
            if (pts[i].actualClass === 0 && predicted === 0) tn++;
            if (pts[i].actualClass === 0 && predicted === 1) fp++;
            if (pts[i].actualClass === 1 && predicted === 0) fn++;

            setDataPoints([...pts]);
            setIteration(i - splitIdx + 1);
        }

        setConfusionMatrix({ tp, tn, fp, fn });
        setStep('complete');
        setIsRunning(false);
    };

    const getPointColor = (point: DataPoint) => {
        if (point.state === 'processing') return '#FBBF24'; // yellow
        if (point.state === 'correct') return point.actualClass === 0 ? '#10B981' : '#3B82F6'; // green/blue
        if (point.state === 'incorrect') return '#EF4444'; // red
        if (point.state === 'classified') return point.actualClass === 0 ? '#6EE7B7' : '#93C5FD'; // light green/blue
        return point.actualClass === 0 ? '#10B981' : '#3B82F6'; // default: green for 0, blue for 1
    };

    const getPointStroke = (point: DataPoint) => {
        if (point.state === 'processing') return '#000';
        if (point.state === 'incorrect') return '#991B1B';
        return 'none';
    };

    const accuracy = confusionMatrix
        ? ((confusionMatrix.tp + confusionMatrix.tn) / (confusionMatrix.tp + confusionMatrix.tn + confusionMatrix.fp + confusionMatrix.fn) * 100)
        : 0;

    const algorithmName = {
        'logistic-regression': 'Logistic Regression',
        'svm': 'SVM',
        'decision-tree': 'Decision Tree',
        'random-forest': 'Random Forest',
        'knn': 'KNN'
    }[algorithmId] || 'Classification';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    {algorithmName} Visualization
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
                        onClick={runClassification}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Running...' : 'Run'}
                    </Button>
                </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Train/Test Split</span>
                        <span className="font-mono text-[#004040]">{trainTestSplit}% / {100 - trainTestSplit}%</span>
                    </div>
                    <Slider
                        value={[trainTestSplit]}
                        onValueChange={(v) => setTrainTestSplit(v[0])}
                        min={50}
                        max={90}
                        step={5}
                        disabled={isRunning}
                    />
                </div>

                {algorithmId === 'knn' && (
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-600">K Neighbors</span>
                            <span className="font-mono text-blue-600">{kNeighbors}</span>
                        </div>
                        <Slider
                            value={[kNeighbors]}
                            onValueChange={(v) => setKNeighbors(v[0])}
                            min={1}
                            max={11}
                            step={2}
                            disabled={isRunning}
                        />
                    </div>
                )}

                {['decision-tree', 'random-forest'].includes(algorithmId) && (
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Max Depth</span>
                            <span className="font-mono text-emerald-600">{maxDepth}</span>
                        </div>
                        <Slider
                            value={[maxDepth]}
                            onValueChange={(v) => setMaxDepth(v[0])}
                            min={2}
                            max={10}
                            step={1}
                            disabled={isRunning}
                        />
                    </div>
                )}
            </div>

            {/* Visualization */}
            <div className="relative h-72 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    {/* Grid lines */}
                    <defs>
                        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#E5E7EB" strokeWidth="0.2" />
                        </pattern>
                    </defs>
                    <rect width="100" height="100" fill="url(#grid)" />

                    {/* Decision boundary line (for linear classifiers) */}
                    {decisionBoundary && ['logistic-regression', 'svm'].includes(algorithmId) && (
                        <line
                            x1={0}
                            y1={100 - decisionBoundary.intercept}
                            x2={100}
                            y2={100 - (decisionBoundary.slope * 100 + decisionBoundary.intercept)}
                            stroke="#7C3AED"
                            strokeWidth="0.8"
                            strokeDasharray="2,2"
                            className="transition-all duration-300"
                        />
                    )}

                    {/* Decision regions for tree-based */}
                    {['decision-tree', 'random-forest'].includes(algorithmId) && step === 'complete' && (
                        <>
                            <rect x="45" y="0" width="55" height="100" fill="#3B82F6" fillOpacity="0.1" />
                            <rect x="0" y="55" width="45" height="45" fill="#10B981" fillOpacity="0.1" />
                            <line x1="45" y1="0" x2="45" y2="55" stroke="#6B7280" strokeWidth="0.5" strokeDasharray="2,2" />
                            <line x1="0" y1="55" x2="45" y2="55" stroke="#6B7280" strokeWidth="0.5" strokeDasharray="2,2" />
                        </>
                    )}

                    {/* Data points */}
                    {dataPoints.map((point, idx) => (
                        <circle
                            key={idx}
                            cx={point.x}
                            cy={100 - point.y}
                            r={point.state === 'processing' ? 3 : 2.5}
                            fill={getPointColor(point)}
                            stroke={getPointStroke(point)}
                            strokeWidth={point.state === 'processing' ? 0.8 : 0}
                            className="transition-all duration-200"
                        />
                    ))}
                </svg>

                {/* Step indicator */}
                <div className="absolute top-2 left-2 text-xs font-medium px-2 py-1 bg-white/80 rounded">
                    {step === 'idle' && 'Ready'}
                    {step === 'training' && '🔄 Training...'}
                    {step === 'classifying' && '🎯 Classifying...'}
                    {step === 'complete' && '✅ Complete'}
                </div>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 text-xs flex-wrap">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                    <span className="text-gray-600">Class 0 (Train)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-gray-600">Class 1 (Train)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-400" />
                    <span className="text-gray-600">Processing</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <span className="text-gray-600">Misclassified</span>
                </div>
            </div>

            {/* Stats & Confusion Matrix */}
            {step === 'complete' && confusionMatrix && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Accuracy Card */}
                    <div className="p-4 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg border border-emerald-200 text-center">
                        <div className="text-xs text-emerald-600 uppercase font-bold mb-1">Accuracy</div>
                        <div className="text-3xl font-bold text-emerald-700">{accuracy.toFixed(1)}%</div>
                        <div className="text-xs text-gray-500 mt-1">
                            {confusionMatrix.tp + confusionMatrix.tn} / {confusionMatrix.tp + confusionMatrix.tn + confusionMatrix.fp + confusionMatrix.fn} correct
                        </div>
                    </div>

                    {/* Confusion Matrix */}
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <div className="text-xs text-gray-600 uppercase font-bold mb-2 text-center">Confusion Matrix</div>
                        <div className="grid grid-cols-2 gap-1 text-center text-xs">
                            <div className="p-2 bg-emerald-100 rounded text-emerald-700 font-bold">
                                TN: {confusionMatrix.tn}
                            </div>
                            <div className="p-2 bg-rose-100 rounded text-rose-700 font-bold">
                                FP: {confusionMatrix.fp}
                            </div>
                            <div className="p-2 bg-rose-100 rounded text-rose-700 font-bold">
                                FN: {confusionMatrix.fn}
                            </div>
                            <div className="p-2 bg-emerald-100 rounded text-emerald-700 font-bold">
                                TP: {confusionMatrix.tp}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Iteration counter */}
            {isRunning && (
                <div className="flex justify-center">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-[#004040]">{iteration}</div>
                        <div className="text-xs text-gray-500 uppercase">
                            {step === 'classifying' ? 'Points Classified' : 'Iteration'}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
