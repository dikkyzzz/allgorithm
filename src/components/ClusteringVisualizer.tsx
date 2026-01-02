"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle } from 'lucide-react';

interface ClusteringVisualizerProps {
    algorithmId: string;
}

type Point = {
    x: number;
    y: number;
    cluster: number;
    state: 'default' | 'processing' | 'assigned';
};

type Centroid = {
    x: number;
    y: number;
    cluster: number;
};

const clusterColors = [
    '#3B82F6', // blue
    '#EF4444', // red
    '#10B981', // green
    '#F59E0B', // amber
    '#8B5CF6', // purple
    '#EC4899', // pink
    '#06B6D4', // cyan
    '#84CC16', // lime
];

export const ClusteringVisualizer = ({ algorithmId }: ClusteringVisualizerProps) => {
    const [points, setPoints] = useState<Point[]>([]);
    const [centroids, setCentroids] = useState<Centroid[]>([]);
    const [numClusters, setNumClusters] = useState(3);
    const [isRunning, setIsRunning] = useState(false);
    const [iteration, setIteration] = useState(0);
    const [dendrogramLevels, setDendrogramLevels] = useState<number[][]>([]);

    const generatePoints = useCallback(() => {
        const newPoints: Point[] = [];
        // Generate clustered data
        const centers = [
            { x: 25, y: 25 },
            { x: 75, y: 25 },
            { x: 50, y: 75 },
            { x: 25, y: 75 },
            { x: 75, y: 75 },
        ];

        for (let i = 0; i < 30; i++) {
            const center = centers[i % centers.length];
            newPoints.push({
                x: center.x + (Math.random() - 0.5) * 25,
                y: center.y + (Math.random() - 0.5) * 25,
                cluster: -1,
                state: 'default'
            });
        }
        setPoints(newPoints);
        setCentroids([]);
        setIteration(0);
        setDendrogramLevels([]);
    }, []);

    useEffect(() => {
        generatePoints();
    }, [generatePoints]);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const distance = (p1: { x: number; y: number }, p2: { x: number; y: number }) => {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    };

    // K-Means Algorithm
    const runKMeans = async () => {
        const pts = [...points];

        // Initialize random centroids
        const initialCentroids: Centroid[] = [];
        const shuffled = [...pts].sort(() => Math.random() - 0.5);
        for (let i = 0; i < numClusters; i++) {
            initialCentroids.push({
                x: shuffled[i].x,
                y: shuffled[i].y,
                cluster: i
            });
        }
        setCentroids(initialCentroids);
        await sleep(500);

        let changed = true;
        let iter = 0;
        const maxIter = 20;

        while (changed && iter < maxIter) {
            changed = false;
            iter++;
            setIteration(iter);

            // Assign points to nearest centroid
            for (let i = 0; i < pts.length; i++) {
                pts[i].state = 'processing';
                setPoints([...pts]);
                await sleep(30);

                let minDist = Infinity;
                let newCluster = 0;
                for (let j = 0; j < initialCentroids.length; j++) {
                    const d = distance(pts[i], initialCentroids[j]);
                    if (d < minDist) {
                        minDist = d;
                        newCluster = j;
                    }
                }

                if (pts[i].cluster !== newCluster) {
                    changed = true;
                    pts[i].cluster = newCluster;
                }
                pts[i].state = 'assigned';
                setPoints([...pts]);
            }

            await sleep(300);

            // Update centroids
            for (let i = 0; i < initialCentroids.length; i++) {
                const clusterPoints = pts.filter(p => p.cluster === i);
                if (clusterPoints.length > 0) {
                    initialCentroids[i].x = clusterPoints.reduce((s, p) => s + p.x, 0) / clusterPoints.length;
                    initialCentroids[i].y = clusterPoints.reduce((s, p) => s + p.y, 0) / clusterPoints.length;
                }
            }
            setCentroids([...initialCentroids]);
            await sleep(300);
        }
    };

    // Hierarchical Clustering (Agglomerative)
    const runHierarchical = async () => {
        const pts = [...points];

        // Initially each point is its own cluster
        let clusters: number[][] = pts.map((_, i) => [i]);
        pts.forEach((p, i) => p.cluster = i);
        setPoints([...pts]);
        await sleep(300);

        const levels: number[][] = [];
        let clusterNum = pts.length;

        while (clusters.length > numClusters) {
            setIteration(pts.length - clusters.length + 1);

            // Find two closest clusters
            let minDist = Infinity;
            let mergeI = 0, mergeJ = 1;

            for (let i = 0; i < clusters.length; i++) {
                for (let j = i + 1; j < clusters.length; j++) {
                    // Complete linkage - max distance
                    let maxD = 0;
                    for (const pi of clusters[i]) {
                        for (const pj of clusters[j]) {
                            const d = distance(pts[pi], pts[pj]);
                            if (d > maxD) maxD = d;
                        }
                    }
                    if (maxD < minDist) {
                        minDist = maxD;
                        mergeI = i;
                        mergeJ = j;
                    }
                }
            }

            // Highlight merging clusters
            for (const idx of clusters[mergeI]) {
                pts[idx].state = 'processing';
            }
            for (const idx of clusters[mergeJ]) {
                pts[idx].state = 'processing';
            }
            setPoints([...pts]);
            await sleep(400);

            // Merge clusters
            const newCluster = [...clusters[mergeI], ...clusters[mergeJ]];
            const newClusterNum = Math.min(
                pts[clusters[mergeI][0]].cluster,
                pts[clusters[mergeJ][0]].cluster
            );

            for (const idx of newCluster) {
                pts[idx].cluster = newClusterNum;
                pts[idx].state = 'assigned';
            }
            setPoints([...pts]);

            levels.push([...clusters.map(c => c.length)]);
            setDendrogramLevels([...levels]);

            clusters = clusters.filter((_, i) => i !== mergeI && i !== mergeJ);
            clusters.push(newCluster);

            await sleep(300);
        }

        // Final coloring
        clusters.forEach((cluster, i) => {
            for (const idx of cluster) {
                pts[idx].cluster = i;
            }
        });
        setPoints([...pts]);
    };

    // DBSCAN
    const runDBSCAN = async () => {
        const pts = [...points];
        const eps = 12;
        const minPts = 3;
        let clusterNum = 0;

        pts.forEach(p => p.cluster = -1); // -1 = unvisited
        setPoints([...pts]);

        for (let i = 0; i < pts.length; i++) {
            if (pts[i].cluster !== -1) continue;

            pts[i].state = 'processing';
            setPoints([...pts]);
            await sleep(100);

            // Find neighbors
            const neighbors: number[] = [];
            for (let j = 0; j < pts.length; j++) {
                if (distance(pts[i], pts[j]) <= eps) {
                    neighbors.push(j);
                }
            }

            if (neighbors.length < minPts) {
                pts[i].cluster = -2; // Noise
                pts[i].state = 'assigned';
                setPoints([...pts]);
                continue;
            }

            // Start new cluster
            pts[i].cluster = clusterNum;
            pts[i].state = 'assigned';
            setIteration(clusterNum + 1);

            const seeds = [...neighbors];
            while (seeds.length > 0) {
                const q = seeds.shift()!;

                if (pts[q].cluster === -2) {
                    pts[q].cluster = clusterNum;
                    pts[q].state = 'assigned';
                }
                if (pts[q].cluster !== -1) continue;

                pts[q].cluster = clusterNum;
                pts[q].state = 'processing';
                setPoints([...pts]);
                await sleep(50);

                const qNeighbors: number[] = [];
                for (let j = 0; j < pts.length; j++) {
                    if (distance(pts[q], pts[j]) <= eps) {
                        qNeighbors.push(j);
                    }
                }

                if (qNeighbors.length >= minPts) {
                    seeds.push(...qNeighbors);
                }

                pts[q].state = 'assigned';
                setPoints([...pts]);
            }

            clusterNum++;
        }
    };

    const runAlgorithm = async () => {
        if (isRunning) return;
        setIsRunning(true);
        setIteration(0);
        setCentroids([]);
        setDendrogramLevels([]);

        // Reset clusters
        const resetPts = points.map(p => ({ ...p, cluster: -1, state: 'default' as const }));
        setPoints(resetPts);
        await sleep(100);

        switch (algorithmId) {
            case 'k-means':
                await runKMeans();
                break;
            case 'hierarchical-clustering':
                await runHierarchical();
                break;
            case 'dbscan':
                await runDBSCAN();
                break;
            default:
                await runKMeans();
        }

        setIsRunning(false);
    };

    const getPointColor = (point: Point) => {
        if (point.cluster === -1) return '#9CA3AF'; // gray
        if (point.cluster === -2) return '#374151'; // dark gray (noise)
        return clusterColors[point.cluster % clusterColors.length];
    };

    const algorithmName = {
        'k-means': 'K-Means',
        'hierarchical-clustering': 'Hierarchical',
        'dbscan': 'DBSCAN'
    }[algorithmId] || 'Clustering';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    {algorithmName} Clustering Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generatePoints}
                        disabled={isRunning}
                    >
                        <Shuffle className="w-4 h-4 mr-1" />
                        New Data
                    </Button>
                    <Button
                        size="sm"
                        onClick={runAlgorithm}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Running...' : 'Run'}
                    </Button>
                </div>
            </div>

            {/* Parameters */}
            {algorithmId !== 'dbscan' && (
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Number of Clusters</span>
                        <span className="font-mono text-[#004040]">{numClusters}</span>
                    </div>
                    <Slider
                        value={[numClusters]}
                        onValueChange={(v) => setNumClusters(v[0])}
                        min={2}
                        max={6}
                        step={1}
                        disabled={isRunning}
                    />
                </div>
            )}

            {/* Visualization */}
            <div className="relative h-72 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    {/* Points */}
                    {points.map((point, idx) => (
                        <circle
                            key={idx}
                            cx={point.x}
                            cy={point.y}
                            r={point.state === 'processing' ? 2.5 : 2}
                            fill={getPointColor(point)}
                            stroke={point.state === 'processing' ? '#000' : 'none'}
                            strokeWidth={0.5}
                            className="transition-all duration-150"
                        />
                    ))}

                    {/* Centroids (for K-Means) */}
                    {centroids.map((centroid, idx) => (
                        <g key={`c-${idx}`}>
                            <circle
                                cx={centroid.x}
                                cy={centroid.y}
                                r={3}
                                fill={clusterColors[centroid.cluster % clusterColors.length]}
                                stroke="#000"
                                strokeWidth={0.5}
                            />
                            <line
                                x1={centroid.x - 2}
                                y1={centroid.y}
                                x2={centroid.x + 2}
                                y2={centroid.y}
                                stroke="#000"
                                strokeWidth={0.5}
                            />
                            <line
                                x1={centroid.x}
                                y1={centroid.y - 2}
                                x2={centroid.x}
                                y2={centroid.y + 2}
                                stroke="#000"
                                strokeWidth={0.5}
                            />
                        </g>
                    ))}
                </svg>
            </div>

            {/* Stats */}
            <div className="flex justify-center gap-8">
                <div className="text-center">
                    <div className="text-2xl font-bold text-[#004040]">{iteration}</div>
                    <div className="text-xs text-gray-500 uppercase">
                        {algorithmId === 'hierarchical-clustering' ? 'Merges' : 'Iterations'}
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-2xl font-bold text-gray-700">
                        {new Set(points.filter(p => p.cluster >= 0).map(p => p.cluster)).size}
                    </div>
                    <div className="text-xs text-gray-500 uppercase">Clusters Found</div>
                </div>
                {algorithmId === 'dbscan' && (
                    <div className="text-center">
                        <div className="text-2xl font-bold text-gray-400">
                            {points.filter(p => p.cluster === -2).length}
                        </div>
                        <div className="text-xs text-gray-500 uppercase">Noise Points</div>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-xs flex-wrap">
                {Array.from(new Set(points.filter(p => p.cluster >= 0).map(p => p.cluster))).slice(0, 6).map(c => (
                    <div key={c} className="flex items-center gap-1">
                        <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: clusterColors[c % clusterColors.length] }}
                        />
                        <span className="text-gray-600">Cluster {c + 1}</span>
                    </div>
                ))}
                {algorithmId === 'dbscan' && points.some(p => p.cluster === -2) && (
                    <div className="flex items-center gap-1">
                        <div className="w-3 h-3 rounded-full bg-gray-700" />
                        <span className="text-gray-600">Noise</span>
                    </div>
                )}
            </div>
        </div>
    );
};
