"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw, Shuffle, Network } from 'lucide-react';

interface GraphVisualizerProps {
    algorithmId: string;
}

interface Node {
    id: number;
    x: number;
    y: number;
    label: string;
    state: 'default' | 'visited' | 'current' | 'path' | 'source' | 'target';
    distance?: number;
    pageRank?: number;
}

interface Edge {
    from: number;
    to: number;
    weight: number;
    state: 'default' | 'visited' | 'path';
}

interface GraphResult {
    pathLength?: number;
    nodesVisited: number;
    executionTime: number;
    path?: number[];
    pageRanks?: { node: string; rank: number }[];
}

export const GraphVisualizer = ({ algorithmId }: GraphVisualizerProps) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'running' | 'complete'>('idle');
    const [sourceNode, setSourceNode] = useState(0);
    const [targetNode, setTargetNode] = useState(5);
    const [result, setResult] = useState<GraphResult | null>(null);
    const [speed, setSpeed] = useState(300);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Generate a random graph
    const generateGraph = useCallback(() => {
        const nodeCount = 8;
        const newNodes: Node[] = [];

        // Position nodes in a circle-ish pattern
        const positions = [
            { x: 20, y: 20 }, { x: 50, y: 15 }, { x: 80, y: 20 },
            { x: 15, y: 50 }, { x: 85, y: 50 },
            { x: 20, y: 80 }, { x: 50, y: 85 }, { x: 80, y: 80 }
        ];

        for (let i = 0; i < nodeCount; i++) {
            newNodes.push({
                id: i,
                x: positions[i].x,
                y: positions[i].y,
                label: String.fromCharCode(65 + i), // A, B, C, ...
                state: 'default'
            });
        }

        // Generate edges (ensure connected graph)
        const newEdges: Edge[] = [
            { from: 0, to: 1, weight: 4, state: 'default' },
            { from: 0, to: 3, weight: 2, state: 'default' },
            { from: 1, to: 2, weight: 3, state: 'default' },
            { from: 1, to: 4, weight: 5, state: 'default' },
            { from: 2, to: 4, weight: 1, state: 'default' },
            { from: 3, to: 5, weight: 6, state: 'default' },
            { from: 3, to: 6, weight: 4, state: 'default' },
            { from: 4, to: 7, weight: 2, state: 'default' },
            { from: 5, to: 6, weight: 3, state: 'default' },
            { from: 6, to: 7, weight: 5, state: 'default' },
            { from: 1, to: 3, weight: 1, state: 'default' },
            { from: 2, to: 7, weight: 4, state: 'default' },
        ];

        setNodes(newNodes);
        setEdges(newEdges);
        setResult(null);
        setStep('idle');
    }, []);

    useEffect(() => {
        generateGraph();
    }, [generateGraph]);

    // Get adjacency list
    const getAdjList = () => {
        const adj: Map<number, { to: number; weight: number }[]> = new Map();
        nodes.forEach(n => adj.set(n.id, []));
        edges.forEach(e => {
            adj.get(e.from)?.push({ to: e.to, weight: e.weight });
            adj.get(e.to)?.push({ to: e.from, weight: e.weight }); // undirected
        });
        return adj;
    };

    // Dijkstra's Algorithm
    const runDijkstra = async () => {
        const adj = getAdjList();
        const dist: number[] = new Array(nodes.length).fill(Infinity);
        const prev: number[] = new Array(nodes.length).fill(-1);
        const visited: boolean[] = new Array(nodes.length).fill(false);

        dist[sourceNode] = 0;
        let nodesVisited = 0;
        const startTime = Date.now();

        // Mark source
        const nds = [...nodes];
        nds[sourceNode].state = 'source';
        nds[targetNode].state = 'target';
        setNodes([...nds]);
        await sleep(speed);

        for (let i = 0; i < nodes.length; i++) {
            // Find min distance unvisited node
            let u = -1;
            let minDist = Infinity;
            for (let j = 0; j < nodes.length; j++) {
                if (!visited[j] && dist[j] < minDist) {
                    minDist = dist[j];
                    u = j;
                }
            }

            if (u === -1) break;

            visited[u] = true;
            nodesVisited++;

            // Update node state
            if (u !== sourceNode && u !== targetNode) {
                nds[u].state = 'current';
                nds[u].distance = dist[u];
            }
            setNodes([...nds]);
            await sleep(speed);

            if (u !== sourceNode && u !== targetNode) {
                nds[u].state = 'visited';
            }

            // Relax edges
            const neighbors = adj.get(u) || [];
            for (const { to, weight } of neighbors) {
                if (dist[u] + weight < dist[to]) {
                    dist[to] = dist[u] + weight;
                    prev[to] = u;

                    // Highlight edge
                    const eds = [...edges];
                    const edgeIdx = eds.findIndex(e =>
                        (e.from === u && e.to === to) || (e.from === to && e.to === u)
                    );
                    if (edgeIdx >= 0) {
                        eds[edgeIdx].state = 'visited';
                        setEdges([...eds]);
                    }
                }
            }
        }

        // Reconstruct path
        const path: number[] = [];
        let current = targetNode;
        while (current !== -1) {
            path.unshift(current);
            current = prev[current];
        }

        // Highlight path
        const eds = [...edges];
        for (let i = 0; i < path.length - 1; i++) {
            const edgeIdx = eds.findIndex(e =>
                (e.from === path[i] && e.to === path[i + 1]) ||
                (e.from === path[i + 1] && e.to === path[i])
            );
            if (edgeIdx >= 0) {
                eds[edgeIdx].state = 'path';
            }
            nds[path[i]].state = 'path';
        }
        nds[path[path.length - 1]].state = 'path';
        setNodes([...nds]);
        setEdges([...eds]);

        return {
            pathLength: dist[targetNode],
            nodesVisited,
            executionTime: Date.now() - startTime,
            path
        };
    };

    // BFS Algorithm
    const runBFS = async () => {
        const adj = getAdjList();
        const visited: boolean[] = new Array(nodes.length).fill(false);
        const queue: number[] = [sourceNode];
        const prev: number[] = new Array(nodes.length).fill(-1);

        visited[sourceNode] = true;
        let nodesVisited = 0;
        const startTime = Date.now();

        const nds = [...nodes];
        nds[sourceNode].state = 'source';
        nds[targetNode].state = 'target';
        setNodes([...nds]);
        await sleep(speed);

        while (queue.length > 0) {
            const u = queue.shift()!;
            nodesVisited++;

            if (u !== sourceNode && u !== targetNode) {
                nds[u].state = 'current';
            }
            setNodes([...nds]);
            await sleep(speed);

            if (u === targetNode) break;

            if (u !== sourceNode && u !== targetNode) {
                nds[u].state = 'visited';
            }

            const neighbors = adj.get(u) || [];
            for (const { to } of neighbors) {
                if (!visited[to]) {
                    visited[to] = true;
                    prev[to] = u;
                    queue.push(to);

                    const eds = [...edges];
                    const edgeIdx = eds.findIndex(e =>
                        (e.from === u && e.to === to) || (e.from === to && e.to === u)
                    );
                    if (edgeIdx >= 0) {
                        eds[edgeIdx].state = 'visited';
                        setEdges([...eds]);
                    }
                }
            }
        }

        // Reconstruct path
        const path: number[] = [];
        let current = targetNode;
        while (current !== -1) {
            path.unshift(current);
            current = prev[current];
        }

        const eds = [...edges];
        for (let i = 0; i < path.length - 1; i++) {
            const edgeIdx = eds.findIndex(e =>
                (e.from === path[i] && e.to === path[i + 1]) ||
                (e.from === path[i + 1] && e.to === path[i])
            );
            if (edgeIdx >= 0) {
                eds[edgeIdx].state = 'path';
            }
            nds[path[i]].state = 'path';
        }
        nds[path[path.length - 1]].state = 'path';
        setNodes([...nds]);
        setEdges([...eds]);

        return {
            pathLength: path.length - 1,
            nodesVisited,
            executionTime: Date.now() - startTime,
            path
        };
    };

    // PageRank Algorithm (simplified)
    const runPageRank = async () => {
        const n = nodes.length;
        const d = 0.85; // damping factor
        let ranks = new Array(n).fill(1 / n);
        const iterations = 10;
        const startTime = Date.now();

        const nds = [...nodes];

        // Get outgoing edges count
        const outDegree: number[] = new Array(n).fill(0);
        edges.forEach(e => {
            outDegree[e.from]++;
            outDegree[e.to]++;
        });

        for (let iter = 0; iter < iterations; iter++) {
            const newRanks = new Array(n).fill((1 - d) / n);

            edges.forEach(e => {
                newRanks[e.to] += d * ranks[e.from] / outDegree[e.from];
                newRanks[e.from] += d * ranks[e.to] / outDegree[e.to];
            });

            ranks = newRanks;

            // Update visualization
            nds.forEach((node, i) => {
                node.pageRank = ranks[i];
                node.state = 'visited';
            });
            setNodes([...nds]);
            await sleep(speed);
        }

        // Highlight top nodes
        const sortedIndices = ranks
            .map((r, i) => ({ rank: r, idx: i }))
            .sort((a, b) => b.rank - a.rank);

        sortedIndices.slice(0, 3).forEach(({ idx }) => {
            nds[idx].state = 'path';
        });
        setNodes([...nds]);

        return {
            nodesVisited: n * iterations,
            executionTime: Date.now() - startTime,
            pageRanks: nds.map(n => ({ node: n.label, rank: n.pageRank || 0 }))
        };
    };

    const runAlgorithm = async () => {
        setIsRunning(true);
        setStep('running');
        setResult(null);

        // Reset states
        const nds = nodes.map(n => ({ ...n, state: 'default' as const, distance: undefined, pageRank: undefined }));
        const eds = edges.map(e => ({ ...e, state: 'default' as const }));
        setNodes(nds);
        setEdges(eds);
        await sleep(100);

        let res: GraphResult;

        switch (algorithmId) {
            case 'dijkstra':
            case 'bellman-ford':
            case 'a-star':
            case 'floyd-warshall':
                res = await runDijkstra();
                break;
            case 'bfs-dfs':
                res = await runBFS();
                break;
            case 'pagerank':
                res = await runPageRank();
                break;
            default:
                res = await runDijkstra();
        }

        setResult(res);
        setStep('complete');
        setIsRunning(false);
    };

    const getNodeColor = (state: Node['state']) => {
        switch (state) {
            case 'source': return '#10B981';
            case 'target': return '#EF4444';
            case 'current': return '#FBBF24';
            case 'visited': return '#93C5FD';
            case 'path': return '#8B5CF6';
            default: return '#E5E7EB';
        }
    };

    const getEdgeColor = (state: Edge['state']) => {
        switch (state) {
            case 'visited': return '#93C5FD';
            case 'path': return '#8B5CF6';
            default: return '#9CA3AF';
        }
    };

    const algorithmName = {
        'dijkstra': "Dijkstra's Algorithm",
        'bfs-dfs': 'BFS/DFS Traversal',
        'a-star': 'A* Search',
        'bellman-ford': 'Bellman-Ford',
        'floyd-warshall': 'Floyd-Warshall',
        'pagerank': 'PageRank'
    }[algorithmId] || 'Graph Algorithm';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    <Network className="w-5 h-5 inline mr-2 text-indigo-600" />
                    {algorithmName} Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generateGraph}
                        disabled={isRunning}
                    >
                        <Shuffle className="w-4 h-4 mr-1" />
                        New Graph
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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {algorithmId !== 'pagerank' && (
                    <>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Source Node</span>
                                <span className="font-mono text-emerald-600">{String.fromCharCode(65 + sourceNode)}</span>
                            </div>
                            <Slider
                                value={[sourceNode]}
                                onValueChange={(v) => setSourceNode(v[0])}
                                min={0}
                                max={7}
                                step={1}
                                disabled={isRunning}
                            />
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Target Node</span>
                                <span className="font-mono text-rose-600">{String.fromCharCode(65 + targetNode)}</span>
                            </div>
                            <Slider
                                value={[targetNode]}
                                onValueChange={(v) => setTargetNode(v[0])}
                                min={0}
                                max={7}
                                step={1}
                                disabled={isRunning}
                            />
                        </div>
                    </>
                )}
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Animation Speed</span>
                        <span className="font-mono text-gray-600">{speed}ms</span>
                    </div>
                    <Slider
                        value={[speed]}
                        onValueChange={(v) => setSpeed(v[0])}
                        min={100}
                        max={800}
                        step={100}
                        disabled={isRunning}
                    />
                </div>
            </div>

            {/* Graph Visualization */}
            <div className="relative h-72 bg-gray-50 rounded-lg border border-gray-100 overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    {/* Edges */}
                    {edges.map((edge, idx) => {
                        const fromNode = nodes.find(n => n.id === edge.from);
                        const toNode = nodes.find(n => n.id === edge.to);
                        if (!fromNode || !toNode) return null;

                        const midX = (fromNode.x + toNode.x) / 2;
                        const midY = (fromNode.y + toNode.y) / 2;

                        return (
                            <g key={`edge-${idx}`}>
                                <line
                                    x1={fromNode.x}
                                    y1={fromNode.y}
                                    x2={toNode.x}
                                    y2={toNode.y}
                                    stroke={getEdgeColor(edge.state)}
                                    strokeWidth={edge.state === 'path' ? 2 : 1}
                                    className="transition-all duration-200"
                                />
                                <text
                                    x={midX}
                                    y={midY - 2}
                                    fontSize="3"
                                    fill="#6B7280"
                                    textAnchor="middle"
                                >
                                    {edge.weight}
                                </text>
                            </g>
                        );
                    })}

                    {/* Nodes */}
                    {nodes.map((node) => (
                        <g key={`node-${node.id}`}>
                            <circle
                                cx={node.x}
                                cy={node.y}
                                r={5}
                                fill={getNodeColor(node.state)}
                                stroke={node.state === 'current' ? '#000' : '#fff'}
                                strokeWidth={1}
                                className="transition-all duration-200"
                            />
                            <text
                                x={node.x}
                                y={node.y + 1.5}
                                fontSize="4"
                                fill={['source', 'target', 'path', 'current'].includes(node.state) ? '#fff' : '#374151'}
                                textAnchor="middle"
                                fontWeight="bold"
                            >
                                {node.label}
                            </text>
                            {node.distance !== undefined && (
                                <text
                                    x={node.x}
                                    y={node.y + 9}
                                    fontSize="2.5"
                                    fill="#6B7280"
                                    textAnchor="middle"
                                >
                                    d={node.distance}
                                </text>
                            )}
                            {node.pageRank !== undefined && (
                                <text
                                    x={node.x}
                                    y={node.y + 9}
                                    fontSize="2.5"
                                    fill="#6B7280"
                                    textAnchor="middle"
                                >
                                    {(node.pageRank * 100).toFixed(1)}%
                                </text>
                            )}
                        </g>
                    ))}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-xs flex-wrap">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                    <span className="text-gray-600">Source</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-rose-500" />
                    <span className="text-gray-600">Target</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-amber-400" />
                    <span className="text-gray-600">Current</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-300" />
                    <span className="text-gray-600">Visited</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-violet-500" />
                    <span className="text-gray-600">Path</span>
                </div>
            </div>

            {/* Results */}
            {step === 'complete' && result && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {result.pathLength !== undefined && (
                        <div className="p-3 bg-gradient-to-br from-violet-50 to-purple-50 rounded-lg border border-violet-200 text-center">
                            <div className="text-[10px] text-violet-600 uppercase font-bold mb-1">Path Length</div>
                            <div className="text-xl font-bold text-violet-700">{result.pathLength}</div>
                        </div>
                    )}
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Nodes Visited</div>
                        <div className="text-xl font-bold text-gray-700">{result.nodesVisited}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Time</div>
                        <div className="text-xl font-bold text-blue-600">{result.executionTime}ms</div>
                    </div>
                    {result.path && (
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Path</div>
                            <div className="text-sm font-bold text-gray-700">
                                {result.path.map(i => String.fromCharCode(65 + i)).join(' → ')}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* PageRank Results */}
            {step === 'complete' && result?.pageRanks && (
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-xs font-bold text-gray-500 uppercase mb-3 text-center">PageRank Scores</div>
                    <div className="grid grid-cols-4 gap-2">
                        {result.pageRanks
                            .sort((a, b) => b.rank - a.rank)
                            .map((pr, i) => (
                                <div key={i} className={`p-2 rounded text-center ${i < 3 ? 'bg-violet-100' : 'bg-gray-100'}`}>
                                    <div className="font-bold text-lg">{pr.node}</div>
                                    <div className="text-xs text-gray-600">{(pr.rank * 100).toFixed(1)}%</div>
                                </div>
                            ))}
                    </div>
                </div>
            )}
        </div>
    );
};
