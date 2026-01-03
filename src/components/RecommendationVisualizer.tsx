"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RefreshCw, Star, User, Package, Database } from 'lucide-react';

interface RecommendationVisualizerProps {
    algorithmId: string;
}

interface UserItem {
    id: number;
    name: string;
    ratings: number[];
}

interface CacheItem {
    key: string;
    value: string;
    accessTime: number;
}

export const RecommendationVisualizer = ({ algorithmId }: RecommendationVisualizerProps) => {
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'processing' | 'complete'>('idle');
    const [recommendations, setRecommendations] = useState<{ item: string, score: number }[]>([]);
    const [selectedUser, setSelectedUser] = useState(0);
    const [animationSteps, setAnimationSteps] = useState<string[]>([]);

    // Collaborative filtering data
    const [users] = useState<UserItem[]>([
        { id: 0, name: 'Alice', ratings: [5, 3, 0, 1, 4] },
        { id: 1, name: 'Bob', ratings: [4, 0, 4, 1, 2] },
        { id: 2, name: 'Carol', ratings: [1, 1, 0, 5, 4] },
        { id: 3, name: 'Dave', ratings: [0, 4, 5, 4, 0] },
        { id: 4, name: 'Eve', ratings: [5, 4, 3, 0, 3] },
    ]);

    const [items] = useState(['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']);

    // LRU Cache state
    const [cache, setCache] = useState<CacheItem[]>([]);
    const [cacheSize, setCacheSize] = useState(4);
    const [cacheAccess, setCacheAccess] = useState('');
    const [cacheHits, setCacheHits] = useState(0);
    const [cacheMisses, setCacheMisses] = useState(0);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Cosine similarity
    const cosineSimilarity = (a: number[], b: number[]): number => {
        let dotProduct = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return normA && normB ? dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
    };

    // Collaborative Filtering
    const runCollaborativeFiltering = async () => {
        setIsRunning(true);
        setStep('processing');
        setAnimationSteps([]);
        setRecommendations([]);

        const targetUser = users[selectedUser];
        setAnimationSteps(prev => [...prev, `Finding similar users to ${targetUser.name}...`]);
        await sleep(500);

        // Calculate similarities
        const similarities: { user: UserItem, sim: number }[] = [];
        for (const user of users) {
            if (user.id !== targetUser.id) {
                const sim = cosineSimilarity(targetUser.ratings, user.ratings);
                similarities.push({ user, sim });
                setAnimationSteps(prev => [...prev, `Similarity with ${user.name}: ${sim.toFixed(3)}`]);
                await sleep(300);
            }
        }

        similarities.sort((a, b) => b.sim - a.sim);
        setAnimationSteps(prev => [...prev, `Most similar: ${similarities[0].user.name} (${similarities[0].sim.toFixed(3)})`]);
        await sleep(400);

        // Predict ratings for unrated items
        const predictions: { item: string, score: number }[] = [];
        for (let i = 0; i < items.length; i++) {
            if (targetUser.ratings[i] === 0) {
                let weightedSum = 0, simSum = 0;
                for (const { user, sim } of similarities) {
                    if (user.ratings[i] > 0) {
                        weightedSum += sim * user.ratings[i];
                        simSum += Math.abs(sim);
                    }
                }
                const predicted = simSum > 0 ? weightedSum / simSum : 0;
                predictions.push({ item: items[i], score: predicted });
            }
        }

        predictions.sort((a, b) => b.score - a.score);
        setAnimationSteps(prev => [...prev, `✓ Generated ${predictions.length} recommendations!`]);

        setRecommendations(predictions);
        setStep('complete');
        setIsRunning(false);
    };

    // Content-Based Filtering
    const runContentBasedFiltering = async () => {
        setIsRunning(true);
        setStep('processing');
        setAnimationSteps([]);
        setRecommendations([]);

        const targetUser = users[selectedUser];
        setAnimationSteps(prev => [...prev, `Analyzing ${targetUser.name}'s preferences...`]);
        await sleep(500);

        // Find highest rated items
        const likedItems: number[] = [];
        targetUser.ratings.forEach((r, i) => {
            if (r >= 4) likedItems.push(i);
        });

        setAnimationSteps(prev => [...prev, `Liked items: ${likedItems.map(i => items[i]).join(', ')}`]);
        await sleep(400);

        // Simulate content similarity (item features)
        const itemFeatures = [
            [1, 0, 1, 0, 1], // Movie A: action, drama, sci-fi
            [0, 1, 1, 1, 0], // Movie B: comedy, drama, romance
            [1, 0, 0, 0, 1], // Movie C: action, sci-fi
            [0, 1, 0, 1, 0], // Movie D: comedy, romance
            [1, 1, 1, 0, 0], // Movie E: action, comedy, drama
        ];

        setAnimationSteps(prev => [...prev, 'Computing content similarity...']);
        await sleep(400);

        const predictions: { item: string, score: number }[] = [];
        for (let i = 0; i < items.length; i++) {
            if (targetUser.ratings[i] === 0) {
                let score = 0;
                for (const liked of likedItems) {
                    score += cosineSimilarity(itemFeatures[i], itemFeatures[liked]);
                }
                predictions.push({ item: items[i], score: score / likedItems.length * 5 });
                setAnimationSteps(prev => [...prev, `${items[i]} similarity score: ${(score / likedItems.length).toFixed(3)}`]);
                await sleep(200);
            }
        }

        predictions.sort((a, b) => b.score - a.score);
        setAnimationSteps(prev => [...prev, '✓ Content-based recommendations ready!']);

        setRecommendations(predictions);
        setStep('complete');
        setIsRunning(false);
    };

    // LRU Cache
    const accessCache = async (key: string) => {
        if (!key.trim()) return;

        setIsRunning(true);
        setAnimationSteps([]);

        const existing = cache.find(c => c.key === key);

        if (existing) {
            // Cache hit
            setAnimationSteps(prev => [...prev, `✓ Cache HIT for "${key}"`]);
            setCacheHits(h => h + 1);

            // Move to front (most recently used)
            const newCache = cache.filter(c => c.key !== key);
            newCache.unshift({ ...existing, accessTime: Date.now() });
            setCache(newCache);
        } else {
            // Cache miss
            setAnimationSteps(prev => [...prev, `✗ Cache MISS for "${key}"`]);
            setCacheMisses(m => m + 1);
            await sleep(300);

            const newItem: CacheItem = { key, value: `Data for ${key}`, accessTime: Date.now() };

            if (cache.length >= cacheSize) {
                // Evict LRU
                const evicted = cache[cache.length - 1];
                setAnimationSteps(prev => [...prev, `Evicting LRU: "${evicted.key}"`]);
                await sleep(300);
                setCache([newItem, ...cache.slice(0, -1)]);
            } else {
                setCache([newItem, ...cache]);
            }

            setAnimationSteps(prev => [...prev, `Added "${key}" to cache`]);
        }

        setCacheAccess('');
        setIsRunning(false);
    };

    const runAlgorithm = () => {
        switch (algorithmId) {
            case 'collaborative-filtering':
                runCollaborativeFiltering();
                break;
            case 'content-based-filtering':
                runContentBasedFiltering();
                break;
            case 'lru-cache':
                // LRU has its own UI
                break;
        }
    };

    const algorithmName = {
        'collaborative-filtering': 'Collaborative Filtering',
        'content-based-filtering': 'Content-Based Filtering',
        'lru-cache': 'LRU Cache'
    }[algorithmId] || 'Recommendation';

    const isLRU = algorithmId === 'lru-cache';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    {isLRU ? <Database className="w-5 h-5 inline mr-2 text-pink-600" /> : <Star className="w-5 h-5 inline mr-2 text-pink-600" />}
                    {algorithmName}
                </h3>
                {!isLRU && (
                    <Button
                        size="sm"
                        onClick={runAlgorithm}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Computing...' : 'Get Recommendations'}
                    </Button>
                )}
            </div>

            {/* Collaborative / Content-Based UI */}
            {!isLRU && (
                <>
                    {/* User-Item Matrix */}
                    <div className="overflow-x-auto">
                        <div className="text-xs text-gray-500 uppercase mb-2">User-Item Rating Matrix</div>
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="bg-gray-50">
                                    <th className="p-2 text-left">User</th>
                                    {items.map((item, i) => (
                                        <th key={i} className="p-2 text-center">{item}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {users.map((user, ui) => (
                                    <tr
                                        key={ui}
                                        className={`border-t ${ui === selectedUser ? 'bg-pink-50' : ''}`}
                                        onClick={() => setSelectedUser(ui)}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        <td className="p-2 font-medium">
                                            <User className="w-4 h-4 inline mr-1" />
                                            {user.name}
                                        </td>
                                        {user.ratings.map((r, ri) => (
                                            <td key={ri} className="p-2 text-center">
                                                {r === 0 ? (
                                                    <span className="text-gray-300">-</span>
                                                ) : (
                                                    <span className="flex justify-center gap-0.5">
                                                        {Array(r).fill(0).map((_, i) => (
                                                            <Star key={i} className="w-3 h-3 text-amber-400 fill-amber-400" />
                                                        ))}
                                                    </span>
                                                )}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <p className="text-xs text-gray-500 mt-2">Click a user row to select them for recommendations</p>
                    </div>

                    {/* Animation Steps */}
                    {animationSteps.length > 0 && (
                        <div className="p-4 bg-gray-900 rounded-lg font-mono text-sm space-y-1 max-h-40 overflow-y-auto">
                            {animationSteps.map((step, i) => (
                                <div key={i} className={`${step.startsWith('✓') ? 'text-emerald-400' : 'text-gray-300'}`}>
                                    {step}
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Recommendations */}
                    {step === 'complete' && recommendations.length > 0 && (
                        <div className="space-y-2">
                            <div className="text-xs text-gray-500 uppercase">Recommendations for {users[selectedUser].name}</div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                {recommendations.map((rec, i) => (
                                    <div
                                        key={i}
                                        className="p-4 bg-gradient-to-br from-pink-50 to-rose-50 rounded-lg border border-pink-200"
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="font-medium">{rec.item}</span>
                                            <Package className="w-4 h-4 text-pink-600" />
                                        </div>
                                        <div className="flex items-center gap-1 mt-2">
                                            <div className="flex-1 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className="bg-pink-500 rounded-full h-2"
                                                    style={{ width: `${(rec.score / 5) * 100}%` }}
                                                />
                                            </div>
                                            <span className="text-sm font-bold text-pink-700">{rec.score.toFixed(1)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* LRU Cache UI */}
            {isLRU && (
                <div className="space-y-4">
                    {/* Cache Size */}
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Cache Size</span>
                            <span className="font-mono text-pink-600">{cacheSize}</span>
                        </div>
                        <Slider
                            value={[cacheSize]}
                            onValueChange={(v) => { setCacheSize(v[0]); setCache([]); setCacheHits(0); setCacheMisses(0); }}
                            min={2}
                            max={6}
                            step={1}
                            disabled={isRunning}
                        />
                    </div>

                    {/* Access Input */}
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={cacheAccess}
                            onChange={(e) => setCacheAccess(e.target.value)}
                            placeholder="Enter key (A, B, C...)"
                            className="flex-1 px-3 py-2 border rounded-lg text-sm"
                            onKeyDown={(e) => e.key === 'Enter' && accessCache(cacheAccess)}
                        />
                        <Button
                            onClick={() => accessCache(cacheAccess)}
                            disabled={isRunning || !cacheAccess.trim()}
                            className="bg-[#004040] hover:bg-[#003030]"
                        >
                            Access
                        </Button>
                        <Button
                            variant="outline"
                            onClick={() => { setCache([]); setCacheHits(0); setCacheMisses(0); setAnimationSteps([]); }}
                        >
                            <RefreshCw className="w-4 h-4" />
                        </Button>
                    </div>

                    {/* Cache Visualization */}
                    <div className="space-y-2">
                        <div className="text-xs text-gray-500 uppercase">Cache State (Most Recent → Least Recent)</div>
                        <div className="flex gap-2 flex-wrap">
                            {cache.length === 0 ? (
                                <div className="text-gray-400 text-sm">Cache is empty</div>
                            ) : (
                                cache.map((item, i) => (
                                    <div
                                        key={item.key}
                                        className={`px-4 py-2 rounded-lg border-2 ${i === 0 ? 'border-pink-400 bg-pink-50' : 'border-gray-200 bg-gray-50'}`}
                                    >
                                        <div className="font-bold text-gray-700">{item.key}</div>
                                        <div className="text-xs text-gray-500">{item.value}</div>
                                    </div>
                                ))
                            )}
                            {Array(cacheSize - cache.length).fill(0).map((_, i) => (
                                <div key={`empty-${i}`} className="px-4 py-2 rounded-lg border-2 border-dashed border-gray-200">
                                    <div className="text-gray-300">Empty</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 text-center">
                            <div className="text-[10px] text-emerald-600 uppercase">Hits</div>
                            <div className="text-xl font-bold text-emerald-700">{cacheHits}</div>
                        </div>
                        <div className="p-3 bg-rose-50 rounded-lg border border-rose-200 text-center">
                            <div className="text-[10px] text-rose-600 uppercase">Misses</div>
                            <div className="text-xl font-bold text-rose-700">{cacheMisses}</div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase">Hit Rate</div>
                            <div className="text-xl font-bold text-gray-700">
                                {cacheHits + cacheMisses > 0 ? ((cacheHits / (cacheHits + cacheMisses)) * 100).toFixed(0) : 0}%
                            </div>
                        </div>
                    </div>

                    {/* Log */}
                    {animationSteps.length > 0 && (
                        <div className="p-3 bg-gray-100 rounded-lg text-sm space-y-1 max-h-32 overflow-y-auto">
                            {animationSteps.map((step, i) => (
                                <div key={i} className={step.startsWith('✓') ? 'text-emerald-600' : step.startsWith('✗') ? 'text-rose-600' : 'text-gray-600'}>
                                    {step}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
