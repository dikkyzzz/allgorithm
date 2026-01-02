"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Play, RotateCcw, Search } from 'lucide-react';

interface SearchVisualizerProps {
    algorithmId: string;
}

type ArrayElement = {
    value: number;
    state: 'default' | 'current' | 'checked' | 'found' | 'eliminated';
};

export const SearchVisualizer = ({ algorithmId }: SearchVisualizerProps) => {
    const [array, setArray] = useState<ArrayElement[]>([]);
    const [target, setTarget] = useState(42);
    const [isRunning, setIsRunning] = useState(false);
    const [found, setFound] = useState<number | null>(null);
    const [comparisons, setComparisons] = useState(0);
    const [message, setMessage] = useState('');

    const generateArray = useCallback(() => {
        const newArray: ArrayElement[] = Array.from({ length: 15 }, (_, i) => ({
            value: (i + 1) * 7,
            state: 'default' as const
        }));
        setArray(newArray);
        setFound(null);
        setComparisons(0);
        setMessage('');
    }, []);

    useEffect(() => {
        generateArray();
    }, [generateArray]);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Binary Search
    const binarySearch = async () => {
        const arr = [...array];
        let left = 0;
        let right = arr.length - 1;
        let compCount = 0;

        while (left <= right) {
            // Highlight search range
            arr.forEach((el, i) => {
                if (i < left || i > right) el.state = 'eliminated';
            });
            setArray([...arr]);
            await sleep(300);

            const mid = Math.floor((left + right) / 2);
            arr[mid].state = 'current';
            setArray([...arr]);
            setMessage(`Checking middle: ${arr[mid].value}`);
            compCount++;
            setComparisons(compCount);
            await sleep(500);

            if (arr[mid].value === target) {
                arr[mid].state = 'found';
                setArray([...arr]);
                setFound(mid);
                setMessage(`Found ${target} at index ${mid}!`);
                return;
            } else if (arr[mid].value < target) {
                arr[mid].state = 'checked';
                left = mid + 1;
                setMessage(`${arr[mid].value} < ${target}, search right half`);
            } else {
                arr[mid].state = 'checked';
                right = mid - 1;
                setMessage(`${arr[mid].value} > ${target}, search left half`);
            }
            setArray([...arr]);
            await sleep(300);
        }

        setMessage(`${target} not found in array`);
    };

    // Linear Search
    const linearSearch = async () => {
        const arr = [...array];
        let compCount = 0;

        for (let i = 0; i < arr.length; i++) {
            arr[i].state = 'current';
            setArray([...arr]);
            setMessage(`Checking index ${i}: ${arr[i].value}`);
            compCount++;
            setComparisons(compCount);
            await sleep(400);

            if (arr[i].value === target) {
                arr[i].state = 'found';
                setArray([...arr]);
                setFound(i);
                setMessage(`Found ${target} at index ${i}!`);
                return;
            }

            arr[i].state = 'checked';
            setArray([...arr]);
            await sleep(200);
        }

        setMessage(`${target} not found in array`);
    };

    const runSearch = async () => {
        if (isRunning) return;
        setIsRunning(true);
        setFound(null);
        setComparisons(0);

        // Reset states
        const resetArr = array.map(item => ({ ...item, state: 'default' as const }));
        setArray(resetArr);
        await sleep(100);

        if (algorithmId === 'binary-search') {
            await binarySearch();
        } else {
            await linearSearch();
        }

        setIsRunning(false);
    };

    const getElementColor = (state: ArrayElement['state']) => {
        switch (state) {
            case 'current':
                return 'bg-yellow-400 text-yellow-900 scale-110';
            case 'checked':
                return 'bg-gray-300 text-gray-600';
            case 'found':
                return 'bg-green-500 text-white scale-125';
            case 'eliminated':
                return 'bg-gray-100 text-gray-400 opacity-50';
            default:
                return 'bg-gradient-to-b from-[#004040] to-[#006060] text-white';
        }
    };

    const algorithmName = algorithmId === 'binary-search' ? 'Binary Search' : 'Linear Search';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    {algorithmName} Visualization
                </h3>
                <div className="flex gap-2 items-center">
                    <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600">Target:</span>
                        <Input
                            type="number"
                            value={target}
                            onChange={(e) => setTarget(parseInt(e.target.value) || 0)}
                            className="w-20 h-9"
                            disabled={isRunning}
                        />
                    </div>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generateArray}
                        disabled={isRunning}
                    >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        Reset
                    </Button>
                    <Button
                        size="sm"
                        onClick={runSearch}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Search className="w-4 h-4 mr-1" />
                        {isRunning ? 'Searching...' : 'Search'}
                    </Button>
                </div>
            </div>

            {/* Array Visualization */}
            <div className="flex justify-center gap-2 py-8">
                {array.map((el, idx) => (
                    <div
                        key={idx}
                        className={`
                            w-12 h-12 rounded-lg flex items-center justify-center font-bold text-sm
                            transition-all duration-300 ${getElementColor(el.state)}
                        `}
                    >
                        {el.value}
                    </div>
                ))}
            </div>

            {/* Index labels */}
            <div className="flex justify-center gap-2">
                {array.map((_, idx) => (
                    <div key={idx} className="w-12 text-center text-xs text-gray-400">
                        [{idx}]
                    </div>
                ))}
            </div>

            {/* Message */}
            {message && (
                <div className={`text-center p-3 rounded-lg ${found !== null ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-700'
                    }`}>
                    {message}
                </div>
            )}

            {/* Stats */}
            <div className="flex justify-center gap-8">
                <div className="text-center">
                    <div className="text-2xl font-bold text-[#004040]">{comparisons}</div>
                    <div className="text-xs text-gray-500 uppercase">Comparisons</div>
                </div>
                {found !== null && (
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-500">{found}</div>
                        <div className="text-xs text-gray-500 uppercase">Found at Index</div>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-xs flex-wrap">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-gradient-to-b from-[#004040] to-[#006060]" />
                    <span className="text-gray-600">Unchecked</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-yellow-400" />
                    <span className="text-gray-600">Current</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-gray-300" />
                    <span className="text-gray-600">Checked</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-500" />
                    <span className="text-gray-600">Found</span>
                </div>
            </div>
        </div>
    );
};
