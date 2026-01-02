"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, RotateCcw, Shuffle } from 'lucide-react';

interface SortingVisualizerProps {
    algorithmId: string;
}

type ArrayBar = {
    value: number;
    state: 'default' | 'comparing' | 'swapping' | 'sorted';
};

export const SortingVisualizer = ({ algorithmId }: SortingVisualizerProps) => {
    const [array, setArray] = useState<ArrayBar[]>([]);
    const [arraySize, setArraySize] = useState(20);
    const [speed, setSpeed] = useState(50);
    const [isRunning, setIsRunning] = useState(false);
    const [isSorted, setIsSorted] = useState(false);
    const [comparisons, setComparisons] = useState(0);
    const [swaps, setSwaps] = useState(0);

    const generateArray = useCallback(() => {
        const newArray: ArrayBar[] = Array.from({ length: arraySize }, () => ({
            value: Math.floor(Math.random() * 100) + 5,
            state: 'default' as const
        }));
        setArray(newArray);
        setIsSorted(false);
        setComparisons(0);
        setSwaps(0);
    }, [arraySize]);

    useEffect(() => {
        generateArray();
    }, [generateArray]);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const getDelay = () => Math.max(10, 200 - speed * 2);

    // Bubble Sort
    const bubbleSort = async () => {
        const arr = [...array];
        const n = arr.length;
        let compCount = 0;
        let swapCount = 0;

        for (let i = 0; i < n - 1; i++) {
            for (let j = 0; j < n - i - 1; j++) {
                arr[j].state = 'comparing';
                arr[j + 1].state = 'comparing';
                setArray([...arr]);
                compCount++;
                setComparisons(compCount);
                await sleep(getDelay());

                if (arr[j].value > arr[j + 1].value) {
                    arr[j].state = 'swapping';
                    arr[j + 1].state = 'swapping';
                    setArray([...arr]);
                    await sleep(getDelay());

                    [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
                    swapCount++;
                    setSwaps(swapCount);
                }

                arr[j].state = 'default';
                arr[j + 1].state = 'default';
            }
            arr[n - 1 - i].state = 'sorted';
        }
        arr[0].state = 'sorted';
        setArray([...arr]);
    };

    // Quick Sort
    const quickSort = async () => {
        const arr = [...array];
        let compCount = 0;
        let swapCount = 0;

        const partition = async (low: number, high: number): Promise<number> => {
            const pivot = arr[high].value;
            arr[high].state = 'comparing';
            setArray([...arr]);

            let i = low - 1;

            for (let j = low; j < high; j++) {
                arr[j].state = 'comparing';
                setArray([...arr]);
                compCount++;
                setComparisons(compCount);
                await sleep(getDelay());

                if (arr[j].value < pivot) {
                    i++;
                    arr[i].state = 'swapping';
                    arr[j].state = 'swapping';
                    setArray([...arr]);
                    await sleep(getDelay());

                    [arr[i], arr[j]] = [arr[j], arr[i]];
                    swapCount++;
                    setSwaps(swapCount);
                }
                arr[j].state = 'default';
                if (i >= low) arr[i].state = 'default';
            }

            arr[i + 1].state = 'swapping';
            arr[high].state = 'swapping';
            setArray([...arr]);
            await sleep(getDelay());

            [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
            swapCount++;
            setSwaps(swapCount);

            arr[i + 1].state = 'sorted';
            arr[high].state = 'default';
            setArray([...arr]);

            return i + 1;
        };

        const sort = async (low: number, high: number) => {
            if (low < high) {
                const pi = await partition(low, high);
                await sort(low, pi - 1);
                await sort(pi + 1, high);
            } else if (low === high && low >= 0 && low < arr.length) {
                arr[low].state = 'sorted';
                setArray([...arr]);
            }
        };

        await sort(0, arr.length - 1);
        arr.forEach(item => item.state = 'sorted');
        setArray([...arr]);
    };

    // Merge Sort
    const mergeSort = async () => {
        const arr = [...array];
        let compCount = 0;
        let swapCount = 0;

        const merge = async (l: number, m: number, r: number) => {
            const n1 = m - l + 1;
            const n2 = r - m;
            const L = arr.slice(l, m + 1);
            const R = arr.slice(m + 1, r + 1);

            let i = 0, j = 0, k = l;

            while (i < n1 && j < n2) {
                arr[k].state = 'comparing';
                setArray([...arr]);
                compCount++;
                setComparisons(compCount);
                await sleep(getDelay());

                if (L[i].value <= R[j].value) {
                    arr[k] = { ...L[i], state: 'swapping' };
                    i++;
                } else {
                    arr[k] = { ...R[j], state: 'swapping' };
                    j++;
                }
                swapCount++;
                setSwaps(swapCount);
                setArray([...arr]);
                await sleep(getDelay());
                arr[k].state = 'default';
                k++;
            }

            while (i < n1) {
                arr[k] = { ...L[i], state: 'default' };
                i++;
                k++;
                setArray([...arr]);
                await sleep(getDelay() / 2);
            }

            while (j < n2) {
                arr[k] = { ...R[j], state: 'default' };
                j++;
                k++;
                setArray([...arr]);
                await sleep(getDelay() / 2);
            }
        };

        const sort = async (l: number, r: number) => {
            if (l < r) {
                const m = Math.floor((l + r) / 2);
                await sort(l, m);
                await sort(m + 1, r);
                await merge(l, m, r);
            }
        };

        await sort(0, arr.length - 1);
        arr.forEach(item => item.state = 'sorted');
        setArray([...arr]);
    };

    // Heap Sort
    const heapSort = async () => {
        const arr = [...array];
        let compCount = 0;
        let swapCount = 0;
        const n = arr.length;

        const heapify = async (size: number, i: number) => {
            let largest = i;
            const left = 2 * i + 1;
            const right = 2 * i + 2;

            arr[i].state = 'comparing';
            if (left < size) arr[left].state = 'comparing';
            if (right < size) arr[right].state = 'comparing';
            setArray([...arr]);
            compCount++;
            setComparisons(compCount);
            await sleep(getDelay());

            if (left < size && arr[left].value > arr[largest].value) {
                largest = left;
            }
            if (right < size && arr[right].value > arr[largest].value) {
                largest = right;
            }

            if (largest !== i) {
                arr[i].state = 'swapping';
                arr[largest].state = 'swapping';
                setArray([...arr]);
                await sleep(getDelay());

                [arr[i], arr[largest]] = [arr[largest], arr[i]];
                swapCount++;
                setSwaps(swapCount);

                arr[i].state = 'default';
                arr[largest].state = 'default';
                setArray([...arr]);

                await heapify(size, largest);
            } else {
                arr[i].state = 'default';
                if (left < size) arr[left].state = 'default';
                if (right < size) arr[right].state = 'default';
                setArray([...arr]);
            }
        };

        // Build heap
        for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
            await heapify(n, i);
        }

        // Extract elements
        for (let i = n - 1; i > 0; i--) {
            arr[0].state = 'swapping';
            arr[i].state = 'swapping';
            setArray([...arr]);
            await sleep(getDelay());

            [arr[0], arr[i]] = [arr[i], arr[0]];
            swapCount++;
            setSwaps(swapCount);

            arr[i].state = 'sorted';
            arr[0].state = 'default';
            setArray([...arr]);

            await heapify(i, 0);
        }

        arr[0].state = 'sorted';
        setArray([...arr]);
    };

    // Insertion Sort
    const insertionSort = async () => {
        const arr = [...array];
        let compCount = 0;
        let swapCount = 0;

        for (let i = 1; i < arr.length; i++) {
            const key = arr[i];
            let j = i - 1;

            arr[i].state = 'comparing';
            setArray([...arr]);
            await sleep(getDelay());

            while (j >= 0 && arr[j].value > key.value) {
                arr[j].state = 'comparing';
                arr[j + 1].state = 'swapping';
                setArray([...arr]);
                compCount++;
                setComparisons(compCount);
                await sleep(getDelay());

                arr[j + 1] = arr[j];
                swapCount++;
                setSwaps(swapCount);

                arr[j].state = 'default';
                j--;
            }

            arr[j + 1] = key;
            arr[j + 1].state = 'default';
            setArray([...arr]);
        }

        arr.forEach(item => item.state = 'sorted');
        setArray([...arr]);
    };

    const runSort = async () => {
        if (isRunning) return;
        setIsRunning(true);
        setIsSorted(false);

        // Reset states
        const resetArr = array.map(item => ({ ...item, state: 'default' as const }));
        setArray(resetArr);
        await sleep(100);

        switch (algorithmId) {
            case 'bubble-sort':
                await bubbleSort();
                break;
            case 'quick-sort':
                await quickSort();
                break;
            case 'merge-sort':
                await mergeSort();
                break;
            case 'heap-sort':
                await heapSort();
                break;
            case 'insertion-sort':
                await insertionSort();
                break;
            default:
                await bubbleSort();
        }

        setIsRunning(false);
        setIsSorted(true);
    };

    const getBarColor = (state: ArrayBar['state']) => {
        switch (state) {
            case 'comparing':
                return 'bg-yellow-400';
            case 'swapping':
                return 'bg-red-500';
            case 'sorted':
                return 'bg-green-500';
            default:
                return 'bg-gradient-to-t from-[#004040] to-[#006060]';
        }
    };

    const algorithmName = {
        'bubble-sort': 'Bubble Sort',
        'quick-sort': 'Quick Sort',
        'merge-sort': 'Merge Sort',
        'heap-sort': 'Heap Sort',
        'insertion-sort': 'Insertion Sort'
    }[algorithmId] || 'Sorting';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-900">
                    {algorithmName} Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={generateArray}
                        disabled={isRunning}
                    >
                        <Shuffle className="w-4 h-4 mr-1" />
                        New Array
                    </Button>
                    <Button
                        size="sm"
                        onClick={runSort}
                        disabled={isRunning || isSorted}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        {isRunning ? (
                            <><Pause className="w-4 h-4 mr-1" /> Sorting...</>
                        ) : (
                            <><Play className="w-4 h-4 mr-1" /> Start</>
                        )}
                    </Button>
                    {isSorted && (
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={generateArray}
                        >
                            <RotateCcw className="w-4 h-4 mr-1" />
                            Reset
                        </Button>
                    )}
                </div>
            </div>

            {/* Controls */}
            <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Array Size</span>
                        <span className="font-mono text-[#004040]">{arraySize}</span>
                    </div>
                    <Slider
                        value={[arraySize]}
                        onValueChange={(v) => { setArraySize(v[0]); generateArray(); }}
                        min={5}
                        max={50}
                        step={1}
                        disabled={isRunning}
                    />
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Speed</span>
                        <span className="font-mono text-[#004040]">{speed}%</span>
                    </div>
                    <Slider
                        value={[speed]}
                        onValueChange={(v) => setSpeed(v[0])}
                        min={1}
                        max={100}
                        step={1}
                    />
                </div>
            </div>

            {/* Visualization */}
            <div className="h-64 bg-gray-50 rounded-lg border border-gray-100 p-4 flex items-end justify-center gap-[2px]">
                {array.map((bar, idx) => (
                    <div
                        key={idx}
                        className={`${getBarColor(bar.state)} rounded-t transition-all duration-75`}
                        style={{
                            height: `${bar.value}%`,
                            width: `${Math.max(100 / array.length - 1, 4)}%`,
                            minWidth: '4px'
                        }}
                    />
                ))}
            </div>

            {/* Stats */}
            <div className="flex justify-center gap-8">
                <div className="text-center">
                    <div className="text-2xl font-bold text-[#004040]">{comparisons}</div>
                    <div className="text-xs text-gray-500 uppercase">Comparisons</div>
                </div>
                <div className="text-center">
                    <div className="text-2xl font-bold text-red-500">{swaps}</div>
                    <div className="text-xs text-gray-500 uppercase">Swaps</div>
                </div>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 text-xs">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-gradient-to-t from-[#004040] to-[#006060]" />
                    <span className="text-gray-600">Default</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-yellow-400" />
                    <span className="text-gray-600">Comparing</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-red-500" />
                    <span className="text-gray-600">Swapping</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-500" />
                    <span className="text-gray-600">Sorted</span>
                </div>
            </div>
        </div>
    );
};
