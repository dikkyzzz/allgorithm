"use client";

import React, { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Play, Download, CheckCircle2, AlertCircle, Loader2, Upload, FileDown } from "lucide-react";

interface AlgorithmDemoProps {
    algorithmId: string;
}

interface DemoResult {
    // K-Means
    inertia?: number;
    silhouetteScore?: number;
    clusters?: number[];
    // Naive Bayes / Classification
    accuracy?: number;
    precision?: number | number[];
    recall?: number | number[];
    f1Score?: number;
    // PCA
    totalVarianceExplained?: number;
    components?: number;
    // Linear Regression
    r2Score?: number;
    rmse?: number;
    mae?: number;
    // DBSCAN
    nClusters?: number;
    nNoise?: number;
    // SVM
    supportVectors?: number;
}

interface APIResponse {
    results: DemoResult;
}

// Sample CSV content matching Real-World Use Cases
const sampleCSV: Record<string, { content: string; filename: string; description: string }> = {
    'k-means': {
        filename: 'customer_segmentation.csv',
        description: 'Customer Segmentation - purchasing behavior',
        content: `customer_id,annual_income,spending_score,age,purchase_frequency
C001,75000,82,32,45
C002,42000,35,22,12
C003,120000,78,45,67
C004,38000,42,28,18
C005,95000,91,35,52
C006,55000,28,42,8
C007,150000,95,38,78
C008,32000,38,25,15
C009,88000,72,48,42
C010,45000,45,30,22
C011,135000,88,41,71
C012,28000,22,23,6`
    },
    'naive-bayes': {
        filename: 'spam_detection.csv',
        description: 'Spam Detection - email classification',
        content: `email_id,word_count,has_links,has_urgent,caps_ratio,is_spam
E001,120,1,1,0.35,1
E002,45,0,0,0.02,0
E003,89,1,1,0.42,1
E004,156,0,0,0.05,0
E005,34,1,1,0.55,1
E006,210,0,0,0.03,0
E007,67,1,0,0.28,1
E008,178,1,0,0.08,0
E009,23,1,1,0.61,1
E010,95,0,0,0.04,0`
    },
    'pca': {
        filename: 'sensor_visualization.csv',
        description: 'Data Visualization - sensor readings',
        content: `sensor_id,temp,humidity,pressure,light,co2,noise
S001,23.5,45.2,1013.2,850,420,35.2
S002,24.1,48.7,1012.8,720,480,42.1
S003,22.8,42.1,1014.1,920,390,28.5
S004,25.2,52.3,1011.5,580,550,48.7
S005,23.9,46.8,1012.5,780,445,38.9
S006,21.5,38.5,1015.2,980,320,22.3
S007,26.1,55.2,1010.2,450,620,55.2
S008,24.5,49.5,1012.0,680,490,41.5`
    },
    'logistic-regression': {
        filename: 'churn_prediction.csv',
        description: 'Churn Prediction - customer retention',
        content: `customer_id,tenure_months,monthly_charges,total_charges,contract_type,churned
C001,12,65.5,786,0,0
C002,3,89.2,267.6,0,1
C003,48,45.0,2160,1,0
C004,6,78.5,471,0,1
C005,24,55.0,1320,1,0
C006,2,92.0,184,0,1
C007,36,42.5,1530,1,0
C008,1,105.0,105,0,1
C009,60,38.0,2280,2,0
C010,8,88.5,708,0,1`
    },
    'linear-regression': {
        filename: 'house_prices.csv',
        description: 'House Price Prediction - real estate',
        content: `house_id,sqft,bedrooms,bathrooms,age_years,price
H001,1500,3,2,10,285000
H002,2200,4,3,5,425000
H003,1200,2,1,25,195000
H004,1800,3,2,15,325000
H005,2500,4,3,3,510000
H006,1100,2,1,30,175000
H007,1950,3,2,8,365000
H008,2800,5,4,2,595000
H009,1650,3,2,12,305000
H010,2100,4,2,7,395000`
    },
    'dbscan': {
        filename: 'anomaly_detection.csv',
        description: 'Anomaly Detection - transaction data',
        content: `transaction_id,amount,time_hour,location_distance,merchant_category,is_anomaly
T001,45.50,14,2.5,1,0
T002,892.00,3,150.2,5,1
T003,23.75,10,1.2,2,0
T004,15000.00,2,500.0,8,1
T005,67.80,16,3.8,1,0
T006,34.20,11,0.5,3,0
T007,5500.00,4,280.5,7,1
T008,89.90,15,4.2,2,0
T009,12.50,9,1.0,1,0
T010,3200.00,1,420.0,6,1`
    },
    'svm': {
        filename: 'image_classification.csv',
        description: 'Image Classification - feature vectors',
        content: `image_id,feature1,feature2,feature3,feature4,feature5,category
IMG001,0.82,0.15,0.45,0.92,0.33,0
IMG002,0.23,0.88,0.67,0.12,0.78,1
IMG003,0.91,0.22,0.38,0.85,0.28,0
IMG004,0.18,0.92,0.71,0.08,0.82,1
IMG005,0.78,0.19,0.52,0.88,0.35,0
IMG006,0.25,0.85,0.62,0.15,0.75,1
IMG007,0.88,0.25,0.41,0.79,0.38,0
IMG008,0.21,0.89,0.68,0.11,0.81,1`
    },
    'tf-idf-nb': {
        filename: 'sentiment_analysis.csv',
        description: 'Sentiment Analysis - text reviews',
        content: `review_id,text,sentiment
R001,"Great product highly recommend",positive
R002,"Terrible quality waste of money",negative
R003,"Love it will buy again",positive
R004,"Broke after one week",negative
R005,"Exceeded my expectations",positive
R006,"Do not buy this",negative
R007,"Perfect exactly what I needed",positive
R008,"Disappointing poor quality",negative`
    },
    // NEW ALGORITHMS
    'decision-tree': {
        filename: 'decision_tree_data.csv',
        description: 'Decision Tree - classification data',
        content: `id,feature1,feature2,feature3,label\n1,0.5,0.3,0.8,0\n2,0.9,0.7,0.2,1\n3,0.3,0.5,0.9,0\n4,0.8,0.6,0.1,1`
    },
    'random-forest': {
        filename: 'random_forest_data.csv',
        description: 'Random Forest - ensemble classification',
        content: `id,feature1,feature2,feature3,label\n1,0.5,0.3,0.8,0\n2,0.9,0.7,0.2,1\n3,0.3,0.5,0.9,0\n4,0.8,0.6,0.1,1`
    },
    'knn': {
        filename: 'knn_data.csv',
        description: 'KNN - neighbor-based classification',
        content: `id,x,y,class\n1,1.0,1.0,A\n2,1.2,0.8,A\n3,5.0,5.0,B\n4,5.2,4.8,B`
    },
    'quick-sort': {
        filename: 'sort_data.csv',
        description: 'Sorting - array of numbers',
        content: `value\n64\n34\n25\n12\n22\n11\n90`
    },
    'merge-sort': {
        filename: 'sort_data.csv',
        description: 'Sorting - array of numbers',
        content: `value\n38\n27\n43\n3\n9\n82\n10`
    },
    'bubble-sort': {
        filename: 'sort_data.csv',
        description: 'Sorting - array of numbers',
        content: `value\n5\n1\n4\n2\n8`
    },
    'binary-search': {
        filename: 'search_data.csv',
        description: 'Searching - sorted array',
        content: `value\n2\n5\n8\n12\n16\n23\n38\n56\n72\n91`
    },
    'dijkstra': {
        filename: 'graph_data.csv',
        description: 'Graph - weighted edges',
        content: `from,to,weight\nA,B,1\nA,C,4\nB,C,2\nB,D,5\nC,D,1`
    },
    'neural-network': {
        filename: 'nn_data.csv',
        description: 'Neural Network - training data',
        content: `input1,input2,output\n0,0,0\n0,1,1\n1,0,1\n1,1,0`
    },
    'sha-256': {
        filename: 'hash_data.csv',
        description: 'Hashing - text to hash',
        content: `text\nHello World\nTest 123\nAlgorithm`
    },
    'collaborative-filtering': {
        filename: 'ratings_data.csv',
        description: 'Recommendation - user ratings',
        content: `user,item,rating\nU1,I1,5\nU1,I2,3\nU2,I1,4\nU2,I3,5`
    }
};

// Algorithm display names
const algorithmNames: Record<string, string> = {
    'k-means': 'K-Means',
    'naive-bayes': 'Naive Bayes',
    'pca': 'PCA',
    'logistic-regression': 'Logistic Regression',
    'linear-regression': 'Linear Regression',
    'dbscan': 'DBSCAN',
    'svm': 'SVM',
    'tf-idf-nb': 'TF-IDF + NB',
    'decision-tree': 'Decision Tree',
    'random-forest': 'Random Forest',
    'knn': 'KNN',
    'hierarchical-clustering': 'Hierarchical Clustering',
    'quick-sort': 'Quick Sort',
    'merge-sort': 'Merge Sort',
    'bubble-sort': 'Bubble Sort',
    'heap-sort': 'Heap Sort',
    'insertion-sort': 'Insertion Sort',
    'binary-search': 'Binary Search',
    'linear-search': 'Linear Search',
    'dijkstra': 'Dijkstra',
    'bfs-dfs': 'BFS/DFS',
    'a-star': 'A*',
    'bellman-ford': 'Bellman-Ford',
    'floyd-warshall': 'Floyd-Warshall',
    'pagerank': 'PageRank',
    'neural-network': 'Neural Network',
    'cnn': 'CNN',
    'rnn-lstm': 'RNN/LSTM',
    'transformer': 'Transformer',
    'descriptive-stats': 'Mean/Median/Mode',
    'standard-deviation': 'Standard Deviation',
    'correlation': 'Correlation',
    'bayesian-inference': 'Bayesian Inference',
    'markov-chain': 'Markov Chain',
    'aes': 'AES',
    'rsa': 'RSA',
    'sha-256': 'SHA-256',
    'hmac': 'HMAC',
    'diffie-hellman': 'Diffie-Hellman',
    'collaborative-filtering': 'Collaborative Filtering',
    'content-based-filtering': 'Content-Based Filtering',
    'lru-cache': 'LRU Cache'
};

export const AlgorithmDemo = ({ algorithmId }: AlgorithmDemoProps) => {
    const [selectedDataset, setSelectedDataset] = useState("default");
    const [isRunning, setIsRunning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<APIResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [customDataset, setCustomDataset] = useState<{ name: string; id: string } | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Algorithm-specific parameters
    const [kValue, setKValue] = useState(3);
    const [trainRatio, setTrainRatio] = useState(80);
    const [nComponents, setNComponents] = useState(2);
    const [epsilon, setEpsilon] = useState(0.5);
    const [minSamples, setMinSamples] = useState(5);
    const [cValue, setCValue] = useState(1.0);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.success) {
                setCustomDataset({ name: data.data.name, id: data.data.id });
                setSelectedDataset(data.data.id);
            } else {
                setError('Upload failed: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Upload failed:', err);
            setError('Failed to upload file');
        }
    };

    const downloadSampleCSV = () => {
        const sample = sampleCSV[algorithmId];
        if (!sample) return;

        const blob = new Blob([sample.content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = sample.filename;
        link.click();
        URL.revokeObjectURL(url);
    };

    // Simulated results for algorithms without backend
    const getSimulatedResult = (): DemoResult => {
        switch (algorithmId) {
            case 'logistic-regression':
            case 'decision-tree':
            case 'random-forest':
            case 'knn':
                return {
                    accuracy: 87.5 + Math.random() * 10,
                    precision: 0.85 + Math.random() * 0.1,
                    recall: 0.82 + Math.random() * 0.1,
                    f1Score: 0.84 + Math.random() * 0.08
                };
            case 'linear-regression':
                return {
                    r2Score: 0.89 + Math.random() * 0.08,
                    rmse: 18500 + Math.random() * 5000,
                    mae: 12500 + Math.random() * 3000
                };
            case 'dbscan':
            case 'hierarchical-clustering':
                return {
                    nClusters: Math.floor(3 + Math.random() * 4),
                    nNoise: Math.floor(2 + Math.random() * 5),
                    silhouetteScore: 0.65 + Math.random() * 0.2
                };
            case 'svm':
                return {
                    accuracy: 92.5 + Math.random() * 5,
                    supportVectors: Math.floor(15 + Math.random() * 20),
                    precision: 0.91 + Math.random() * 0.06
                };
            case 'tf-idf-nb':
                return {
                    accuracy: 88.0 + Math.random() * 8,
                    precision: 0.87 + Math.random() * 0.08,
                    recall: 0.85 + Math.random() * 0.1
                };
            // Sorting algorithms
            case 'quick-sort':
            case 'merge-sort':
            case 'bubble-sort':
            case 'heap-sort':
            case 'insertion-sort':
                return { accuracy: 100 } as DemoResult;
            // Searching algorithms  
            case 'binary-search':
            case 'linear-search':
                return { accuracy: 100 } as DemoResult;
            // Graph algorithms
            case 'dijkstra':
            case 'bfs-dfs':
            case 'a-star':
            case 'bellman-ford':
            case 'floyd-warshall':
            case 'pagerank':
                return { accuracy: 100 } as DemoResult;
            // Deep Learning
            case 'neural-network':
            case 'cnn':
            case 'rnn-lstm':
            case 'transformer':
                return {
                    accuracy: 92 + Math.random() * 6,
                    precision: 0.90 + Math.random() * 0.08
                };
            // Statistics
            case 'descriptive-stats':
            case 'standard-deviation':
            case 'correlation':
            case 'bayesian-inference':
            case 'markov-chain':
                return { accuracy: 100 } as DemoResult;
            // Security
            case 'aes':
            case 'rsa':
            case 'sha-256':
            case 'hmac':
            case 'diffie-hellman':
                return { accuracy: 100 } as DemoResult;
            // Recommendation
            case 'collaborative-filtering':
            case 'content-based-filtering':
            case 'lru-cache':
                return { accuracy: 85 + Math.random() * 10 };
            default:
                return { accuracy: 85 };
        }
    };

    const runAlgorithm = async () => {
        setIsRunning(true);
        setResult(null);
        setError(null);
        setProgress(20);

        try {
            // Algorithms with real backend
            if (['k-means', 'naive-bayes', 'pca'].includes(algorithmId)) {
                let endpoint = '';
                let body = {};

                switch (algorithmId) {
                    case 'k-means':
                        endpoint = '/api/run/kmeans';
                        body = { datasetId: selectedDataset === 'default' ? 'iris' : selectedDataset, k: kValue, maxIterations: 100, initialization: 'kmeans++' };
                        break;
                    case 'naive-bayes':
                        endpoint = '/api/run/naive-bayes';
                        body = { datasetId: selectedDataset === 'default' ? 'titanic' : selectedDataset, trainRatio: trainRatio / 100 };
                        break;
                    case 'pca':
                        endpoint = '/api/run/pca';
                        body = { datasetId: selectedDataset === 'default' ? 'iris' : selectedDataset, nComponents };
                        break;
                }

                setProgress(40);
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });

                setProgress(80);
                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Algorithm failed');
                }

                setProgress(100);
                setResult(data.data);
            } else {
                // Simulate for other algorithms
                setProgress(40);
                await new Promise(resolve => setTimeout(resolve, 800));
                setProgress(70);
                await new Promise(resolve => setTimeout(resolve, 600));
                setProgress(100);
                setResult({ results: getSimulatedResult() });
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsRunning(false);
        }
    };

    const exportResults = () => {
        if (!result) return;
        const dataStr = JSON.stringify(result, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `allgorithm-${algorithmId}-results.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const sample = sampleCSV[algorithmId];
    const algorithmName = algorithmNames[algorithmId] || algorithmId;

    // Render algorithm-specific parameters
    const renderParameters = () => {
        switch (algorithmId) {
            case 'k-means':
                return (
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Clusters (K)</label>
                            <span className="text-xs text-[#004040] font-mono font-bold">{kValue}</span>
                        </div>
                        <Slider value={[kValue]} onValueChange={(v) => setKValue(v[0])} min={2} max={10} step={1} />
                    </div>
                );
            case 'naive-bayes':
            case 'logistic-regression':
            case 'svm':
            case 'tf-idf-nb':
                return (
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Train Ratio</label>
                            <span className="text-xs text-violet-600 font-mono font-bold">{trainRatio}%</span>
                        </div>
                        <Slider value={[trainRatio]} onValueChange={(v) => setTrainRatio(v[0])} min={50} max={90} step={5} />
                    </div>
                );
            case 'pca':
                return (
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Components</label>
                            <span className="text-xs text-emerald-600 font-mono font-bold">{nComponents}</span>
                        </div>
                        <Slider value={[nComponents]} onValueChange={(v) => setNComponents(v[0])} min={1} max={4} step={1} />
                    </div>
                );
            case 'linear-regression':
                return (
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Train/Test Split</label>
                            <span className="text-xs text-blue-600 font-mono font-bold">{trainRatio}%</span>
                        </div>
                        <Slider value={[trainRatio]} onValueChange={(v) => setTrainRatio(v[0])} min={60} max={90} step={5} />
                    </div>
                );
            case 'dbscan':
                return (
                    <>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Epsilon (ε)</label>
                                <span className="text-xs text-orange-600 font-mono font-bold">{epsilon.toFixed(1)}</span>
                            </div>
                            <Slider value={[epsilon * 10]} onValueChange={(v) => setEpsilon(v[0] / 10)} min={1} max={20} step={1} />
                        </div>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Min Samples</label>
                                <span className="text-xs text-orange-600 font-mono font-bold">{minSamples}</span>
                            </div>
                            <Slider value={[minSamples]} onValueChange={(v) => setMinSamples(v[0])} min={2} max={15} step={1} />
                        </div>
                    </>
                );
            default:
                return null;
        }
    };

    // Render algorithm-specific results
    const renderResults = () => {
        if (!result?.results) return null;
        const r = result.results;

        switch (algorithmId) {
            case 'k-means':
                return (
                    <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Inertia</div>
                            <div className="text-xl font-bold text-gray-900">{r.inertia?.toFixed(2)}</div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Silhouette</div>
                            <div className="text-xl font-bold text-gray-900">{r.silhouetteScore?.toFixed(3)}</div>
                        </div>
                    </div>
                );
            case 'naive-bayes':
            case 'logistic-regression':
            case 'svm':
            case 'tf-idf-nb':
                return (
                    <div className="space-y-3">
                        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Accuracy</div>
                            <div className="text-3xl font-bold text-[#004040]">{r.accuracy?.toFixed(1)}%</div>
                        </div>
                        {r.precision !== undefined && (
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                    <div className="text-[10px] text-gray-500 uppercase mb-1">Precision</div>
                                    <div className="text-lg font-bold text-gray-900">
                                        {Array.isArray(r.precision)
                                            ? (r.precision.reduce((a, b) => a + b, 0) / r.precision.length).toFixed(1) + '%'
                                            : r.precision.toFixed(2)}
                                    </div>
                                </div>
                                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                    <div className="text-[10px] text-gray-500 uppercase mb-1">Recall</div>
                                    <div className="text-lg font-bold text-gray-900">
                                        {Array.isArray(r.recall)
                                            ? (r.recall.reduce((a, b) => a + b, 0) / r.recall.length).toFixed(1) + '%'
                                            : typeof r.recall === 'number' ? r.recall.toFixed(2) : '-'}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                );
            case 'pca':
                return (
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 text-center">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Variance Explained</div>
                        <div className="text-3xl font-bold text-[#004040]">{r.totalVarianceExplained?.toFixed(1)}%</div>
                    </div>
                );
            case 'linear-regression':
                return (
                    <div className="space-y-3">
                        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">R² Score</div>
                            <div className="text-3xl font-bold text-blue-600">{r.r2Score?.toFixed(3)}</div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                <div className="text-[10px] text-gray-500 uppercase mb-1">RMSE</div>
                                <div className="text-lg font-bold text-gray-900">${r.rmse?.toFixed(0)}</div>
                            </div>
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                <div className="text-[10px] text-gray-500 uppercase mb-1">MAE</div>
                                <div className="text-lg font-bold text-gray-900">${r.mae?.toFixed(0)}</div>
                            </div>
                        </div>
                    </div>
                );
            case 'dbscan':
                return (
                    <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                <div className="text-[10px] text-gray-500 uppercase mb-1">Clusters Found</div>
                                <div className="text-2xl font-bold text-orange-600">{r.nClusters}</div>
                            </div>
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                                <div className="text-[10px] text-gray-500 uppercase mb-1">Noise Points</div>
                                <div className="text-2xl font-bold text-gray-600">{r.nNoise}</div>
                            </div>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-center">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Silhouette Score</div>
                            <div className="text-xl font-bold text-gray-900">{r.silhouetteScore?.toFixed(3)}</div>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <Card className="bg-white border-gray-200 shadow-sm">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg text-gray-900">
                    <Play className="w-5 h-5 text-[#004040]" />
                    Try It Live
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
                {/* Dataset Selection */}
                <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Dataset</label>
                    <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                        <SelectTrigger className="bg-white border-gray-300">
                            <SelectValue placeholder="Select dataset" />
                        </SelectTrigger>
                        <SelectContent className="bg-white border-gray-200">
                            <SelectItem value="default">{sample?.description.split(' - ')[0] || 'Sample Data'}</SelectItem>
                            <SelectItem value="iris">Iris Dataset (150 rows)</SelectItem>
                            <SelectItem value="titanic">Titanic Dataset (891 rows)</SelectItem>
                            {customDataset && (
                                <SelectItem value={customDataset.id}>{customDataset.name} (Custom)</SelectItem>
                            )}
                        </SelectContent>
                    </Select>
                </div>

                {/* Upload & Download Buttons */}
                <div className="flex gap-2">
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        ref={fileInputRef}
                        className="hidden"
                    />
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fileInputRef.current?.click()}
                        className="flex-1 border-dashed border-gray-300 text-gray-600 hover:border-[#004040] hover:text-[#004040]"
                    >
                        <Upload className="w-4 h-4 mr-1" />
                        Upload CSV
                    </Button>
                    {sample && (
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={downloadSampleCSV}
                            className="flex-1 border-gray-300 text-gray-600 hover:border-[#004040] hover:text-[#004040]"
                        >
                            <FileDown className="w-4 h-4 mr-1" />
                            Sample CSV
                        </Button>
                    )}
                </div>

                {/* Sample Description */}
                {sample && (
                    <p className="text-xs text-gray-500 italic">
                        Format: {sample.description}
                    </p>
                )}

                {/* Algorithm-specific Parameters */}
                {renderParameters()}

                {/* Run Button */}
                <Button
                    onClick={runAlgorithm}
                    disabled={isRunning}
                    className="w-full bg-[#004040] hover:bg-[#003333] text-white"
                >
                    {isRunning ? (
                        <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Processing...
                        </>
                    ) : (
                        <>
                            <Play className="w-4 h-4 mr-2 fill-current" />
                            Run {algorithmName}
                        </>
                    )}
                </Button>

                {/* Progress */}
                {isRunning && (
                    <Progress value={progress} className="h-2" />
                )}

                {/* Error */}
                {error && (
                    <div className="flex items-center gap-2 p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-700 text-sm">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                        {error}
                    </div>
                )}

                {/* Results */}
                {result && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-emerald-600 text-sm font-medium">
                            <CheckCircle2 className="w-4 h-4" />
                            Analysis Complete
                        </div>

                        {renderResults()}

                        {/* Export Button */}
                        <Button
                            onClick={exportResults}
                            variant="outline"
                            className="w-full border-gray-300 text-gray-700"
                        >
                            <Download className="w-4 h-4 mr-2" />
                            Export Results
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};
