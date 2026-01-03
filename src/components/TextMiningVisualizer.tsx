"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Play, RotateCcw, Type, BarChart3, MessageSquare } from 'lucide-react';

interface TextMiningVisualizerProps {
    algorithmId: string;
}

interface WordWeight {
    word: string;
    tfidf: number;
    tf: number;
    idf: number;
    sentiment: 'positive' | 'negative' | 'neutral';
}

interface ClassificationResult {
    text: string;
    prediction: 'positive' | 'negative';
    confidence: number;
    keywords: string[];
}

// Sample training data
const sampleDocuments = [
    { text: "Great product highly recommend excellent quality", label: "positive" },
    { text: "Love it will buy again amazing experience", label: "positive" },
    { text: "Perfect exactly what I needed wonderful", label: "positive" },
    { text: "Exceeded my expectations fantastic service", label: "positive" },
    { text: "Terrible quality waste of money awful", label: "negative" },
    { text: "Broke after one week disappointed bad", label: "negative" },
    { text: "Do not buy this horrible experience", label: "negative" },
    { text: "Disappointing poor quality worst purchase", label: "negative" },
];

// Positive and negative word lists for simple classification
const positiveWords = ['great', 'love', 'excellent', 'amazing', 'perfect', 'wonderful', 'fantastic', 'recommend', 'best', 'good', 'happy', 'exceeded', 'beautiful', 'awesome'];
const negativeWords = ['terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointed', 'poor', 'waste', 'broke', 'disappointing', 'hate', 'never', 'refund', 'scam'];

export const TextMiningVisualizer = ({ algorithmId }: TextMiningVisualizerProps) => {
    const [wordWeights, setWordWeights] = useState<WordWeight[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [step, setStep] = useState<'idle' | 'tokenizing' | 'calculating-tf' | 'calculating-idf' | 'training' | 'complete'>('idle');
    const [inputText, setInputText] = useState('');
    const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
    const [activeTab, setActiveTab] = useState<'visualization' | 'classify'>('visualization');

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Calculate TF-IDF for the sample documents
    const calculateTFIDF = useCallback(async () => {
        setIsRunning(true);
        setWordWeights([]);
        setStep('tokenizing');
        await sleep(500);

        // Tokenize all documents
        const allWords: string[] = [];
        const documentWords: string[][] = [];

        sampleDocuments.forEach(doc => {
            const words = doc.text.toLowerCase().split(/\s+/).filter(w => w.length > 2);
            documentWords.push(words);
            allWords.push(...words);
        });

        setStep('calculating-tf');
        await sleep(600);

        // Calculate term frequency
        const wordCounts: Record<string, number> = {};
        allWords.forEach(word => {
            wordCounts[word] = (wordCounts[word] || 0) + 1;
        });

        setStep('calculating-idf');
        await sleep(600);

        // Calculate IDF
        const numDocs = documentWords.length;
        const docFrequency: Record<string, number> = {};
        Object.keys(wordCounts).forEach(word => {
            docFrequency[word] = documentWords.filter(doc => doc.includes(word)).length;
        });

        setStep('training');
        await sleep(700);

        // Calculate TF-IDF and create word weights
        const weights: WordWeight[] = [];
        Object.keys(wordCounts).forEach(word => {
            const tf = wordCounts[word] / allWords.length;
            const idf = Math.log(numDocs / (docFrequency[word] || 1)) + 1;
            const tfidf = tf * idf;

            let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
            if (positiveWords.includes(word)) sentiment = 'positive';
            else if (negativeWords.includes(word)) sentiment = 'negative';

            weights.push({ word, tfidf, tf, idf, sentiment });
        });

        // Sort by TF-IDF and take top words
        weights.sort((a, b) => b.tfidf - a.tfidf);

        // Animate adding words
        for (let i = 0; i < Math.min(weights.length, 15); i++) {
            await sleep(100);
            setWordWeights(prev => [...prev, weights[i]]);
        }

        setStep('complete');
        setIsRunning(false);
    }, []);

    // Simple Naive Bayes-like classification
    const classifyText = useCallback(() => {
        if (!inputText.trim()) return;

        const words = inputText.toLowerCase().split(/\s+/);
        let positiveScore = 0;
        let negativeScore = 0;
        const foundKeywords: string[] = [];

        words.forEach(word => {
            if (positiveWords.includes(word)) {
                positiveScore += 1;
                foundKeywords.push(word);
            }
            if (negativeWords.includes(word)) {
                negativeScore += 1;
                foundKeywords.push(word);
            }
        });

        const total = positiveScore + negativeScore || 1;
        const prediction: 'positive' | 'negative' = positiveScore >= negativeScore ? 'positive' : 'negative';
        const confidence = Math.max(positiveScore, negativeScore) / total;

        setClassificationResult({
            text: inputText,
            prediction,
            confidence: Math.min(0.95, 0.5 + confidence * 0.45),
            keywords: foundKeywords
        });
    }, [inputText]);

    const reset = () => {
        setWordWeights([]);
        setStep('idle');
        setClassificationResult(null);
        setInputText('');
    };

    const getBarColor = (sentiment: 'positive' | 'negative' | 'neutral') => {
        switch (sentiment) {
            case 'positive': return '#10B981';
            case 'negative': return '#EF4444';
            default: return '#6B7280';
        }
    };

    const getStepLabel = () => {
        switch (step) {
            case 'tokenizing': return 'Tokenizing documents...';
            case 'calculating-tf': return 'Calculating Term Frequency...';
            case 'calculating-idf': return 'Calculating IDF scores...';
            case 'training': return 'Training Naive Bayes classifier...';
            case 'complete': return 'Analysis Complete!';
            default: return 'Ready to analyze';
        }
    };

    const maxTfidf = Math.max(...wordWeights.map(w => w.tfidf), 0.01);

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    TF-IDF + Naive Bayes Visualization
                </h3>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={reset}
                        disabled={isRunning}
                    >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        Reset
                    </Button>
                    <Button
                        size="sm"
                        onClick={calculateTFIDF}
                        disabled={isRunning}
                        className="bg-[#004040] hover:bg-[#003030]"
                    >
                        <Play className="w-4 h-4 mr-1" />
                        {isRunning ? 'Processing...' : 'Run Analysis'}
                    </Button>
                </div>
            </div>

            {/* Tab Selector */}
            <div className="flex gap-2 border-b border-gray-200">
                <button
                    onClick={() => setActiveTab('visualization')}
                    className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'visualization'
                            ? 'border-[#004040] text-[#004040]'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                >
                    <BarChart3 className="w-4 h-4" />
                    TF-IDF Weights
                </button>
                <button
                    onClick={() => setActiveTab('classify')}
                    className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'classify'
                            ? 'border-[#004040] text-[#004040]'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                >
                    <MessageSquare className="w-4 h-4" />
                    Classify Text
                </button>
            </div>

            {activeTab === 'visualization' && (
                <>
                    {/* Status */}
                    <div className="flex items-center gap-2 text-sm">
                        <Type className="w-4 h-4 text-[#004040]" />
                        <span className={step === 'complete' ? 'text-emerald-600 font-medium' : 'text-gray-600'}>
                            {getStepLabel()}
                        </span>
                    </div>

                    {/* TF-IDF Bar Chart */}
                    <div className="relative h-80 bg-gray-50 rounded-lg border border-gray-100 p-4 overflow-hidden">
                        {wordWeights.length === 0 ? (
                            <div className="h-full flex items-center justify-center text-gray-400">
                                Click "Run Analysis" to visualize TF-IDF weights
                            </div>
                        ) : (
                            <div className="space-y-2 h-full overflow-y-auto">
                                {wordWeights.map((item, idx) => (
                                    <div key={idx} className="flex items-center gap-3 animate-fade-in">
                                        <div className="w-24 text-right">
                                            <span className="text-sm font-mono text-gray-700">{item.word}</span>
                                        </div>
                                        <div className="flex-1 h-6 bg-gray-200 rounded-full overflow-hidden">
                                            <div
                                                className="h-full rounded-full transition-all duration-500"
                                                style={{
                                                    width: `${(item.tfidf / maxTfidf) * 100}%`,
                                                    backgroundColor: getBarColor(item.sentiment)
                                                }}
                                            />
                                        </div>
                                        <div className="w-16 text-right">
                                            <span className="text-xs font-mono text-gray-500">
                                                {item.tfidf.toFixed(4)}
                                            </span>
                                        </div>
                                        <div className={`w-16 text-xs font-medium ${item.sentiment === 'positive' ? 'text-emerald-600' :
                                                item.sentiment === 'negative' ? 'text-rose-600' : 'text-gray-400'
                                            }`}>
                                            {item.sentiment}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Legend */}
                    <div className="flex justify-center gap-6 text-xs">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-emerald-500" />
                            <span className="text-gray-600">Positive Words</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-rose-500" />
                            <span className="text-gray-600">Negative Words</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-gray-400" />
                            <span className="text-gray-600">Neutral</span>
                        </div>
                    </div>
                </>
            )}

            {activeTab === 'classify' && (
                <div className="space-y-4">
                    {/* Input Area */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-700">
                            Enter text to classify:
                        </label>
                        <textarea
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            placeholder="Type a product review or any text to classify as positive or negative sentiment..."
                            className="w-full h-24 p-3 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#004040] focus:border-transparent"
                        />
                        <Button
                            onClick={classifyText}
                            disabled={!inputText.trim()}
                            className="w-full bg-[#004040] hover:bg-[#003030]"
                        >
                            <MessageSquare className="w-4 h-4 mr-2" />
                            Classify Sentiment
                        </Button>
                    </div>

                    {/* Classification Result */}
                    {classificationResult && (
                        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 space-y-4 animate-fade-in">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Prediction:</span>
                                <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${classificationResult.prediction === 'positive'
                                        ? 'bg-emerald-100 text-emerald-700'
                                        : 'bg-rose-100 text-rose-700'
                                    }`}>
                                    {classificationResult.prediction === 'positive' ? '😊 Positive' : '😞 Negative'}
                                </span>
                            </div>

                            <div>
                                <span className="text-sm font-medium text-gray-700">Confidence:</span>
                                <div className="mt-1 h-3 bg-gray-200 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-500 ${classificationResult.prediction === 'positive'
                                                ? 'bg-emerald-500'
                                                : 'bg-rose-500'
                                            }`}
                                        style={{ width: `${classificationResult.confidence * 100}%` }}
                                    />
                                </div>
                                <span className="text-xs text-gray-500 mt-1">
                                    {(classificationResult.confidence * 100).toFixed(1)}%
                                </span>
                            </div>

                            {classificationResult.keywords.length > 0 && (
                                <div>
                                    <span className="text-sm font-medium text-gray-700">Key Words Found:</span>
                                    <div className="flex flex-wrap gap-2 mt-2">
                                        {classificationResult.keywords.map((kw, i) => (
                                            <span
                                                key={i}
                                                className={`px-2 py-1 rounded text-xs font-medium ${positiveWords.includes(kw)
                                                        ? 'bg-emerald-100 text-emerald-700'
                                                        : 'bg-rose-100 text-rose-700'
                                                    }`}
                                            >
                                                {kw}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Sample texts */}
                    <div className="pt-2">
                        <span className="text-xs font-medium text-gray-500 uppercase">Try these examples:</span>
                        <div className="flex flex-wrap gap-2 mt-2">
                            {[
                                "Great product love it amazing quality",
                                "Terrible waste of money disappointed",
                                "This is okay nothing special"
                            ].map((text, i) => (
                                <button
                                    key={i}
                                    onClick={() => setInputText(text)}
                                    className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-600 transition-colors"
                                >
                                    "{text.substring(0, 25)}..."
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Stats (shown after complete) */}
            {step === 'complete' && activeTab === 'visualization' && (
                <div className="flex justify-center gap-8 pt-2">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-[#004040]">{wordWeights.length}</div>
                        <div className="text-xs text-gray-500 uppercase">Unique Words</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-emerald-600">
                            {wordWeights.filter(w => w.sentiment === 'positive').length}
                        </div>
                        <div className="text-xs text-gray-500 uppercase">Positive</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-rose-600">
                            {wordWeights.filter(w => w.sentiment === 'negative').length}
                        </div>
                        <div className="text-xs text-gray-500 uppercase">Negative</div>
                    </div>
                </div>
            )}
        </div>
    );
};
