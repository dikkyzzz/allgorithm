"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Play, Database, CheckCircle2, Info, AlertCircle, RefreshCw, Download, Upload } from "lucide-react";

interface KMeansResult {
    inertia: number;
    silhouetteScore: number;
    iterations: number;
    clusterSizes: Record<number, number>;
    converged: boolean;
}

interface NaiveBayesResult {
    accuracy: number;
    precision: number[];
    recall: number[];
    confusionMatrix: number[][];
    classNames: string[];
    trainSize: number;
    testSize: number;
}

interface PCAResult {
    explainedVariance: number[];
    cumulativeVariance: number[];
    totalVarianceExplained: number;
}

type AlgorithmResult = KMeansResult | NaiveBayesResult | PCAResult | null;

export const Demo = () => {
    const [activeTab, setActiveTab] = useState("kmeans");
    const [selectedDataset, setSelectedDataset] = useState("iris");
    const [kValue, setKValue] = useState(3);
    const [initialization, setInitialization] = useState("kmeans++");
    const [trainRatio, setTrainRatio] = useState(80);
    const [nComponents, setNComponents] = useState(2);

    const [isRunning, setIsRunning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<AlgorithmResult>(null);
    const [error, setError] = useState<string | null>(null);
    const [customDataset, setCustomDataset] = useState<{ name: string; id: string } | null>(null);
    const [apiResponse, setApiResponse] = useState<Record<string, unknown> | null>(null);

    const fileInputRef = React.useRef<HTMLInputElement>(null);

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
            }
        } catch (err) {
            console.error('Upload failed:', err);
        }
    };

    const exportResults = () => {
        if (!apiResponse) return;
        const dataStr = JSON.stringify(apiResponse, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `allgorithm-${activeTab}-results.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const runAlgorithm = async () => {
        setIsRunning(true);
        setResult(null);
        setError(null);
        setProgress(10);

        try {
            let endpoint = "";
            let body = {};

            switch (activeTab) {
                case "kmeans":
                    endpoint = "/api/run/kmeans";
                    body = {
                        datasetId: selectedDataset,
                        k: kValue,
                        maxIterations: 100,
                        initialization,
                    };
                    break;
                case "nb":
                    endpoint = "/api/run/naive-bayes";
                    body = {
                        datasetId: selectedDataset === "iris" ? "titanic" : selectedDataset,
                        trainRatio: trainRatio / 100,
                    };
                    break;
                case "pca":
                    endpoint = "/api/run/pca";
                    body = {
                        datasetId: selectedDataset,
                        nComponents,
                    };
                    break;
            }

            setProgress(30);

            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            setProgress(70);

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || "Algorithm failed");
            }

            setProgress(100);
            setResult(data.data.results);
            setApiResponse(data.data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setIsRunning(false);
        }
    };

    const resetAnalysis = () => {
        setResult(null);
        setError(null);
        setProgress(0);
        setApiResponse(null);
    };

    const renderKMeansResults = (r: KMeansResult) => (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
        >
            <div className="flex items-center gap-2 text-emerald-600 text-sm font-medium">
                <CheckCircle2 className="w-4 h-4" /> K-Means Completed
            </div>
            <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Inertia</div>
                    <div className="text-xl font-bold text-gray-900">{r.inertia}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Silhouette</div>
                    <div className="text-xl font-bold text-gray-900">{r.silhouetteScore}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Iterations</div>
                    <div className="text-xl font-bold text-gray-900">{r.iterations}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Converged</div>
                    <div className="text-xl font-bold text-gray-900">{r.converged ? "Yes" : "No"}</div>
                </div>
            </div>
            <div className="p-3 bg-[#004040]/5 border border-[#004040]/20 rounded-lg">
                <div className="flex gap-2 items-start text-xs text-[#004040] leading-relaxed">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>
                        Silhouette score of {r.silhouetteScore} indicates {r.silhouetteScore > 0.5 ? "good" : r.silhouetteScore > 0.25 ? "fair" : "weak"} cluster separation.
                    </span>
                </div>
            </div>
        </motion.div>
    );

    const renderNaiveBayesResults = (r: NaiveBayesResult) => (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
        >
            <div className="flex items-center gap-2 text-emerald-600 text-sm font-medium">
                <CheckCircle2 className="w-4 h-4" /> Classification Completed
            </div>
            <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 col-span-2">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Accuracy</div>
                    <div className="text-2xl font-bold text-gray-900">{r.accuracy}%</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Train Size</div>
                    <div className="text-lg font-bold text-gray-900">{r.trainSize}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Test Size</div>
                    <div className="text-lg font-bold text-gray-900">{r.testSize}</div>
                </div>
            </div>
            <div className="p-3 bg-violet-50 border border-violet-200 rounded-lg">
                <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-2">Confusion Matrix</div>
                <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${r.confusionMatrix.length}, 1fr)` }}>
                    {r.confusionMatrix.flat().map((val, i) => (
                        <div key={i} className={`text-center p-1 rounded text-xs font-mono ${i % (r.confusionMatrix.length + 1) === 0 ? "bg-emerald-100 text-emerald-700" : "bg-gray-100 text-gray-600"}`}>
                            {val}
                        </div>
                    ))}
                </div>
            </div>
        </motion.div>
    );

    const renderPCAResults = (r: PCAResult) => (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
        >
            <div className="flex items-center gap-2 text-emerald-600 text-sm font-medium">
                <CheckCircle2 className="w-4 h-4" /> PCA Completed
            </div>
            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Total Variance Explained</div>
                <div className="text-2xl font-bold text-gray-900">{r.totalVarianceExplained}%</div>
            </div>
            <div className="space-y-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-tighter">Variance per Component</div>
                {r.explainedVariance.map((v, i) => (
                    <div key={i} className="flex items-center gap-3">
                        <span className="text-xs text-gray-500 w-12">PC{i + 1}</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div className="h-full bg-[#004040] rounded-full" style={{ width: `${v}%` }} />
                        </div>
                        <span className="text-xs text-gray-900 font-mono w-12 text-right">{v}%</span>
                    </div>
                ))}
            </div>
        </motion.div>
    );

    return (
        <section id="demo" className="py-24 md:py-32 bg-gray-50">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-6">Interactive Playground</h2>
                    <p className="text-gray-600 text-lg max-w-2xl mx-auto">
                        Experience the workflow of state-of-the-art algorithms without writing a single line of code.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                    {/* Steps */}
                    <div className="lg:col-span-4 space-y-6">
                        {[
                            { step: 1, title: "Select Dataset", desc: "Choose from our samples or upload your own CSV/JSON." },
                            { step: 2, title: "Tune Parameters", desc: "Configure hyperparameters using intuitive sliders and inputs." },
                            { step: 3, title: "Run Analysis", desc: "Execute the algorithm and watch the results converge in real-time." },
                            { step: 4, title: "Interpret Results", desc: "Get detailed metrics, charts, and plain-English explanations." },
                        ].map((s, idx) => (
                            <div key={idx} className="flex gap-4 p-4 rounded-xl hover:bg-white hover:shadow-md transition-all border border-transparent hover:border-gray-200">
                                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#004040] flex items-center justify-center text-sm font-bold text-white">
                                    {s.step}
                                </div>
                                <div>
                                    <h3 className="text-gray-900 font-semibold mb-1">{s.title}</h3>
                                    <p className="text-sm text-gray-500">{s.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Interactive Panel */}
                    <div className="lg:col-span-8">
                        <Card className="bg-white border-gray-200 shadow-lg overflow-hidden">
                            <CardHeader className="border-b border-gray-200 bg-gray-50">
                                <div className="flex items-center justify-between">
                                    <CardTitle className="text-lg text-gray-900 font-medium flex items-center gap-2">
                                        Analysis Workbench
                                    </CardTitle>
                                    <Tabs value={activeTab} onValueChange={(v) => { setActiveTab(v); resetAnalysis(); }}>
                                        <TabsList className="bg-white border border-gray-200">
                                            <TabsTrigger value="kmeans" className="text-xs data-[state=active]:bg-[#004040] data-[state=active]:text-white">K-Means</TabsTrigger>
                                            <TabsTrigger value="nb" className="text-xs data-[state=active]:bg-[#004040] data-[state=active]:text-white">Naive Bayes</TabsTrigger>
                                            <TabsTrigger value="pca" className="text-xs data-[state=active]:bg-[#004040] data-[state=active]:text-white">PCA</TabsTrigger>
                                        </TabsList>
                                    </Tabs>
                                </div>
                            </CardHeader>
                            <CardContent className="p-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <div className="space-y-6">
                                        <div className="space-y-3">
                                            <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Dataset</label>
                                            <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                                                <SelectTrigger className="bg-white border-gray-300 text-gray-900">
                                                    <SelectValue placeholder="Select dataset" />
                                                </SelectTrigger>
                                                <SelectContent className="bg-white border-gray-200 text-gray-900">
                                                    <SelectItem value="iris">Iris Flowers (50 rows)</SelectItem>
                                                    <SelectItem value="customers">Customer Segments (40 rows)</SelectItem>
                                                    <SelectItem value="titanic">Titanic Survival (40 rows)</SelectItem>
                                                    {customDataset && (
                                                        <SelectItem value={customDataset.id}>{customDataset.name} (Custom)</SelectItem>
                                                    )}
                                                </SelectContent>
                                            </Select>
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
                                                className="w-full border-dashed border-gray-300 text-gray-600 hover:border-[#004040] hover:text-[#004040]"
                                            >
                                                <Upload className="w-4 h-4 mr-2" />
                                                Upload CSV
                                            </Button>
                                        </div>

                                        {activeTab === "kmeans" && (
                                            <>
                                                <div className="space-y-3">
                                                    <div className="flex justify-between">
                                                        <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Clusters (K)</label>
                                                        <span className="text-xs text-[#004040] font-mono font-bold">{kValue}</span>
                                                    </div>
                                                    <Slider value={[kValue]} onValueChange={(v) => setKValue(v[0])} min={2} max={10} step={1} className="py-4" />
                                                </div>
                                                <div className="space-y-3">
                                                    <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Initialization</label>
                                                    <Select value={initialization} onValueChange={setInitialization}>
                                                        <SelectTrigger className="bg-white border-gray-300 text-gray-900">
                                                            <SelectValue placeholder="Select style" />
                                                        </SelectTrigger>
                                                        <SelectContent className="bg-white border-gray-200 text-gray-900">
                                                            <SelectItem value="kmeans++">K-Means++</SelectItem>
                                                            <SelectItem value="random">Random</SelectItem>
                                                        </SelectContent>
                                                    </Select>
                                                </div>
                                            </>
                                        )}

                                        {activeTab === "nb" && (
                                            <div className="space-y-3">
                                                <div className="flex justify-between">
                                                    <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Train Ratio</label>
                                                    <span className="text-xs text-violet-600 font-mono font-bold">{trainRatio}%</span>
                                                </div>
                                                <Slider value={[trainRatio]} onValueChange={(v) => setTrainRatio(v[0])} min={50} max={90} step={5} className="py-4" />
                                            </div>
                                        )}

                                        {activeTab === "pca" && (
                                            <div className="space-y-3">
                                                <div className="flex justify-between">
                                                    <label className="text-xs font-bold text-gray-500 uppercase tracking-widest">Components</label>
                                                    <span className="text-xs text-emerald-600 font-mono font-bold">{nComponents}</span>
                                                </div>
                                                <Slider value={[nComponents]} onValueChange={(v) => setNComponents(v[0])} min={1} max={4} step={1} className="py-4" />
                                            </div>
                                        )}

                                        <Button
                                            onClick={result ? resetAnalysis : runAlgorithm}
                                            disabled={isRunning}
                                            className="w-full bg-[#004040] hover:bg-[#003333] text-white h-12"
                                        >
                                            {isRunning ? "Processing..." : result ? "Reset" : "Run Algorithm"}
                                            {!isRunning && !result && <Play className="ml-2 w-4 h-4 fill-current" />}
                                            {result && <RefreshCw className="ml-2 w-4 h-4" />}
                                        </Button>

                                        {result && (
                                            <Button
                                                onClick={exportResults}
                                                variant="outline"
                                                className="w-full border-gray-300 text-gray-700 hover:bg-gray-100"
                                            >
                                                <Download className="w-4 h-4 mr-2" />
                                                Export Results
                                            </Button>
                                        )}
                                    </div>

                                    <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 relative overflow-hidden flex flex-col justify-center min-h-[300px]">
                                        {!isRunning && !result && !error && (
                                            <div className="flex flex-col items-center text-center space-y-4">
                                                <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
                                                    <Database className="w-8 h-8" />
                                                </div>
                                                <p className="text-sm text-gray-500 max-w-[200px]">
                                                    Ready to analyze. Configure settings and click Run.
                                                </p>
                                            </div>
                                        )}

                                        {isRunning && (
                                            <div className="space-y-6">
                                                <div className="flex flex-col items-center">
                                                    <div className="text-sm text-gray-600 mb-2">Processing {activeTab === "kmeans" ? "K-Means" : activeTab === "nb" ? "Naive Bayes" : "PCA"}...</div>
                                                    <Progress value={progress} className="h-2 w-full bg-gray-200" />
                                                </div>
                                            </div>
                                        )}

                                        {error && (
                                            <div className="flex flex-col items-center text-center space-y-4">
                                                <div className="w-16 h-16 rounded-full bg-rose-50 flex items-center justify-center text-rose-500">
                                                    <AlertCircle className="w-8 h-8" />
                                                </div>
                                                <p className="text-sm text-rose-600">{error}</p>
                                            </div>
                                        )}

                                        {result && activeTab === "kmeans" && renderKMeansResults(result as KMeansResult)}
                                        {result && activeTab === "nb" && renderNaiveBayesResults(result as NaiveBayesResult)}
                                        {result && activeTab === "pca" && renderPCAResults(result as PCAResult)}

                                        {result && <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 blur-3xl -z-10" />}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </section>
    );
};
