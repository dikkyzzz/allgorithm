import { algorithms } from '@/lib/algorithms';
import { algorithmDetails } from '@/lib/algorithm-details';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, BookOpen, Code, CheckCircle2, XCircle, Settings, Briefcase, ArrowRight } from 'lucide-react';
import { AlgorithmDemo } from '@/components/AlgorithmDemo';
import { DevToReferences } from '@/components/DevToReferences';
import { Header } from '@/components/landing/Header';
import { SortingVisualizer } from '@/components/SortingVisualizer';
import { SearchVisualizer } from '@/components/SearchVisualizer';
import { StatsVisualizer } from '@/components/StatsVisualizer';
import { ClusteringVisualizer } from '@/components/ClusteringVisualizer';

interface PageProps {
    params: Promise<{ id: string }>;
}

export async function generateStaticParams() {
    return algorithms.map((algo) => ({ id: algo.id }));
}

export default async function AlgorithmPage({ params }: PageProps) {
    const { id } = await params;
    const algorithm = algorithms.find((a) => a.id === id);
    const details = algorithmDetails[id];

    if (!algorithm) {
        notFound();
    }

    const categoryColors: Record<string, string> = {
        'Clustering': 'bg-blue-50 text-blue-700 border-blue-200',
        'Classification': 'bg-violet-50 text-violet-700 border-violet-200',
        'Regression': 'bg-emerald-50 text-emerald-700 border-emerald-200',
        'Dimensionality Reduction': 'bg-amber-50 text-amber-700 border-amber-200',
        'Text Mining': 'bg-rose-50 text-rose-700 border-rose-200',
        'Sorting': 'bg-orange-50 text-orange-700 border-orange-200',
        'Searching': 'bg-cyan-50 text-cyan-700 border-cyan-200',
        'Graph': 'bg-indigo-50 text-indigo-700 border-indigo-200',
        'Deep Learning': 'bg-purple-50 text-purple-700 border-purple-200',
        'Statistics': 'bg-teal-50 text-teal-700 border-teal-200',
        'Security': 'bg-red-50 text-red-700 border-red-200',
        'Recommendation': 'bg-pink-50 text-pink-700 border-pink-200',
    };

    const isSortingAlgorithm = ['bubble-sort', 'quick-sort', 'merge-sort', 'heap-sort', 'insertion-sort'].includes(id);
    const isSearchAlgorithm = ['binary-search', 'linear-search'].includes(id);
    const isStatsAlgorithm = ['descriptive-stats', 'standard-deviation', 'correlation'].includes(id);
    const isClusteringAlgorithm = ['k-means', 'dbscan', 'hierarchical-clustering'].includes(id);




    return (
        <div className="min-h-screen bg-gray-50">
            {/* Navbar */}
            <Header />

            {/* Algorithm Header */}
            <div className="bg-white border-b border-gray-200 pt-24">
                <div className="container max-w-5xl mx-auto px-4 py-12">

                    <div className="mb-4">
                        <Badge className={categoryColors[algorithm.category]}>
                            {algorithm.category}
                        </Badge>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-3">
                        {details?.fullName || algorithm.name}
                    </h1>
                    <p className="text-xl text-gray-600 mb-6">{details?.tagline || algorithm.description}</p>

                    <div className="flex flex-wrap gap-3">
                        <Link href={`/docs/${id}`}>
                            <Button variant="outline" className="border-gray-300 text-gray-700 hover:bg-gray-100">
                                <BookOpen className="w-4 h-4 mr-2" /> Full Documentation
                            </Button>
                        </Link>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="container max-w-5xl mx-auto px-4 py-12">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Main Content */}
                    <div className="lg:col-span-2 space-y-8">
                        {/* Overview */}
                        <Card className="bg-white border-gray-200 shadow-sm">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-lg text-gray-900">
                                    <BookOpen className="w-5 h-5 text-[#004040]" />
                                    Overview
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-gray-600 leading-relaxed">
                                    {details?.overview || algorithm.description}
                                </p>
                            </CardContent>
                        </Card>

                        {/* Interactive Visualizers */}
                        {isSortingAlgorithm && (
                            <SortingVisualizer algorithmId={id} />
                        )}
                        {isSearchAlgorithm && (
                            <SearchVisualizer algorithmId={id} />
                        )}
                        {isStatsAlgorithm && (
                            <StatsVisualizer algorithmId={id} />
                        )}
                        {isClusteringAlgorithm && (
                            <ClusteringVisualizer algorithmId={id} />
                        )}

                        {/* How It Works */}
                        {details?.howItWorks && (
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg text-gray-900">
                                        <Code className="w-5 h-5 text-[#004040]" />
                                        How It Works
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-6">
                                    <ol className="space-y-3">
                                        {details.howItWorks.steps.map((step, i) => (
                                            <li key={i} className="flex gap-3">
                                                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-[#004040] text-white text-xs flex items-center justify-center font-bold">
                                                    {i + 1}
                                                </span>
                                                <span className="text-gray-600">{step}</span>
                                            </li>
                                        ))}
                                    </ol>
                                    {details.howItWorks.formula && (
                                        <div className="p-4 bg-gray-100 rounded-lg border border-gray-200">
                                            <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Formula</div>
                                            <code className="text-[#004040] font-mono text-lg">{details.howItWorks.formula}</code>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        )}

                        {/* Pros & Cons */}
                        {details?.prosAndCons && (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-5">
                                    <div className="flex items-center gap-2 text-base font-semibold text-emerald-700 mb-3">
                                        <CheckCircle2 className="w-5 h-5" />
                                        Pros
                                    </div>
                                    <ul className="space-y-2">
                                        {details.prosAndCons.pros.map((pro, i) => (
                                            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                                <span className="text-emerald-500 mt-0.5">✓</span>
                                                {pro}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-5">
                                    <div className="flex items-center gap-2 text-base font-semibold text-rose-700 mb-3">
                                        <XCircle className="w-5 h-5" />
                                        Cons
                                    </div>
                                    <ul className="space-y-2">
                                        {details.prosAndCons.cons.map((con, i) => (
                                            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                                <span className="text-rose-500 mt-0.5">✗</span>
                                                {con}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        )}

                        {/* Use Cases */}
                        {details?.useCases && (
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg text-gray-900">
                                        <Briefcase className="w-5 h-5 text-[#004040]" />
                                        Real-World Use Cases
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        {details.useCases.map((useCase, i) => (
                                            <div key={i} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                                                <h4 className="font-semibold text-gray-900 mb-2">{useCase.title}</h4>
                                                <p className="text-sm text-gray-600">{useCase.description}</p>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Code Example */}
                        {details?.codeExample && (
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg text-gray-900">
                                        <Code className="w-5 h-5 text-[#004040]" />
                                        Code Example
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <pre className="p-4 bg-gray-900 text-gray-100 rounded-lg overflow-x-auto text-sm font-mono">
                                        <code>{details.codeExample}</code>
                                    </pre>
                                </CardContent>
                            </Card>
                        )}
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* Live Demo */}
                        <AlgorithmDemo algorithmId={id} />

                        {/* Requirements */}
                        <Card className="bg-white border-gray-200 shadow-sm">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-base text-gray-900">Requirements</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="flex flex-wrap gap-2">
                                    {algorithm.requirements.map((req) => (
                                        <span key={req} className="px-3 py-1.5 rounded-full bg-gray-100 text-gray-700 text-xs border border-gray-200">
                                            {req}
                                        </span>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Hyperparameters */}
                        {details?.hyperparameters && (
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader className="pb-3">
                                    <CardTitle className="flex items-center gap-2 text-base text-gray-900">
                                        <Settings className="w-4 h-4" />
                                        Hyperparameters
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    {details.hyperparameters.map((param, i) => (
                                        <div key={i} className="space-y-1">
                                            <div className="font-mono text-sm text-[#004040] font-semibold">{param.name}</div>
                                            <div className="text-xs text-gray-600">{param.description}</div>
                                            <div className="text-xs text-gray-400">Typical: {param.typical}</div>
                                        </div>
                                    ))}
                                </CardContent>
                            </Card>
                        )}

                        {/* Related Algorithms */}
                        {details?.relatedAlgorithms && (
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader className="pb-3">
                                    <CardTitle className="text-base text-gray-900">Related Algorithms</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-2">
                                    {details.relatedAlgorithms.map((relId) => {
                                        const relAlgo = algorithms.find(a => a.id === relId);
                                        if (!relAlgo) return null;
                                        return (
                                            <Link key={relId} href={`/algorithms/${relId}`} className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-100 transition-colors group">
                                                <span className="text-sm text-gray-700 group-hover:text-[#004040]">{relAlgo.name}</span>
                                                <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-[#004040]" />
                                            </Link>
                                        );
                                    })}
                                </CardContent>
                            </Card>
                        )}

                        {/* Dev.to References */}
                        <DevToReferences algorithmId={id} algorithmName={algorithm.name} />
                    </div>
                </div>
            </div>
        </div>
    );
}
